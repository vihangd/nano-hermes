"""ACE delta-playbook over the principles store.

P1: embedding dedup on write — a near-duplicate principle merges into the
existing row (deterministic by-id update) instead of accumulating duplicates.
"""
from __future__ import annotations

import time

import nano_hermes
from conftest import _patch_embedding, _unset_embedding_keys


def _count(db) -> int:
    return db.execute("SELECT COUNT(*) FROM principles").fetchone()[0]


def _add_principle(
    db, condition, action, *, success=0, harmful=0, age_days=0.0, origin="agent"
) -> int:
    now = time.time()
    cur = db.execute(
        "INSERT INTO principles (condition, action, expected_outcome, confidence, "
        "created_at, updated_at, origin, success_count, harmful_count) "
        "VALUES (?, ?, NULL, 0.5, ?, ?, ?, ?, ?)",
        (condition, action, now - age_days * 86400, now, origin, success, harmful),
    )
    pid = int(cur.lastrowid)
    db.execute(
        "INSERT INTO principles_fts (rowid, condition, action, content_id) VALUES (?, ?, ?, ?)",
        (pid, condition, action, pid),
    )
    db.commit()
    return pid


def _counts(db, pid) -> tuple[int, int]:
    return db.execute(
        "SELECT success_count, harmful_count FROM principles WHERE id = ?", (pid,)
    ).fetchone()


class TestDedupOnWrite:
    async def test_near_duplicate_merges(self, loop, monkeypatch):
        # Fake embedder maps any text containing 'duckduckgo' to one vector,
        # so these two principles are cosine-identical -> merge.
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("record_principle")

        out1 = await tool.execute(
            condition="when searching duckduckgo",
            action="prefer the lite endpoint",
        )
        out2 = await tool.execute(
            condition="while using duckduckgo search",
            action="use the lite endpoint and set a UA",
        )
        assert out1.startswith("ok"), out1
        assert "merged" in out2, out2
        assert _count(hook.db) == 1
        # The surviving row kept the latest action + one vec row.
        action = hook.db.execute(
            "SELECT action FROM principles"
        ).fetchone()[0]
        assert "set a UA" in action
        assert hook.db.execute("SELECT COUNT(*) FROM principles_vec").fetchone()[0] == 1

    async def test_distinct_principles_kept_separate(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("record_principle")

        await tool.execute(condition="when searching duckduckgo", action="use lite")
        await tool.execute(condition="when searching arxiv", action="use the api")
        assert _count(hook.db) == 2  # orthogonal vectors -> no merge

    async def test_manual_principle_is_agent_origin(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        await loop.tools.get("record_principle").execute(
            condition="when searching duckduckgo", action="use lite"
        )
        origin = hook.db.execute("SELECT origin FROM principles").fetchone()[0]
        assert origin == "agent"

    async def test_no_embedding_still_records_fts_only(self, loop, monkeypatch):
        """Providers down: principle is stored (FTS-only), no vec row, and
        without an embedding it cannot be deduped — two identical ones persist."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("record_principle")

        out = await tool.execute(condition="when X", action="do Y")
        assert out.startswith("ok"), out
        await tool.execute(condition="when X", action="do Y")
        assert _count(hook.db) == 2
        assert hook.db.execute("SELECT COUNT(*) FROM principles_vec").fetchone()[0] == 0


class TestAttribution:
    def test_ok_credits_helpful_fail_credits_harmful(self, loop):
        hook = nano_hermes.install(loop)
        rc = hook._reflection_coord
        pid = _add_principle(hook.db, "when deploying", "check the config")

        rc._injected_principle_ids = {pid}
        rc.attribute_principles("ok")
        assert _counts(hook.db, pid) == (1, 0)
        assert rc._injected_principle_ids == set()  # cleared

        rc._injected_principle_ids = {pid}
        rc.attribute_principles("fail")
        assert _counts(hook.db, pid) == (1, 1)

    def test_partial_is_neutral(self, loop):
        hook = nano_hermes.install(loop)
        rc = hook._reflection_coord
        pid = _add_principle(hook.db, "when deploying", "check the config")
        rc._injected_principle_ids = {pid}
        rc.attribute_principles("partial")
        assert _counts(hook.db, pid) == (0, 0)
        assert rc._injected_principle_ids == set()


class TestInjectionRanking:
    def test_proven_recent_beats_harmful_stale(self, loop):
        hook = nano_hermes.install(loop, config={"principles": {"enabled": True}})
        rc = hook._reflection_coord
        # Both match the FTS term 'deploy'; ranking must pick the proven one.
        _add_principle(hook.db, "when deploy fails", "rollback first", success=5, harmful=0)
        _add_principle(
            hook.db, "when deploy starts", "skip the smoke test",
            success=0, harmful=5, age_days=120,
        )
        msgs = [{"role": "user", "content": "help me deploy the service"}]
        out = rc.get_principle_injections(msgs, limit=1)
        assert len(out) == 1
        assert "rollback first" in out[0]["content"]
        assert "skip the smoke test" not in out[0]["content"]
        # The injected principle is tracked for attribution.
        assert len(rc._injected_principle_ids) == 1


# --- P3: curator delta loop -------------------------------------------------
from unittest.mock import AsyncMock, MagicMock  # noqa: E402

from nano_hermes.skills.principle_curator import (  # noqa: E402
    _prune_over_budget,
    apply_ops,
    parse_ops,
    run_principle_curator,
)


class TestParseOps:
    def test_valid(self):
        assert parse_ops('{"ops":[{"op":"add","condition":"c","action":"a"}]}') == [
            {"op": "add", "condition": "c", "action": "a"}
        ]

    def test_code_fenced(self):
        assert parse_ops('```json\n{"ops":[{"op":"prune","id":3}]}\n```') == [
            {"op": "prune", "id": 3}
        ]

    def test_garbage_is_empty(self):
        assert parse_ops("not json at all") == []
        assert parse_ops('{"nope": 1}') == []

    def test_unknown_op_filtered(self):
        assert parse_ops(
            '{"ops":[{"op":"nuke"},{"op":"add","condition":"c","action":"a"}]}'
        ) == [{"op": "add", "condition": "c", "action": "a"}]


class TestApplyOps:
    async def test_add_creates_curator_principle(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop, config={"principles": {"enabled": True}})
        counts = await apply_ops(
            hook,
            [{"op": "add", "condition": "when deploy", "action": "rollback"}],
            hook.config.principles,
        )
        assert counts["added"] == 1
        assert hook.db.execute("SELECT origin FROM principles").fetchone()[0] == "curator"

    async def test_add_does_not_overwrite_manual(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop, config={"principles": {"enabled": True}})
        # Manual principle (embedded via the tool) on the 'deploy' vector.
        await loop.tools.get("record_principle").execute(
            condition="when deploy fails", action="manual action"
        )
        counts = await apply_ops(
            hook,
            [{"op": "add", "condition": "when deploy starts", "action": "curator action"}],
            hook.config.principles,
        )
        assert counts["skipped"] == 1
        assert _count(hook.db) == 1
        assert hook.db.execute("SELECT action FROM principles").fetchone()[0] == "manual action"

    async def test_add_does_not_overwrite_pinned_curator(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop, config={"principles": {"enabled": True}})
        # Pinned curator-origin principle on the 'deploy' vector.
        await loop.tools.get("record_principle").execute(
            condition="when deploy fails", action="pinned action"
        )
        hook.db.execute("UPDATE principles SET origin='curator', pinned=1")
        hook.db.commit()
        counts = await apply_ops(
            hook,
            [{"op": "add", "condition": "when deploy starts", "action": "curator action"}],
            hook.config.principles,
        )
        assert counts["skipped"] == 1
        assert hook.db.execute("SELECT action FROM principles").fetchone()[0] == "pinned action"

    async def test_update_and_prune_only_touch_curator_rows(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop, config={"principles": {"enabled": True}})
        manual = _add_principle(hook.db, "when a", "manual", origin="agent")
        auto = _add_principle(hook.db, "when b", "auto", origin="curator")

        await apply_ops(hook, [{"op": "update", "id": auto, "action": "auto-v2"}], hook.config.principles)
        await apply_ops(hook, [{"op": "update", "id": manual, "action": "hacked"}], hook.config.principles)
        assert hook.db.execute("SELECT action FROM principles WHERE id=?", (auto,)).fetchone()[0] == "auto-v2"
        assert hook.db.execute("SELECT action FROM principles WHERE id=?", (manual,)).fetchone()[0] == "manual"

        await apply_ops(hook, [{"op": "prune", "id": manual}], hook.config.principles)
        await apply_ops(hook, [{"op": "prune", "id": auto}], hook.config.principles)
        names = {r[0] for r in hook.db.execute("SELECT action FROM principles").fetchall()}
        assert names == {"manual"}  # manual survives prune, curator row gone

    def test_prune_over_budget_spares_manual(self, loop):
        hook = nano_hermes.install(loop)
        _add_principle(hook.db, "m", "manual", origin="agent")
        for i in range(4):
            _add_principle(hook.db, f"c{i}", f"auto{i}", origin="curator", success=i)
        removed = _prune_over_budget(hook.db, max_principles=2)
        assert removed == 3  # 5 total -> 2; only curator rows dropped
        assert _count(hook.db) == 2
        assert hook.db.execute(
            "SELECT COUNT(*) FROM principles WHERE origin='agent'"
        ).fetchone()[0] == 1


class TestRunCurator:
    async def test_disabled_returns_empty(self, loop):
        hook = nano_hermes.install(loop)
        assert await run_principle_curator(hook) == {}

    async def test_no_failures_marks_run(self, loop):
        hook = nano_hermes.install(
            loop, config={"principles": {"enabled": True, "cooldown_hours": 0}}
        )
        assert await run_principle_curator(hook) == {}

    async def test_applies_llm_ops(self, loop, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(
            loop, config={"principles": {"enabled": True, "cooldown_hours": 0}}
        )
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES ('deploy broke', 'fail', ?)",
            (time.time(),),
        )
        hook.db.commit()
        resp = MagicMock()
        resp.content = '{"ops":[{"op":"add","condition":"when deploy","action":"rollback first"}]}'
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=resp)

        counts = await run_principle_curator(hook)
        assert counts.get("added") == 1
        assert hook.db.execute(
            "SELECT COUNT(*) FROM principles WHERE origin='curator'"
        ).fetchone()[0] == 1
