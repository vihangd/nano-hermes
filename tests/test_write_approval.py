"""Write-approval gate: staging, conflict-detection, approve/reject replay.

Covers the autonomous write paths (rewriter, umbrella, curator) under
``write_approval == "approve"`` plus the offline approve/reject replay and the
anti-clobber stale-base invariant.
"""
from __future__ import annotations

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.governance import write_approval as wa
from nano_hermes.skills.principle_curator import run_principle_curator
from nano_hermes.skills.rewriter import run_rewriter
from nano_hermes.skills.umbrella import run_umbrella_merge

DIMS = 512


def _unit(axis: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[axis] = 1.0
    return v


def _critic_approved():
    return patch("nano_hermes.skills.rewriter._run_critic", new=AsyncMock(return_value=True))


def _seed_skill_vec(hook, name, vec, *, status="active", origin="agent", pinned=0):
    cur = hook.db.execute(
        "INSERT INTO skill_stats (name, status, origin, pinned) VALUES (?, ?, ?, ?)",
        (name, status, origin, pinned),
    )
    hook.db.execute(
        "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, vec.astype(np.float32).tobytes()),
    )
    hook.db.commit()
    d = hook.workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: does {name}\n---\n\nbody of {name}\n"
    )


def _seed_failing_skill(hook, name):
    hook.db.execute(
        "INSERT OR REPLACE INTO skill_stats "
        "(name, status, use_count, success_count, origin) VALUES (?, 'active', 10, 1, 'agent')",
        (name,),
    )
    hook.db.commit()
    d = hook.workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"# {name}\n\nbad skill\n")
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s_{name}", time.time()),
    )
    sid = cur.lastrowid
    hook.db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', 'could not complete', ?)",
        (sid, time.time()),
    )
    hook.db.execute(
        "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
        "VALUES (?, 'do x', ?, 'fail', ?)",
        (sid, json.dumps([name]), time.time()),
    )
    hook.db.commit()


def _mock_llm(hook, payload: str):
    resp = MagicMock()
    resp.content = payload
    hook._loop.provider = MagicMock()
    hook._loop.provider.chat_with_retry = AsyncMock(return_value=resp)


def _pending(hook, subsystem=None):
    return wa.list_pending(hook.db, subsystem)


# --------------------------------------------------------------------------- #
# Gate off — regression: writes commit as before
# --------------------------------------------------------------------------- #
class TestGateOff:
    def test_off_is_default_and_commits(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path))
        assert wa.is_gated(hook, "skills") is False
        assert wa.is_gated(hook, "principles") is False

    def test_rewriter_commits_when_off(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"write_approval": "off"}},
        )
        _seed_failing_skill(hook, "bad")
        _mock_llm(hook, "# bad\n\nImproved body now.\n")
        with _critic_approved():
            rewritten = asyncio.run(run_rewriter(hook))
        assert "bad" in rewritten
        assert _pending(hook) == []
        assert "Improved body" in (hook.workspace / "skills" / "bad" / "SKILL.md").read_text()


# --------------------------------------------------------------------------- #
# Staging — autonomous writes are held, not committed
# --------------------------------------------------------------------------- #
class TestStaging:
    def test_rewriter_stages_and_excludes_from_evolved(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"write_approval": "approve"}},
        )
        _seed_failing_skill(hook, "bad")
        original = (hook.workspace / "skills" / "bad" / "SKILL.md").read_text()
        _mock_llm(hook, "# bad\n\nImproved body now.\n")
        with _critic_approved():
            rewritten = asyncio.run(run_rewriter(hook))
        # not reported as evolved (keeps skip= chaining honest)
        assert rewritten == []
        # live file untouched
        assert (hook.workspace / "skills" / "bad" / "SKILL.md").read_text() == original
        # one pending skill row
        rows = _pending(hook, "skills")
        assert len(rows) == 1 and rows[0]["origin"] == "rewriter"

    async def test_umbrella_stages_without_deprecating_siblings(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {
                "umbrella_merge_enabled": True, "write_approval": "approve"
            }},
        )
        _seed_skill_vec(hook, "web-fetch", _unit(0))
        _seed_skill_vec(hook, "web-get", _unit(0))
        _mock_llm(
            hook,
            '{"name": "web-tools", "description": "web", "body": "## merged",'
            ' "absorbed": ["web-fetch", "web-get"]}',
        )
        merged = await run_umbrella_merge(hook)
        assert merged == []  # not merged — staged
        assert not (hook.workspace / "skills" / "web-tools" / "SKILL.md").exists()
        # siblings still active
        for sib in ("web-fetch", "web-get"):
            assert hook.db.execute(
                "SELECT status FROM skill_stats WHERE name=?", (sib,)
            ).fetchone()[0] == "active"
        rows = _pending(hook, "skills")
        assert len(rows) == 1 and rows[0]["origin"] == "umbrella"

    async def test_curator_stages_ops_without_touching_table(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"principles": {"enabled": True, "write_approval": "approve"}},
        )
        _seed_failing_skill(hook, "bad")  # produces a failed trajectory
        _mock_llm(
            hook,
            '{"ops": [{"op": "add", "condition": "when X", "action": "do Y"}]}',
        )
        counts = await run_principle_curator(hook)
        assert counts == {"staged": 1}
        # principles table untouched
        assert hook.db.execute("SELECT COUNT(*) FROM principles").fetchone()[0] == 0
        rows = _pending(hook, "principles")
        assert len(rows) == 1 and rows[0]["origin"] == "curator"


# --------------------------------------------------------------------------- #
# Approve / reject — skill replay + anti-clobber
# --------------------------------------------------------------------------- #
class TestApproveSkill:
    def _stage_one(self, hook, name="bad", body="new body text"):
        _seed_failing_skill(hook, name)
        return wa.stage_skill_write(
            hook, skill_name=name, description="", body=body,
            reason="test", origin="rewriter",
        )

    def test_approve_clean_writes_and_snapshots(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"write_approval": "approve"}},
        )
        pid = self._stage_one(hook, body="approved body here")
        out = wa.approve_skill(hook.db, hook.workspace, pid)
        assert out.startswith("approved")
        assert "approved body here" in (hook.workspace / "skills" / "bad" / "SKILL.md").read_text()
        assert wa.get_pending(hook.db, pid)["status"] == "approved"
        # approve-time snapshot taken
        from nano_hermes.skills.evolution_snapshot import latest_snapshot
        assert latest_snapshot(hook.workspace) is not None

    def test_conflict_refuses_and_does_not_clobber(self, tmp_path):
        # Anti-clobber invariant (mutation target): stage, mutate underneath,
        # approve must refuse rather than overwrite the newer content.
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"write_approval": "approve"}},
        )
        pid = self._stage_one(hook, body="stale proposal")
        path = hook.workspace / "skills" / "bad" / "SKILL.md"
        path.write_text("MANUALLY EDITED SINCE STAGING")
        out = wa.approve_skill(hook.db, hook.workspace, pid)
        assert "refused" in out and "stale" in out
        assert path.read_text() == "MANUALLY EDITED SINCE STAGING"  # not clobbered
        assert wa.get_pending(hook.db, pid)["status"] == "stale"

    def test_reject_leaves_store_untouched(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"write_approval": "approve"}},
        )
        pid = self._stage_one(hook, body="rejected body")
        original = (hook.workspace / "skills" / "bad" / "SKILL.md").read_text()
        out = wa.reject(hook.db, pid)
        assert out.startswith("rejected")
        assert (hook.workspace / "skills" / "bad" / "SKILL.md").read_text() == original
        assert wa.get_pending(hook.db, pid)["status"] == "rejected"


# --------------------------------------------------------------------------- #
# Approve — umbrella replay (write + sibling deprecation) and curator ops
# --------------------------------------------------------------------------- #
class TestApproveUmbrella:
    async def test_approve_writes_umbrella_and_deprecates_siblings(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {
                "umbrella_merge_enabled": True, "write_approval": "approve"
            }},
        )
        _seed_skill_vec(hook, "web-fetch", _unit(0))
        _seed_skill_vec(hook, "web-get", _unit(0))
        _mock_llm(
            hook,
            '{"name": "web-tools", "description": "web", "body": "## merged",'
            ' "absorbed": ["web-fetch", "web-get"]}',
        )
        await run_umbrella_merge(hook)
        pid = _pending(hook, "skills")[0]["id"]
        out = wa.approve_skill(hook.db, hook.workspace, pid)
        assert out.startswith("approved")
        assert (hook.workspace / "skills" / "web-tools" / "SKILL.md").exists()
        assert hook.db.execute(
            "SELECT status, origin FROM skill_stats WHERE name='web-tools'"
        ).fetchone() == ("active", "agent")
        for sib in ("web-fetch", "web-get"):
            assert hook.db.execute(
                "SELECT status FROM skill_stats WHERE name=?", (sib,)
            ).fetchone()[0] == "deprecated"

    async def test_approve_refused_when_sibling_changed(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {
                "umbrella_merge_enabled": True, "write_approval": "approve"
            }},
        )
        _seed_skill_vec(hook, "web-fetch", _unit(0))
        _seed_skill_vec(hook, "web-get", _unit(0))
        _mock_llm(
            hook,
            '{"name": "web-tools", "description": "web", "body": "## merged",'
            ' "absorbed": ["web-fetch", "web-get"]}',
        )
        await run_umbrella_merge(hook)
        pid = _pending(hook, "skills")[0]["id"]
        # mutate a sibling after staging
        (hook.workspace / "skills" / "web-fetch" / "SKILL.md").write_text("CHANGED")
        out = wa.approve_skill(hook.db, hook.workspace, pid)
        assert "refused" in out
        assert not (hook.workspace / "skills" / "web-tools" / "SKILL.md").exists()
        assert hook.db.execute(
            "SELECT status FROM skill_stats WHERE name='web-get'"
        ).fetchone()[0] == "active"


class TestApprovePrinciples:
    async def test_approve_replays_ops_via_apply_ops(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"principles": {"enabled": True, "write_approval": "approve"}},
        )
        ops = [{"op": "add", "condition": "when X", "action": "do Y"}]
        pid = wa.stage_principle_ops(hook, ops=ops, reason="test")
        applied = AsyncMock(return_value={"added": 1})
        with patch("nano_hermes.skills.principle_curator.apply_ops", applied):
            out = await wa.approve_principles(hook, pid)
        applied.assert_awaited_once()
        # ops passed through verbatim
        assert applied.await_args.args[1] == ops
        assert out.startswith("approved")
        assert wa.get_pending(hook.db, pid)["status"] == "approved"
