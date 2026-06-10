"""Skill umbrella consolidation — clustering + LLM merge + sibling deprecation."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.umbrella import find_merge_clusters, run_umbrella_merge

DIMS = 512


def _unit(axis: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[axis] = 1.0
    return v


def _make_hook(loop):
    return nano_hermes.install(
        loop,
        config={"skill_stats": {"umbrella_merge_enabled": True, "umbrella_max_merges_per_run": 5}},
    )


def _seed(hook, name, vec, *, status="active", origin="agent", pinned=0) -> None:
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


def _status(hook, name) -> str:
    return hook.db.execute(
        "SELECT status FROM skill_stats WHERE name = ?", (name,)
    ).fetchone()[0]


def _mock_merge(hook, payload: str) -> None:
    resp = MagicMock()
    resp.content = payload
    hook._loop.provider.chat_with_retry = AsyncMock(return_value=resp)


class TestClustering:
    def test_groups_near_duplicates_only(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "web-fetch", _unit(0))
        _seed(hook, "web-get", _unit(0))     # identical vec -> same cluster
        _seed(hook, "math-add", _unit(1))    # orthogonal -> alone
        clusters = find_merge_clusters(
            hook.db, sim_threshold=0.86, min_cluster=2, max_cluster=5
        )
        assert len(clusters) == 1
        assert set(clusters[0]) == {"web-fetch", "web-get"}

    def test_excludes_user_and_pinned(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "agent-a", _unit(0))
        _seed(hook, "user-b", _unit(0), origin="user")   # excluded
        _seed(hook, "pinned-c", _unit(0), pinned=1)       # excluded
        clusters = find_merge_clusters(
            hook.db, sim_threshold=0.86, min_cluster=2, max_cluster=5
        )
        assert clusters == []  # only one eligible candidate left


class TestMerge:
    async def test_disabled_returns_empty(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)  # umbrella off by default
        _seed(hook, "a", _unit(0))
        _seed(hook, "b", _unit(0))
        assert await run_umbrella_merge(hook) == []

    async def test_merges_cluster_and_deprecates_siblings(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "web-fetch", _unit(0))
        _seed(hook, "web-get", _unit(0))
        _mock_merge(
            hook,
            '{"name": "web-tools", "description": "web access",'
            ' "body": "## fetch\\n## get", "absorbed": ["web-fetch", "web-get"]}',
        )
        merged = await run_umbrella_merge(hook)
        assert merged == ["web-tools"]
        # umbrella written + agent-origin active
        row = hook.db.execute(
            "SELECT status, origin FROM skill_stats WHERE name = 'web-tools'"
        ).fetchone()
        assert row == ("active", "agent")
        assert (hook.workspace / "skills" / "web-tools" / "SKILL.md").exists()
        # siblings deprecated with audit trail
        assert _status(hook, "web-fetch") == "deprecated"
        assert _status(hook, "web-get") == "deprecated"
        reason = hook.db.execute(
            "SELECT reason FROM skill_versions WHERE skill_name = 'web-fetch'"
        ).fetchone()[0]
        assert reason == "absorbed_into: web-tools"

    async def test_rejects_absorbed_not_subset(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "web-fetch", _unit(0))
        _seed(hook, "web-get", _unit(0))
        # absorbed references a non-candidate -> filtered to <2 -> rejected
        _mock_merge(
            hook,
            '{"name": "web-tools", "description": "x", "body": "y",'
            ' "absorbed": ["web-fetch", "ghost"]}',
        )
        assert await run_umbrella_merge(hook) == []
        assert _status(hook, "web-fetch") == "active"  # untouched

    async def test_malformed_response_is_noop(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "web-fetch", _unit(0))
        _seed(hook, "web-get", _unit(0))
        _mock_merge(hook, "not json")
        assert await run_umbrella_merge(hook) == []
        assert _status(hook, "web-fetch") == "active"

    async def test_rejects_name_colliding_with_out_of_cluster_skill(self, tmp_path):
        hook = _make_hook(_make_loop(tmp_path))
        _seed(hook, "web-fetch", _unit(0))
        _seed(hook, "web-get", _unit(0))
        # A pinned, user-origin skill outside the cluster owns this name.
        _seed(hook, "important", _unit(1), origin="user", pinned=1)
        (hook.workspace / "skills" / "important" / "SKILL.md").write_text("PRECIOUS")
        _mock_merge(
            hook,
            '{"name": "important", "description": "x", "body": "y",'
            ' "absorbed": ["web-fetch", "web-get"]}',
        )
        assert await run_umbrella_merge(hook) == []
        # The out-of-cluster skill is untouched (file + origin + pin).
        assert (hook.workspace / "skills" / "important" / "SKILL.md").read_text() == "PRECIOUS"
        row = hook.db.execute(
            "SELECT origin, pinned FROM skill_stats WHERE name = 'important'"
        ).fetchone()
        assert row == ("user", 1)
        assert _status(hook, "web-fetch") == "active"  # siblings not deprecated
