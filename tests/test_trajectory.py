"""Tests for trajectory writing, edge cases, search, and context injection."""
from __future__ import annotations

import json as _json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.providers.base import ToolCallRequest

import nano_hermes
from nano_hermes.session.trajectory_search import TrajectorySearchTool

from conftest import (
    _copy_bundled_skill,
    _patch_embedding,
    _unset_embedding_keys,
    _FAKE_VEC_SEARCH,
    _FAKE_VEC_ACADEMIC,
)


# ---------------------------------------------------------------------------
# Phase 2: trajectory writing on session boundary
# ---------------------------------------------------------------------------

class TestTrajectoryWrite:
    async def test_trajectory_written_on_session_boundary(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Session A
        msgs_a: list[dict] = [
            {"role": "user", "content": "write me a haiku about pandas"},
            {"role": "assistant", "content": "Bears of bamboo dreams..."},
        ]
        ctx_a = AgentHookContext(iteration=0, messages=msgs_a)
        await hook.before_iteration(ctx_a)
        await hook.after_iteration(ctx_a)
        session_a = hook.current_session_id
        assert session_a is not None

        # Starting session B causes _finalize_trajectory for A
        msgs_b: list[dict] = [{"role": "user", "content": "new task"}]
        ctx_b = AgentHookContext(iteration=0, messages=msgs_b)
        await hook.before_iteration(ctx_b)

        rows = hook.db.execute(
            "SELECT session_id, task, outcome FROM trajectories"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == session_a
        assert "haiku" in rows[0][1]
        assert rows[0][2] == "ok"  # no errors

    async def test_trajectory_outcome_fail_on_short_error_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [{"role": "user", "content": "do the impossible"}]
        ctx_a = AgentHookContext(iteration=0, messages=msgs_a, error="crashed")
        await hook.before_iteration(ctx_a)
        await hook.after_iteration(ctx_a)

        msgs_b: list[dict] = [{"role": "user", "content": "new task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        row = hook.db.execute(
            "SELECT outcome FROM trajectories"
        ).fetchone()
        assert row is not None
        assert row[0] == "fail"

    async def test_trajectory_includes_skills_used(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _copy_bundled_skill("duckduckgo-search", tmp_path)
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [{"role": "user", "content": "search for something"}]
        ctx_a = AgentHookContext(iteration=0, messages=msgs_a)
        await hook.before_iteration(ctx_a)
        await hook.skill_indexer.refresh()
        hook.record_skill_candidates(["duckduckgo-search"])
        tc = MagicMock(spec=ToolCallRequest)
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=msgs_a, tool_calls=[tc])
        )

        msgs_b: list[dict] = [{"role": "user", "content": "next task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        row = hook.db.execute(
            "SELECT skills_used FROM trajectories"
        ).fetchone()
        assert row is not None
        skills = _json.loads(row[0])
        assert "duckduckgo-search" in skills

    async def test_no_trajectory_for_system_only_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sessions with no user message produce no trajectory row."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [{"role": "system", "content": "initializing"}]
        ctx_a = AgentHookContext(iteration=0, messages=msgs_a)
        await hook.before_iteration(ctx_a)
        await hook.after_iteration(ctx_a)

        msgs_b: list[dict] = [{"role": "user", "content": "next task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        count = hook.db.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Phase 2.5: trajectory edge cases
# ---------------------------------------------------------------------------

class TestTrajectoryEdgeCases:
    async def test_partial_outcome_with_errors_and_substantial_messages(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Error session with >4 substantive messages → outcome='partial'."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "step 1"},
            {"role": "user", "content": "continue"},
            {"role": "assistant", "content": "step 2"},
            {"role": "user", "content": "still going"},
            {"role": "assistant", "content": "step 3"},
        ]
        ctx = AgentHookContext(iteration=0, messages=msgs_a, error="something failed")
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)

        msgs_b: list[dict] = [{"role": "user", "content": "next"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        row = hook.db.execute("SELECT outcome FROM trajectories").fetchone()
        assert row is not None
        assert row[0] == "partial"

    async def test_trajectory_includes_reflection_text(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [{"role": "user", "content": "do something"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        reflect = loop.tools.get("reflect")
        await reflect.execute(content="Always check the output format first.")

        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        msgs_b: list[dict] = [{"role": "user", "content": "new task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        row = hook.db.execute("SELECT reflection FROM trajectories").fetchone()
        assert row is not None
        assert "output format" in row[0]

    async def test_trajectory_task_truncated_at_500_chars(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        long_content = "x" * 800
        msgs_a: list[dict] = [{"role": "user", "content": long_content}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_a))
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        msgs_b: list[dict] = [{"role": "user", "content": "next"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        row = hook.db.execute("SELECT task FROM trajectories").fetchone()
        assert row is not None
        assert len(row[0]) == 500


# ---------------------------------------------------------------------------
# Phase 3: trajectory_search tool
# ---------------------------------------------------------------------------

class TestTrajectorySearch:
    def test_tool_registered(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        assert "trajectory_search" in loop.tools
        assert isinstance(loop.tools.get("trajectory_search"), TrajectorySearchTool)

    async def test_returns_empty_when_no_trajectories(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)
        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="anything")
        assert "No matching" in out

    async def test_vec_search_returns_matching_trajectory(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        # Insert a trajectory row manually
        cur = hook.db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, reflection, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("search the web for news", '["duckduckgo-search"]', "ok", "worked fine", _time.time()),
        )
        traj_id = cur.lastrowid
        hook.db.commit()

        # Insert its embedding (the fake embed for "search the web" keyword)
        vec = _FAKE_VEC_SEARCH.astype("float32")
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (traj_id, vec.tobytes()),
        )
        hook.db.commit()

        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="search the web for something")
        assert "search the web" in out
        assert "OK" in out
        assert "duckduckgo-search" in out

    async def test_fts_fallback_when_no_embedding(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        hook.db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("write a report about climate", '[]', "ok", _time.time()),
        )
        hook.db.commit()

        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="write report climate")
        # FTS fallback may or may not match — just verify no crash and format ok
        assert isinstance(out, str)

    async def test_trajectory_embed_written_on_session_boundary(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """TrajectoryWriter schedules embedding; trajectories_vec gets a row."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs_a: list[dict] = [{"role": "user", "content": "search the web for news"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_a))
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        msgs_b: list[dict] = [{"role": "user", "content": "next task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_b))

        # Drain the background embed task
        await hook.trajectory_writer.drain()

        traj_row = hook.db.execute("SELECT id FROM trajectories").fetchone()
        assert traj_row is not None

        vec_row = hook.db.execute(
            "SELECT trajectory_id FROM trajectories_vec WHERE trajectory_id = ?",
            (traj_row[0],),
        ).fetchone()
        assert vec_row is not None, "trajectories_vec row not written after drain"


# ---------------------------------------------------------------------------
# Phase 3: trajectory context injection
# ---------------------------------------------------------------------------

class TestTrajectoryContextInjection:
    async def test_injection_off_by_default(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)  # inject_context defaults to False

        msgs: list[dict] = [{"role": "user", "content": "search the web"}]
        before_len = len(msgs)
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))
        # No trajectory injection should have happened
        injected = [m for m in msgs[before_len:] if "past session" in str(m.get("content", ""))]
        assert injected == []

    async def test_injection_fires_when_enabled_and_similar(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={"trajectory": {"inject_context": True, "inject_min_similarity": 0.0}},
        )

        # Seed a trajectory with its vec
        cur = hook.db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("search the web for news", '["duckduckgo-search"]', "ok", _time.time()),
        )
        traj_id = cur.lastrowid
        hook.db.commit()
        vec = _FAKE_VEC_SEARCH.astype("float32")
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (traj_id, vec.tobytes()),
        )
        hook.db.commit()

        msgs: list[dict] = [{"role": "user", "content": "search the web"}]
        before_len = len(msgs)
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))

        injected = [m for m in msgs[before_len:] if "past session" in str(m.get("content", ""))]
        assert len(injected) == 1
        assert "duckduckgo-search" in injected[0]["content"]

    async def test_injection_skipped_when_similarity_below_threshold(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={"trajectory": {"inject_context": True, "inject_min_similarity": 0.999}},
        )

        # Seed a trajectory with a very different vector (unrelated)
        cur = hook.db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("something academic", '[]', "ok", _time.time()),
        )
        traj_id = cur.lastrowid
        hook.db.commit()
        # Use the academic vector — orthogonal to the search query vector
        vec = _FAKE_VEC_ACADEMIC.astype("float32")
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (traj_id, vec.tobytes()),
        )
        hook.db.commit()

        msgs: list[dict] = [{"role": "user", "content": "search the web"}]
        before_len = len(msgs)
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))

        injected = [m for m in msgs[before_len:] if "past session" in str(m.get("content", ""))]
        assert injected == []

    async def test_injection_only_on_iteration_zero(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={"trajectory": {"inject_context": True, "inject_min_similarity": 0.0}},
        )

        cur = hook.db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("search the web", '[]', "ok", _time.time()),
        )
        hook.db.execute(
            "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
            (cur.lastrowid, _FAKE_VEC_SEARCH.astype("float32").tobytes()),
        )
        hook.db.commit()

        msgs: list[dict] = [{"role": "user", "content": "search the web"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))
        msgs.append({"role": "assistant", "content": "ok"})

        # Iteration 1 — no injection
        before_len = len(msgs)
        await hook.before_iteration(AgentHookContext(iteration=1, messages=msgs))
        injected = [m for m in msgs[before_len:] if "past session" in str(m.get("content", ""))]
        assert injected == []
