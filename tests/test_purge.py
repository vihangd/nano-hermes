"""Tests for purge on startup and cascade deletion."""
from __future__ import annotations

import pytest
import numpy as np

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes

from conftest import _unset_embedding_keys


# ---------------------------------------------------------------------------
# Phase 2: purge_older_than runs on session start
# ---------------------------------------------------------------------------

class TestPurgeOnStartup:
    async def test_old_trajectories_purged_at_iteration_zero(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop, config={"trajectory_retention_days": 30})

        # Seed an old trajectory (60 days ago)
        old_ts = _time.time() - 60 * 86400
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("old task", "ok", old_ts),
        )
        hook.db.commit()

        assert hook.db.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0] == 1

        # Trigger iteration 0 — purge should fire
        messages: list[dict] = [{"role": "user", "content": "new session"}]
        ctx = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0] == 0

    async def test_recent_trajectories_are_kept(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop, config={"trajectory_retention_days": 30})

        # Seed a recent trajectory (5 days ago)
        recent_ts = _time.time() - 5 * 86400
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("recent task", "ok", recent_ts),
        )
        hook.db.commit()

        messages: list[dict] = [{"role": "user", "content": "new session"}]
        ctx = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Phase 2.5: purge cascades to sessions and chunks
# ---------------------------------------------------------------------------

class TestPurgeSessionsCascade:
    async def test_old_sessions_purged_with_chunks(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop, config={"trajectory_retention_days": 30})

        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:1", old_ts, old_ts),
        )
        old_session_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, 0, 'user', 'old content', ?)",
            (old_session_id, old_ts),
        )
        hook.db.commit()

        assert hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert hook.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1

        messages: list[dict] = [{"role": "user", "content": "new session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=messages))

        assert hook.db.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (old_session_id,)
        ).fetchone()[0] == 0
        assert hook.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?", (old_session_id,)
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Phase 6: purge chunks_vec orphan cleanup
# ---------------------------------------------------------------------------

class TestPurgeChunksVecCleanup:
    async def test_purge_cleans_chunks_vec_orphans(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time
        from nano_hermes.session.db import purge_older_than

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:chunks:vec", old_ts, old_ts),
        )
        old_session_id = cur.lastrowid
        cur2 = hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, 0, 'user', 'stale chunk content', ?)",
            (old_session_id, old_ts),
        )
        old_chunk_id = cur2.lastrowid

        fake_vec = np.ones(hook.config.embedding.target_dims, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        hook.db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (old_chunk_id, fake_vec.tobytes()),
        )
        hook.db.commit()

        assert hook.db.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (old_chunk_id,)
        ).fetchone()[0] == 1

        purge_older_than(hook.db, days=30)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (old_chunk_id,)
        ).fetchone()[0] == 0, "chunks_vec orphan not cleaned by purge_older_than"
