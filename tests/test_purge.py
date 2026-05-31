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
        if hook._purge_task:
            await hook._purge_task

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
        if hook._purge_task:
            await hook._purge_task

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


# ---------------------------------------------------------------------------
# GAP-2: purge reflections_vec orphan cleanup
# ---------------------------------------------------------------------------

class TestPurgeReflectionsVecCleanup:
    async def test_purge_cleans_reflections_vec_orphans(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time
        from nano_hermes.session.db import purge_older_than

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:reflections:vec", old_ts, old_ts),
        )
        old_sid = cur.lastrowid
        cur2 = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (old_sid, "old reflection", old_ts),
        )
        old_ref_id = cur2.lastrowid

        dims = hook.config.embedding.target_dims
        fake_vec = np.ones(dims, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        hook.db.execute(
            "INSERT INTO reflections_vec (reflection_id, embedding) VALUES (?, ?)",
            (old_ref_id, fake_vec.tobytes()),
        )
        hook.db.commit()

        assert hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec WHERE reflection_id = ?", (old_ref_id,)
        ).fetchone()[0] == 1

        purge_older_than(hook.db, days=30)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec WHERE reflection_id = ?", (old_ref_id,)
        ).fetchone()[0] == 0, "reflections_vec orphan not cleaned by purge"


# ---------------------------------------------------------------------------
# Phase 0.4: purge_older_than uses batched IN-clause DELETEs (not N+1)
# ---------------------------------------------------------------------------

class TestPurgeBatchedDelete:
    def test_chunks_vec_delete_is_batched(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """purge_older_than must issue a single batched DELETE for chunks_vec."""
        import time as _time
        from nano_hermes.session.db import purge_older_than

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:batch:1", old_ts, old_ts),
        )
        old_session_id = cur.lastrowid

        fake_vec = np.ones(hook.config.embedding.target_dims, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)

        chunk_ids = []
        for i in range(3):
            cur2 = hook.db.execute(
                "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
                "VALUES (?, ?, 'user', 'content', ?)",
                (old_session_id, i, old_ts),
            )
            cid = cur2.lastrowid
            chunk_ids.append(cid)
            hook.db.execute(
                "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                (cid, fake_vec.tobytes()),
            )
        hook.db.commit()

        # sqlite3.Connection.execute is a C-level slot and cannot be
        # monkeypatched directly. Wrap the connection in a proxy instead.
        class _CountingProxy:
            def __init__(self, conn):
                self._real = conn
                self.delete_count = 0

            def execute(self, sql, *args, **kwargs):
                if "chunks_vec" in sql.lower() and "delete" in sql.lower():
                    self.delete_count += 1
                return self._real.execute(sql, *args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._real, name)

        proxy = _CountingProxy(hook.db)
        purge_older_than(proxy, days=30)  # type: ignore[arg-type]

        # All 3 chunk_ids should be deleted in exactly ONE batched statement.
        assert proxy.delete_count == 1, (
            f"Expected 1 batched DELETE, got {proxy.delete_count} — N+1 bug present"
        )
        for cid in chunk_ids:
            assert hook.db.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (cid,)
            ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Pi hardening: VACUUM cooldown gate + busy_timeout
# ---------------------------------------------------------------------------

class TestVacuumCooldown:
    def _seed_old_trajectory(self, hook) -> None:
        import time as _time

        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("old task", "ok", _time.time() - 60 * 86400),
        )
        hook.db.commit()

    async def test_first_purge_with_deletions_records_vacuum(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nano_hermes.hook import _META_LAST_VACUUM

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop, config={"trajectory_retention_days": 30})
        self._seed_old_trajectory(hook)

        messages: list[dict] = [{"role": "user", "content": "new session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=messages))
        if hook._purge_task:
            await hook._purge_task

        row = hook.db.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_LAST_VACUUM,)
        ).fetchone()
        assert row is not None, "VACUUM timestamp not recorded after first purge"
        assert float(row[0]) > 0

    async def test_purge_within_cooldown_skips_vacuum(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import time as _time
        from nano_hermes.hook import _META_LAST_VACUUM

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={
                "trajectory_retention_days": 30,
                "vacuum_min_interval_days": 7,
            },
        )
        # Pretend a VACUUM ran one day ago — inside the 7-day cooldown.
        recent = _time.time() - 1 * 86400
        hook.db.execute(
            "INSERT INTO meta(key, value) VALUES (?, ?)",
            (_META_LAST_VACUUM, str(recent)),
        )
        hook.db.commit()
        self._seed_old_trajectory(hook)

        messages: list[dict] = [{"role": "user", "content": "new session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=messages))
        if hook._purge_task:
            await hook._purge_task

        # Cooldown branch must NOT overwrite the timestamp (VACUUM skipped),
        # but the purge itself still removed the aged row.
        row = hook.db.execute(
            "SELECT value FROM meta WHERE key = ?", (_META_LAST_VACUUM,)
        ).fetchone()
        assert float(row[0]) == recent, "VACUUM ran despite cooldown"
        assert hook.db.execute(
            "SELECT COUNT(*) FROM trajectories"
        ).fetchone()[0] == 0, "purge did not delete aged row"


class TestBusyTimeout:
    def test_busy_timeout_applied_to_main_connection(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop, config={"sqlite_busy_timeout_ms": 7500}
        )
        assert hook.db.execute("PRAGMA busy_timeout").fetchone()[0] == 7500

    async def test_busy_timeout_applied_to_purge_connection(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The short-lived purge connection holds the VACUUM exclusive lock,
        so it must also honour busy_timeout."""
        import sqlite3 as _sqlite3
        import time as _time

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop,
            config={
                "trajectory_retention_days": 30,
                "sqlite_busy_timeout_ms": 6250,
            },
        )
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("old task", "ok", _time.time() - 60 * 86400),
        )
        hook.db.commit()

        seen: list[int] = []
        real_connect = _sqlite3.connect

        # Connection.execute is a read-only C slot; override it via a subclass
        # factory so isolation_level assignment (used around VACUUM) still hits
        # the real connection.
        class _SpyConn(_sqlite3.Connection):
            def execute(self, sql, *a, **k):  # type: ignore[override]
                if "busy_timeout" in sql.lower():
                    seen.append(int(sql.rsplit("=", 1)[1]))
                return super().execute(sql, *a, **k)

        def _spy_connect(*args, **kwargs):
            kwargs.setdefault("factory", _SpyConn)
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(_sqlite3, "connect", _spy_connect)

        messages: list[dict] = [{"role": "user", "content": "new session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=messages))
        if hook._purge_task:
            await hook._purge_task

        assert 6250 in seen, "busy_timeout not applied to purge connection"
