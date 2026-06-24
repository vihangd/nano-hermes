"""Tests for SessionCoordinator — error-recovery paths in sync() and finalize().

Covers lines 50-51 (ensure_session exception), 64-65 (ended_at update exception),
110-111 (trajectory finalize exception).
"""
from __future__ import annotations

import sqlite3
from unittest.mock import MagicMock

import pytest

from nano_hermes.coordinator.session import SessionCoordinator


def _make_coord(db=None, archiver=None, traj_writer=None):
    if db is None:
        db = sqlite3.connect(":memory:")
    if archiver is None:
        archiver = MagicMock()
    if traj_writer is None:
        traj_writer = MagicMock()
    return SessionCoordinator(
        archiver=archiver, db=db, trajectory_writer=traj_writer
    )


def _mem_db_with_tables():
    db = sqlite3.connect(":memory:")
    db.execute(
        "CREATE TABLE sessions (id INTEGER PRIMARY KEY, session_key TEXT, "
        "started_at REAL, ended_at REAL)"
    )
    db.execute(
        "CREATE TABLE reflections (id INTEGER PRIMARY KEY, session_id INTEGER, content TEXT)"
    )
    db.execute(
        "CREATE TABLE chunks (id INTEGER PRIMARY KEY, session_id INTEGER, "
        "turn_index INTEGER, role TEXT, content TEXT, created_at REAL)"
    )
    db.execute("INSERT INTO sessions (id, session_key, started_at) VALUES (1, 'k', 1.0)")
    db.commit()
    return db


class TestSyncExceptionPaths:
    def test_ensure_session_exception_swallowed(self):
        archiver = MagicMock()
        archiver.current_session_id.return_value = None
        archiver.ensure_session.side_effect = RuntimeError("DB locked")
        coord = _make_coord(archiver=archiver)
        sid, completed = coord.sync([{"role": "user", "content": "hello"}])
        assert sid is None
        assert completed is None

    def test_ended_at_update_exception_swallowed(self):
        archiver = MagicMock()
        archiver.current_session_id.return_value = 2
        archiver.ensure_session.return_value = 2
        db = MagicMock()
        db.execute.side_effect = sqlite3.OperationalError("disk full")
        coord = _make_coord(db=db, archiver=archiver)
        coord.current_session_id = 1
        sid, completed = coord.sync([{"role": "user", "content": "new"}])
        assert completed == 1
        assert sid == 2

    def test_no_messages_skips_archiver(self):
        archiver = MagicMock()
        coord = _make_coord(archiver=archiver)
        coord.current_session_id = 42
        sid, completed = coord.sync([])
        assert sid == 42
        assert completed is None
        archiver.current_session_id.assert_not_called()


class TestFinalizeExceptionPath:
    def test_trajectory_write_exception_swallowed(self):
        db = _mem_db_with_tables()
        traj = MagicMock()
        traj.write.side_effect = RuntimeError("trajectory write failed")
        coord = _make_coord(db=db, traj_writer=traj)
        coord.finalize(session_id=1, skills_used={"my_skill"}, had_errors=False)
        traj.write.assert_called_once()

    def test_finalize_success_passes_correct_args(self):
        db = _mem_db_with_tables()
        traj = MagicMock()
        coord = _make_coord(db=db, traj_writer=traj)
        coord.finalize(session_id=1, skills_used={"skill_a"}, had_errors=True)
        traj.write.assert_called_once()
        kw = traj.write.call_args.kwargs
        assert kw["session_id"] == 1
        assert "skill_a" in kw["skills_used"]
        assert kw["had_errors"] is True

import nano_hermes
from conftest import _make_loop
from nano_hermes.coordinator.session import SessionCoordinator


@pytest.fixture
def hook(tmp_path):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop)


@pytest.fixture
def coord(hook) -> SessionCoordinator:
    return hook._session_coord


class TestSync:
    def test_returns_none_none_on_first_call_with_empty_messages(
        self, coord: SessionCoordinator
    ) -> None:
        existing, completed = coord.sync([])
        assert existing is None
        assert completed is None

    def test_bootstraps_session_for_user_message(
        self, coord: SessionCoordinator
    ) -> None:
        msgs = [{"role": "user", "content": "hello"}]
        existing, completed = coord.sync(msgs)
        assert existing is not None
        assert completed is None

    def test_detects_session_boundary(
        self, coord: SessionCoordinator
    ) -> None:
        msgs_a = [{"role": "user", "content": "session A"}]
        sid_a, _ = coord.sync(msgs_a)
        assert sid_a is not None

        msgs_b = [{"role": "user", "content": "session B"}]
        sid_b, completed = coord.sync(msgs_b)
        assert sid_b is not None
        assert sid_b != sid_a
        assert completed == sid_a

    def test_no_boundary_when_same_session(
        self, coord: SessionCoordinator
    ) -> None:
        msgs = [{"role": "user", "content": "first"}]
        coord.sync(msgs)
        msgs.append({"role": "assistant", "content": "response"})
        _, completed = coord.sync(msgs)
        assert completed is None

    def test_sets_ended_at_on_boundary(
        self, hook, coord: SessionCoordinator
    ) -> None:
        msgs_a = [{"role": "user", "content": "session A"}]
        sid_a, _ = coord.sync(msgs_a)

        msgs_b = [{"role": "user", "content": "session B"}]
        coord.sync(msgs_b)

        row = hook.db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (sid_a,)
        ).fetchone()
        assert row is not None and row[0] is not None


class TestFinalize:
    async def test_writes_trajectory_row(
        self, hook, coord: SessionCoordinator
    ) -> None:
        msgs = [{"role": "user", "content": "test task for trajectory"}]
        sid, _ = coord.sync(msgs)
        assert sid is not None

        coord.finalize(sid, skills_used={"skill-a"}, had_errors=False)

        row = hook.db.execute(
            "SELECT task, outcome FROM trajectories WHERE session_id = ?", (sid,)
        ).fetchone()
        assert row is not None
        assert "test task" in row[0]
        assert row[1] == "ok"

    async def test_finalize_sets_ended_at_if_missing(
        self, hook, coord: SessionCoordinator
    ) -> None:
        msgs = [{"role": "user", "content": "another task"}]
        sid, _ = coord.sync(msgs)
        assert sid is not None

        row_before = hook.db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row_before[0] is None

        coord.finalize(sid, skills_used=set(), had_errors=False)

        row_after = hook.db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (sid,)
        ).fetchone()
        assert row_after[0] is not None
