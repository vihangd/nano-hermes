"""Tests for SessionCoordinator."""
from __future__ import annotations

import pytest

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
