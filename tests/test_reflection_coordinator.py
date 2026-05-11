"""Tests for ReflectionCoordinator."""
from __future__ import annotations

import time

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.coordinator.reflection import ReflectionCoordinator


@pytest.fixture
def hook(tmp_path):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop)


@pytest.fixture
def coord(hook) -> ReflectionCoordinator:
    return hook._reflection_coord


class TestScoreIteration:
    def test_nudge_set_when_threshold_exceeded(
        self, coord: ReflectionCoordinator
    ) -> None:
        # error_score = 3.0 per call; default threshold = 5.0
        coord.score_iteration(had_error=True, user_text=None)
        assert not coord._nudge_pending  # 3.0 < 5.0
        coord.score_iteration(had_error=True, user_text=None)
        assert coord._nudge_pending  # 6.0 >= 5.0

    def test_score_resets_after_nudge_triggered(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._salience_score = 4.9
        coord.score_iteration(had_error=True, user_text=None)
        assert coord._nudge_pending
        assert coord._salience_score == 0.0

    def test_tool_burst_separate_from_score_iteration(
        self, coord: ReflectionCoordinator
    ) -> None:
        # record_tool_burst is separate from score_iteration
        coord.record_tool_burst(5)  # >= _TOOL_BURST_MIN, adds 2.0
        coord.record_tool_burst(5)  # adds another 2.0 → total 4.0
        assert coord._salience_score == pytest.approx(4.0)
        assert not coord._nudge_pending
        coord.score_iteration(had_error=True, user_text=None)  # adds 3.0 → 7.0 >= 5.0
        assert coord._nudge_pending


class TestTakeNudge:
    def test_returns_message_and_clears_flag(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._nudge_pending = True
        msg = coord.take_nudge()
        assert msg is not None
        assert msg["role"] == "system"
        assert "reflect" in msg["content"].lower()
        assert not coord._nudge_pending

    def test_returns_none_when_no_nudge(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._nudge_pending = False
        assert coord.take_nudge() is None


class TestGetSessionInjections:
    def test_returns_empty_when_no_reflections(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:1", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.commit()
        assert coord.get_session_injections(session_id) == []

    def test_returns_unseen_reflection_content(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:2", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (session_id, "key insight here", time.time()),
        )
        hook.db.commit()

        msgs = coord.get_session_injections(session_id)
        assert len(msgs) == 1
        assert "key insight here" in msgs[0]["content"]

    def test_watermark_prevents_reinjection(
        self, hook, coord: ReflectionCoordinator
    ) -> None:
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            ("test:3", time.time()),
        )
        session_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (session_id, "first reflection", time.time()),
        )
        hook.db.commit()

        coord.get_session_injections(session_id)  # sets watermark
        assert coord.get_session_injections(session_id) == []  # no new content

    def test_returns_empty_for_none_session_id(
        self, coord: ReflectionCoordinator
    ) -> None:
        assert coord.get_session_injections(None) == []


class TestOnNewSession:
    def test_prunes_session_from_watermark_dict(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._last_injected_reflection_id[42] = 100
        coord._last_injected_reflection_id[99] = 200
        coord.on_new_session(42)
        assert 42 not in coord._last_injected_reflection_id
        assert 99 in coord._last_injected_reflection_id  # unaffected

    def test_resets_global_watermark(
        self, coord: ReflectionCoordinator
    ) -> None:
        coord._last_injected_global_reflection_id = 999
        coord.on_new_session(1)
        assert coord._last_injected_global_reflection_id == 0
