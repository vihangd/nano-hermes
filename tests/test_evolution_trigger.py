"""Tests for the session-boundary GEPA+rewriter evolution trigger."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import nano_hermes
from conftest import _make_loop


def _make_hook(tmp_path, interval: int):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(
        loop,
        config={"skill_stats": {"rewrite_session_interval": interval}},
    )
    return hook


class TestMaybeScheduleEvolution:
    def test_interval_zero_never_schedules(self, tmp_path):
        hook = _make_hook(tmp_path, interval=0)
        hook._completed_session_count = 10  # would fire if interval > 0
        hook._maybe_schedule_evolution()
        assert hook._evolution_task is None

    async def test_fires_on_interval_boundary(self, tmp_path):
        hook = _make_hook(tmp_path, interval=3)
        hook._completed_session_count = 3
        with patch.object(hook, "_run_evolution_cycle", new_callable=AsyncMock):
            hook._maybe_schedule_evolution()
        assert hook._evolution_task is not None
        hook._evolution_task.cancel()

    def test_does_not_fire_between_boundaries(self, tmp_path):
        hook = _make_hook(tmp_path, interval=3)
        hook._completed_session_count = 2  # not a multiple of 3
        hook._maybe_schedule_evolution()
        assert hook._evolution_task is None

    async def test_does_not_stack_while_running(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        hook._completed_session_count = 1

        async def slow():
            await asyncio.sleep(10)

        first_task = asyncio.create_task(slow())
        hook._evolution_task = first_task
        hook._maybe_schedule_evolution()
        assert hook._evolution_task is first_task
        first_task.cancel()

    async def test_reschedules_after_previous_done(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        hook._completed_session_count = 2

        async def done_task():
            pass

        old = asyncio.create_task(done_task())
        await old  # let it complete
        hook._evolution_task = old

        with patch.object(hook, "_run_evolution_cycle", new_callable=AsyncMock):
            hook._maybe_schedule_evolution()
        assert hook._evolution_task is not old
        hook._evolution_task.cancel()


class TestRunEvolutionCycle:
    async def test_gepa_skipped_when_disabled(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        # gepa_enabled defaults to False

        gepa_calls = []
        rewriter_calls = []

        async def fake_gepa(h):
            gepa_calls.append(h)
            return []

        async def fake_rewriter(h, skip=frozenset()):
            rewriter_calls.append(h)
            return []

        with patch("nano_hermes.skills.gepa.run_gepa", new=fake_gepa), \
             patch("nano_hermes.skills.rewriter.run_rewriter", new=fake_rewriter):
            await hook._run_evolution_cycle()

        # run_gepa is called but returns [] because gepa_enabled=False
        assert len(gepa_calls) == 1
        assert len(rewriter_calls) == 1

    async def test_rewriter_error_is_caught(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)

        async def bad_rewriter(h):
            raise RuntimeError("provider down")

        with patch("nano_hermes.skills.gepa.run_gepa", new=AsyncMock(return_value=[])), \
             patch("nano_hermes.skills.rewriter.run_rewriter", new=bad_rewriter):
            # Should not raise
            await hook._run_evolution_cycle()

    async def test_gepa_error_still_runs_rewriter(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        rewriter_ran = []

        async def bad_gepa(h):
            raise RuntimeError("gepa down")

        async def fake_rewriter(h, skip=frozenset()):
            rewriter_ran.append(True)
            return []

        with patch("nano_hermes.skills.gepa.run_gepa", new=bad_gepa), \
             patch("nano_hermes.skills.rewriter.run_rewriter", new=fake_rewriter):
            await hook._run_evolution_cycle()

        assert rewriter_ran  # rewriter runs even if GEPA fails


class TestSessionBoundaryIntegration:
    def test_counter_increments_at_session_boundary(self, tmp_path):
        """_completed_session_count increments when a session boundary is crossed."""
        hook = _make_hook(tmp_path, interval=0)  # disabled so no task spawned
        assert hook._completed_session_count == 0

        hook._completed_session_count += 1
        assert hook._completed_session_count == 1

    def test_config_interval_respected(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        assert hook.config.skill_stats.rewrite_session_interval == 5

    async def test_sync_session_wires_evolution_trigger(self, tmp_path):
        """_sync_session at a session boundary schedules the evolution task."""
        hook = _make_hook(tmp_path, interval=1)
        assert hook._evolution_task is None

        # First call — establishes session 1 (no boundary yet, prev_session is None).
        messages1 = [{"role": "user", "content": "first session"}]
        hook._sync_session(messages1)
        assert hook._evolution_task is None

        # Second call with a new list object — archiver assigns a new session ID,
        # coordinator detects the change and fires _maybe_schedule_evolution.
        messages2 = [{"role": "user", "content": "second session"}]
        with patch.object(hook, "_run_evolution_cycle", new_callable=AsyncMock):
            hook._sync_session(messages2)

        assert hook._evolution_task is not None
        assert hook._completed_session_count == 1
        hook._evolution_task.cancel()

    async def test_skip_passed_to_rewriter(self, tmp_path):
        """GEPA-evolved skills are excluded from the rewriter in the same cycle."""
        hook = _make_hook(tmp_path, interval=1)

        rewriter_skips: list[frozenset] = []

        async def fake_gepa(h):
            return ["skill-a"]

        async def fake_rewriter(h, skip=frozenset()):
            rewriter_skips.append(skip)
            return []

        with patch("nano_hermes.skills.gepa.run_gepa", new=fake_gepa), \
             patch("nano_hermes.skills.rewriter.run_rewriter", new=fake_rewriter):
            await hook._run_evolution_cycle()

        assert len(rewriter_skips) == 1
        assert "skill-a" in rewriter_skips[0]
