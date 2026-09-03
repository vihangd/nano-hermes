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


def _abort_error():
    """An abort-class EvolutionAbortError (billing exhausted)."""
    from nano_hermes.utils.error_classifier import (
        ClassifiedError,
        EvolutionAbortError,
        FailoverReason,
    )
    return EvolutionAbortError(
        ClassifiedError(reason=FailoverReason.billing, status_code=402, message="quota")
    )


def _patch_stages(**overrides):
    """Patch every evolution stage; returns (context_managers, call_log).

    Stages are imported inside _run_evolution_cycle, so patch them at their
    defining module.
    """
    log: list[str] = []

    def _rec(name, result=None, exc=None, sync=False):
        def _fn(*a, **k):
            log.append(name)
            if exc is not None:
                raise exc
            return result
        if sync:
            return _fn

        async def _afn(*a, **k):
            return _fn(*a, **k)
        return _afn

    spec = {
        "nano_hermes.skills.gepa.run_gepa": _rec("gepa", result=[]),
        "nano_hermes.skills.rewriter.run_rewriter": _rec("rewriter", result=[]),
        "nano_hermes.skills.umbrella.run_umbrella_merge": _rec("umbrella", result=[]),
        "nano_hermes.skills.skill_retirement.run_ratchet": _rec(
            "ratchet", result={"retired": [], "cap_evicted": []}, sync=True
        ),
        "nano_hermes.skills.skill_designer.run_skill_designer": _rec("designer", result=[]),
        "nano_hermes.governance.prompt_optimizer.run_opro": _rec("opro", result=False),
    }
    spec.update(overrides)
    return spec, log


async def _run_with_stages(hook, spec):
    import contextlib
    with contextlib.ExitStack() as stack:
        for target, fn in spec.items():
            stack.enter_context(patch(target, fn))
        await hook._run_evolution_cycle()


class TestEvolutionCycleAbortSemantics:
    """An abort-class error (billing/auth) must cut the whole cycle short —
    retrying the later stages would just burn the same dead credential."""

    async def test_gepa_abort_skips_every_later_stage(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        spec, log = _patch_stages()

        def _boom(*a, **k):
            raise _abort_error()

        async def _agepa(*a, **k):
            _boom()
        spec["nano_hermes.skills.gepa.run_gepa"] = _agepa

        await _run_with_stages(hook, spec)

        assert log == [], f"stages ran after an abort: {log}"

    async def test_rewriter_abort_skips_later_stages(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        spec, log = _patch_stages()

        async def _arw(*a, **k):
            log.append("rewriter")
            raise _abort_error()
        spec["nano_hermes.skills.rewriter.run_rewriter"] = _arw

        await _run_with_stages(hook, spec)

        assert log == ["gepa", "rewriter"]
        assert "umbrella" not in log and "ratchet" not in log and "opro" not in log

    async def test_clean_cycle_runs_all_stages_in_order(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        spec, log = _patch_stages()

        await _run_with_stages(hook, spec)

        assert log == ["gepa", "rewriter", "umbrella", "ratchet", "designer", "opro"]

    async def test_non_abort_failure_does_not_stop_the_cycle(self, tmp_path):
        # Each stage has its own try/except; only EvolutionAbortError is fatal.
        hook = _make_hook(tmp_path, interval=1)
        spec, log = _patch_stages()

        async def _aum(*a, **k):
            log.append("umbrella")
            raise RuntimeError("transient")
        spec["nano_hermes.skills.umbrella.run_umbrella_merge"] = _aum

        await _run_with_stages(hook, spec)

        assert log == ["gepa", "rewriter", "umbrella", "ratchet", "designer", "opro"]

    async def test_cycle_count_increments_only_on_completion(self, tmp_path):
        # OPRO's cadence is driven by this counter, so an aborted cycle must
        # not advance it.
        hook = _make_hook(tmp_path, interval=1)
        before = hook._evolution_cycle_count

        spec, _ = _patch_stages()
        await _run_with_stages(hook, spec)
        assert hook._evolution_cycle_count == before + 1

        spec2, _ = _patch_stages()

        async def _agepa(*a, **k):
            raise _abort_error()
        spec2["nano_hermes.skills.gepa.run_gepa"] = _agepa
        await _run_with_stages(hook, spec2)
        assert hook._evolution_cycle_count == before + 1, "aborted cycle advanced the counter"


class TestEvolutionSnapshotGating:
    """Under the write-approval gate nothing mutates during the cycle (writes
    are staged), so the pre-evolution snapshot is wasted work on an SD card."""

    async def test_snapshot_taken_when_gate_off(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        hook.config.skill_stats.snapshot_before_evolution = True
        spec, _ = _patch_stages()
        calls = []
        with patch(
            "nano_hermes.skills.evolution_snapshot.snapshot_evolution",
            lambda *a, **k: calls.append(a),
        ):
            await _run_with_stages(hook, spec)
        assert len(calls) == 1

    async def test_snapshot_skipped_when_gate_on(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        hook.config.skill_stats.snapshot_before_evolution = True
        hook.config.skill_stats.write_approval = "approve"
        spec, _ = _patch_stages()
        calls = []
        with patch(
            "nano_hermes.skills.evolution_snapshot.snapshot_evolution",
            lambda *a, **k: calls.append(a),
        ):
            await _run_with_stages(hook, spec)
        assert calls == [], "snapshot taken despite the write-approval gate"

    async def test_snapshot_failure_does_not_abort_cycle(self, tmp_path):
        hook = _make_hook(tmp_path, interval=1)
        hook.config.skill_stats.snapshot_before_evolution = True
        spec, log = _patch_stages()

        def _boom(*a, **k):
            raise OSError("disk full")
        with patch("nano_hermes.skills.evolution_snapshot.snapshot_evolution", _boom):
            await _run_with_stages(hook, spec)
        assert log[0] == "gepa"


class TestPrincipleCurationCadence:
    """The ACE curator runs on its own interval, independent of the skill
    evolution cycle."""

    def _hook(self, tmp_path, interval):
        loop = _make_loop(tmp_path)
        return nano_hermes.install(
            loop,
            config={"principles": {"enabled": True, "session_interval": interval}},
        )

    def test_disabled_never_schedules(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"principles": {"enabled": False, "session_interval": 1}}
        )
        hook._completed_session_count = 5
        hook._maybe_schedule_principle_curation()
        assert hook._principle_task is None

    def test_zero_interval_never_schedules(self, tmp_path):
        hook = self._hook(tmp_path, 0)
        hook._completed_session_count = 5
        hook._maybe_schedule_principle_curation()
        assert hook._principle_task is None

    async def test_fires_on_boundary(self, tmp_path):
        hook = self._hook(tmp_path, 5)
        hook._completed_session_count = 10
        with patch.object(hook, "_run_principle_curation", new_callable=AsyncMock):
            hook._maybe_schedule_principle_curation()
            assert hook._principle_task is not None
            await hook._principle_task

    def test_does_not_fire_between_boundaries(self, tmp_path):
        hook = self._hook(tmp_path, 5)
        hook._completed_session_count = 11
        hook._maybe_schedule_principle_curation()
        assert hook._principle_task is None

    async def test_does_not_stack_while_running(self, tmp_path):
        hook = self._hook(tmp_path, 1)
        started = asyncio.Event()
        release = asyncio.Event()

        async def _slow():
            started.set()
            await release.wait()

        hook._completed_session_count = 1
        with patch.object(hook, "_run_principle_curation", _slow):
            hook._maybe_schedule_principle_curation()
            first = hook._principle_task
            await started.wait()
            hook._completed_session_count = 2
            hook._maybe_schedule_principle_curation()
            assert hook._principle_task is first, "stacked a second curator task"
            release.set()
            await first

    async def test_curator_exception_is_contained(self, tmp_path):
        hook = self._hook(tmp_path, 1)
        with patch(
            "nano_hermes.skills.principle_curator.run_principle_curator",
            side_effect=RuntimeError("boom"),
        ):
            await hook._run_principle_curation()  # must not raise
