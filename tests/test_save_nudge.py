"""Tests for the memory-save nudge cadence counter."""
from __future__ import annotations

import pytest

import nano_hermes
from conftest import _make_loop


def _make_hook(tmp_path, interval: int = 4):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(
        loop, config={"reflection": {"memory_save_nudge_interval": interval}}
    )


class TestNoteUserTurnCadence:
    def test_fires_at_interval(self, tmp_path):
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        # Two turns: no nudge.
        coord.note_user_turn()
        coord.note_user_turn()
        assert coord.take_save_nudge() is None
        # Third turn: armed.
        coord.note_user_turn()
        nudge = coord.take_save_nudge()
        assert nudge is not None
        assert "memory_patch" in nudge["content"].lower()

    def test_resets_after_taken(self, tmp_path):
        hook = _make_hook(tmp_path, interval=2)
        coord = hook._reflection_coord
        coord.note_user_turn()
        coord.note_user_turn()
        coord.take_save_nudge()  # consume the first nudge
        coord.note_user_turn()
        assert coord.take_save_nudge() is None  # need another full interval
        coord.note_user_turn()
        assert coord.take_save_nudge() is not None

    def test_zero_interval_disabled(self, tmp_path):
        hook = _make_hook(tmp_path, interval=0)
        coord = hook._reflection_coord
        for _ in range(20):
            coord.note_user_turn()
        assert coord.take_save_nudge() is None

    def test_note_memory_save_resets_counter(self, tmp_path):
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        coord.note_user_turn()
        coord.note_user_turn()
        coord.note_memory_save()  # agent saved — drop the counter
        coord.note_user_turn()
        coord.note_user_turn()
        assert coord.take_save_nudge() is None
        coord.note_user_turn()
        assert coord.take_save_nudge() is not None


class TestHydrationFromHistory:
    def test_hydrate_clamps_below_interval(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        coord = hook._reflection_coord
        # Pretend the conversation already had 20 user turns at restart —
        # clamp to (interval - 1) so we don't fire immediately.
        coord.hydrate_save_counter_from_history(recent_user_turns=20)
        assert coord._user_turns_since_save == 4
        # Next turn fires.
        coord.note_user_turn()
        assert coord.take_save_nudge() is not None

    def test_hydrate_zero_turns(self, tmp_path):
        hook = _make_hook(tmp_path, interval=4)
        coord = hook._reflection_coord
        coord.hydrate_save_counter_from_history(recent_user_turns=0)
        assert coord._user_turns_since_save == 0
        # Need full interval to fire.
        for _ in range(3):
            coord.note_user_turn()
        assert coord.take_save_nudge() is None
        coord.note_user_turn()
        assert coord.take_save_nudge() is not None

    def test_hydrate_disabled_when_interval_zero(self, tmp_path):
        hook = _make_hook(tmp_path, interval=0)
        coord = hook._reflection_coord
        coord.hydrate_save_counter_from_history(recent_user_turns=100)
        assert coord._user_turns_since_save == 0  # untouched


class TestHookIntegration:
    async def test_repeated_identical_user_messages_count_as_separate_turns(self, tmp_path):
        """Two `go` messages in a row should advance the counter by 2."""
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        # Iteration 0 — single "go" user turn. Hydration sets counter = 0
        # (one user turn ≤ interval-1 = 2).
        msgs = [{"role": "user", "content": "go"}]
        ctx = AgentHookContext(iteration=0, messages=msgs)
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        assert coord._user_turns_since_save == 1

        # Iteration 1 — same "go" message appears again. List grew by one
        # user-role message; counter advances despite identical text.
        msgs.append({"role": "assistant", "content": "ack"})
        msgs.append({"role": "user", "content": "go"})  # duplicate text
        ctx2 = AgentHookContext(iteration=1, messages=msgs)
        await hook.before_iteration(ctx2)
        await hook.after_iteration(ctx2)
        assert coord._user_turns_since_save == 2

        # Iteration 2 — third "go". Should arm nudge (interval=3).
        msgs.append({"role": "assistant", "content": "ack"})
        msgs.append({"role": "user", "content": "go"})
        ctx3 = AgentHookContext(iteration=2, messages=msgs)
        await hook.before_iteration(ctx3)
        await hook.after_iteration(ctx3)
        # Counter resets to 0 when armed; nudge waits to fire on next before_iteration.
        assert coord._save_nudge_pending or coord._user_turns_since_save == 0


class TestMemoryPatchResetsCounter:
    async def test_add_calls_note_memory_save(self, tmp_path):
        from nano_hermes.memory.tool import MemoryPatchTool  # noqa: PLC0415
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        coord.note_user_turn()
        coord.note_user_turn()
        # The agent saves to memory.
        tool = MemoryPatchTool(hook=hook)
        out = await tool.execute(action="add", slot="memory", content="durable fact")
        assert out.startswith("ok")
        assert coord._user_turns_since_save == 0
        # Counter restarted — next two turns should not yet trigger.
        coord.note_user_turn()
        coord.note_user_turn()
        assert coord.take_save_nudge() is None
