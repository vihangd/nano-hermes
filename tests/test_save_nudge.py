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


class TestNudgeReachesTheHookPath:
    """The nudge must survive the real before/after_iteration lifecycle.

    The cadence lived entirely in ReflectionCoordinator tests above, so a
    hook-level regression (hydration re-running every turn) went unnoticed:
    unit-green, integration-dead.
    """

    async def _turns(
        self, hook, n, *, msgs=None, compact_at=None, label="turn", session_key=None
    ):
        """Drive ``n`` user turns; return 1/0 per turn for nudge delivered."""
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415

        msgs = [] if msgs is None else msgs
        delivered = []
        for turn in range(n):
            if compact_at is not None and turn == compact_at:
                msgs[:] = msgs[-2:]
            msgs.append({"role": "user", "content": f"{label} {turn}"})
            ctx = AgentHookContext(
                iteration=0, messages=msgs, session_key=session_key
            )
            await hook.before_iteration(ctx)
            delivered.append(
                int(
                    any(
                        "Memory check-in" in str(m.get("content", ""))
                        for m in msgs
                    )
                )
            )
            msgs[:] = [
                m
                for m in msgs
                if "Memory check-in" not in str(m.get("content", ""))
            ]
            msgs.append({"role": "assistant", "content": "ack"})
            await hook.after_iteration(ctx)
        return delivered

    async def test_nudge_is_delivered_on_cadence(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        delivered = await self._turns(hook, 14)
        assert delivered == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0]

    async def test_nudge_survives_history_compaction(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        # Compaction shifts the cadence by one turn but must not retire it.
        delivered = await self._turns(hook, 14, compact_at=7)
        assert delivered == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]

    async def test_resume_does_not_nudge_on_first_turn(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        history = []
        for i in range(25):
            history.append({"role": "user", "content": f"old {i}"})
            history.append({"role": "assistant", "content": "ok"})
        delivered = await self._turns(hook, 1, msgs=history)
        assert delivered == [0]

    async def test_counter_is_per_conversation(self, tmp_path):
        """One hook serves every chat the loop runs.

        A process-wide counter leaks across conversations: the long chat's
        count makes the short chat's delta permanently negative (nudge dead),
        while the short chat's count replays the long chat's history as fresh
        turns (nudge on every single turn).
        """
        hook = _make_hook(tmp_path, interval=5)
        chat_a, chat_b = [], []
        for i in range(30):
            chat_a.append({"role": "user", "content": f"old {i}"})
            chat_a.append({"role": "assistant", "content": "ok"})
        a_hits, b_hits = [], []
        for _ in range(8):
            a_hits += await self._turns(hook, 1, msgs=chat_a, label="a")
            b_hits += await self._turns(hook, 1, msgs=chat_b, label="b")
        # The cadence is shared, so 16 combined turns at interval=5 must
        # deliver several nudges — not zero, which is what a scalar counter
        # degrades to once the long chat pins the short one's delta.
        assert sum(a_hits) + sum(b_hits) >= 2, (a_hits, b_hits)
        # And no conversation is spammed: a scalar counter nudges the short
        # chat on every turn, because the long chat's count replays as new.
        assert sum(b_hits) <= 2, b_hits

    async def test_busiest_conversation_is_not_evicted_by_idle_ones(self, tmp_path):
        """The counter dict is insertion-ordered and capped at 32 keys.

        Dropping the oldest-*added* key evicts the conversation that has been
        talking the longest. Its next turn finds no entry, re-hydrates, and
        resets the nudge cadence — so the symptom is a repeat hydration, not a
        missing key (hydration puts the key straight back). Only the least
        recently *counted* key may go, the same rule
        ``SessionCoordinator.sync`` applies to its own boundary map.
        """
        hook = _make_hook(tmp_path, interval=5)
        hydrations = []
        coord = hook._reflection_coord
        real = coord.hydrate_save_counter_from_history

        def spy(**kwargs):
            hydrations.append(kwargs)
            return real(**kwargs)

        coord.hydrate_save_counter_from_history = spy
        busy = []
        await self._turns(hook, 1, msgs=busy, label="busy", session_key="chan:busy")
        for i in range(40):  # far past the 32-key cap
            await self._turns(
                hook, 1, msgs=[], label=f"idle{i}", session_key=f"chan:idle{i}"
            )
            await self._turns(
                hook, 1, msgs=busy, label="busy", session_key="chan:busy"
            )
        # One hydration per conversation and no more: 40 idle plus the busy one.
        assert len(hydrations) == 41, len(hydrations)

    async def test_recycled_list_address_does_not_replay_history(self, tmp_path):
        """id(messages) can collide with a dead conversation's entry.

        Exercises the ``session_key=None`` fallback deliberately — ``_turns``
        passes no key, so the counter is keyed on ``id(messages)`` and the
        poked entry below is the one the hook will read.

        A stale entry skips hydration, so the delta against a resumed
        transcript would replay its whole history as fresh turns and arm
        the nudge immediately — the very thing the clamp prevents.
        """
        hook = _make_hook(tmp_path, interval=5)
        history = []
        for i in range(25):
            history.append({"role": "user", "content": f"old {i}"})
            history.append({"role": "assistant", "content": "ok"})
        hook._user_msg_counts[id(history)] = 1  # simulate a recycled address
        delivered = await self._turns(hook, 2, msgs=history)
        assert delivered == [0, 0]
