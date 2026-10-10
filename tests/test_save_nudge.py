"""Tests for the memory-save nudge cadence counter."""
from __future__ import annotations

import pytest

import nano_hermes
from conftest import _make_loop


def _count(coord, key=None):
    """[turns since save, nudge armed] for *key* (keyless when None)."""
    st = coord._conversations.get(key or "")
    return [st.turns_since_save, st.save_nudge_pending] if st else [0, False]


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
        assert _count(coord)[0] == 4
        # Next turn fires.
        coord.note_user_turn()
        assert coord.take_save_nudge() is not None

    def test_hydrate_zero_turns(self, tmp_path):
        hook = _make_hook(tmp_path, interval=4)
        coord = hook._reflection_coord
        coord.hydrate_save_counter_from_history(recent_user_turns=0)
        assert _count(coord)[0] == 0
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
        assert _count(coord) == [0, False]  # untouched


class TestHookIntegration:
    async def test_repeated_identical_user_messages_count_as_separate_turns(self, tmp_path):
        """Three `go` turns in a row should advance the counter three times.

        Each turn is its own ``iteration=0`` — the counter is per *turn*, and a
        turn's later iterations must not advance it (a tool-using turn fires
        ``after_iteration`` once per iteration). Identical text must not collapse
        the turns either, which is why the cadence never compared text.
        """
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        msgs = [{"role": "user", "content": "go"}]

        async def turn(iteration=0):
            ctx = AgentHookContext(iteration=iteration, messages=msgs)
            await hook.before_iteration(ctx)
            await hook.after_iteration(ctx)

        # Turn 1. Hydration counts the window's single user turn instead of
        # bumping, so it isn't counted twice.
        await turn()
        assert _count(coord)[0] == 1
        # Still turn 1, second iteration: the model called a tool. No advance.
        msgs.append({"role": "assistant", "content": "calling"})
        msgs.append({"role": "tool", "content": "result", "tool_call_id": "c1"})
        await turn(iteration=1)
        assert _count(coord)[0] == 1

        # Turn 2 — same "go" text again.
        msgs.append({"role": "assistant", "content": "ack"})
        msgs.append({"role": "user", "content": "go"})
        await turn()
        assert _count(coord)[0] == 2

        # Turn 3 — third "go". Should arm the nudge (interval=3).
        msgs.append({"role": "assistant", "content": "ack"})
        msgs.append({"role": "user", "content": "go"})
        await turn()
        # Armed and delivered within the same before_iteration: counter back to 0.
        assert _count(coord) == [0, False]


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
        assert _count(coord)[0] == 0
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
        self,
        hook,
        n,
        *,
        msgs=None,
        compact_at=None,
        label="turn",
        session_key=None,
        cap=None,
    ):
        """Drive ``n`` user turns; return 1/0 per turn for nudge delivered."""
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415

        msgs = [] if msgs is None else msgs
        delivered = []
        for turn in range(n):
            if compact_at is not None and turn == compact_at:
                msgs[:] = msgs[-2:]
            if cap is not None:
                # What get_history does every turn: hand back a bounded window.
                msgs[:] = msgs[-cap:]
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
        # Counted at iteration 0 and delivered in the same before_iteration, so
        # the nudge lands on the fifth turn itself.
        assert delivered == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    async def test_nudge_survives_history_compaction(self, tmp_path):
        hook = _make_hook(tmp_path, interval=5)
        # Counting turns instead of window population means compaction no longer
        # even costs a turn of cadence: the interval holds straight through it.
        delivered = await self._turns(hook, 14, compact_at=7)
        assert delivered == [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]

    async def test_nudge_keeps_firing_after_the_window_saturates(self, tmp_path):
        """The cadence must not ride on window *population*.

        ``context.messages`` is a bounded sliding window, so once it saturates
        the user-message count stops growing and a population delta is 0 for
        every later turn. The nudge froze there: measured at five deliveries in
        24 turns and never again, in the subsystem the cadence exists for.
        """
        hook = _make_hook(tmp_path, interval=5)
        delivered = await self._turns(hook, 24, cap=6)
        assert delivered == ([0] * 4 + [1]) * 4 + [0] * 4
        assert sum(delivered) == 4

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

        A process-wide counter leaks across conversations: one chat's first
        turn re-hydrates the shared count, and an armed nudge goes to whichever
        chat iterates next.
        """
        hook = _make_hook(tmp_path, interval=5)
        chat_a, chat_b = [], []
        for i in range(30):
            chat_a.append({"role": "user", "content": f"old {i}"})
            chat_a.append({"role": "assistant", "content": "ok"})
        a_hits, b_hits = [], []
        for _ in range(8):
            a_hits += await self._turns(
                hook, 1, msgs=chat_a, label="a", session_key="chan:a"
            )
            b_hits += await self._turns(
                hook, 1, msgs=chat_b, label="b", session_key="chan:b"
            )
        # Each chat runs its own cadence: the resumed chat hydrates to
        # interval-1 and fires on its second turn, then every fifth; the fresh
        # chat fires on its fifth.
        assert a_hits == [0, 1, 0, 0, 0, 0, 1, 0], a_hits
        assert b_hits == [0, 0, 0, 0, 1, 0, 0, 0], b_hits

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

    async def test_keyless_conversation_still_counts(self, tmp_path):
        """Without a session key the window is a fresh list every turn.

        Keyed by ``id(messages)``, every turn was a first sighting: hydrated
        instead of counted, so the nudge never fired for a keyless caller.
        Keyless turns now share one cadence.
        """
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415

        hook = _make_hook(tmp_path, interval=3)
        history: list[dict] = []
        delivered = []
        for turn in range(9):
            history.append({"role": "user", "content": f"t {turn}"})
            window = [dict(m) for m in history]  # a new list, like get_history
            ctx = AgentHookContext(iteration=0, messages=window, session_key=None)
            await hook.before_iteration(ctx)
            delivered.append(
                int(any("Memory check-in" in str(m.get("content")) for m in window))
            )
            await hook.after_iteration(ctx)
            history.append({"role": "assistant", "content": "ack"})
        assert delivered == [0, 0, 1] * 3

    async def test_one_conversation_cannot_take_or_erase_anothers_nudge(
        self, tmp_path
    ):
        """A shared counter let chat B's first turn reset chat A's count and
        clear A's armed nudge — or, once B was known, deliver A's check-in into
        B's conversation."""
        hook = _make_hook(tmp_path, interval=3)
        coord = hook._reflection_coord
        for _ in range(3):
            coord.note_user_turn("chan:a")
        assert _count(coord, "chan:a") == [0, True]
        coord.observe_user_turn("chan:b", window_user_turns=1)  # B's first turn
        assert coord.take_save_nudge("chan:b") is None
        assert coord.take_save_nudge("chan:a") is not None

    async def test_memory_save_resets_only_the_calling_conversation(self, tmp_path):
        """memory_patch learns its conversation from nanobot's per-turn
        RequestContext contextvar, not from anything the hook shares."""
        from nanobot.agent.tools.context import (  # noqa: PLC0415
            RequestContext,
            bind_request_context,
            reset_request_context,
        )

        from nano_hermes.coordinator.session import mark_request_hooked

        from nano_hermes.memory.tool import MemoryPatchTool  # noqa: PLC0415

        hook = _make_hook(tmp_path, interval=5)
        coord = hook._reflection_coord
        for key in ("chan:a", "chan:b"):
            coord.note_user_turn(key)
            coord.note_user_turn(key)
        token = bind_request_context(
            RequestContext(channel="cli", chat_id="a", session_key="chan:a")
        )
        mark_request_hooked()  # the hook runs first in every real turn
        try:
            out = await MemoryPatchTool(hook=hook).execute(
                action="add", slot="memory", content="durable fact"
            )
        finally:
            reset_request_context(token)
        assert out.startswith("ok")
        assert _count(coord, "chan:a")[0] == 0
        assert _count(coord, "chan:b")[0] == 2


class TestWhatCountsAsAUserTurn:
    async def test_follow_ups_drained_mid_turn_count(self, tmp_path):
        """nanobot drains user follow-ups into the running turn's list; counting
        only at iteration 0 missed every one of them."""
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415

        hook = _make_hook(tmp_path, interval=10)
        coord = hook._reflection_coord
        msgs = [{"role": "user", "content": "start"}]
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=msgs, session_key="chan:a")
        )
        assert _count(coord, "chan:a")[0] == 1
        msgs.append({"role": "assistant", "content": "", "tool_calls": []})
        msgs.append({"role": "user", "content": "also this"})
        msgs.append({"role": "user", "content": "and this"})
        await hook.before_iteration(
            AgentHookContext(iteration=1, messages=msgs, session_key="chan:a")
        )
        assert _count(coord, "chan:a")[0] == 3
        # Another iteration with nothing new adds nothing.
        await hook.before_iteration(
            AgentHookContext(iteration=2, messages=msgs, session_key="chan:a")
        )
        assert _count(coord, "chan:a")[0] == 3

    async def test_a_subagent_result_turn_is_not_a_user_turn(self, tmp_path):
        """A subagent result arrives as an ``assistant`` current message. The
        window still holds older user messages, which used to be enough to
        count the turn."""
        from nanobot.agent.hook import AgentHookContext  # noqa: PLC0415

        hook = _make_hook(tmp_path, interval=10)
        coord = hook._reflection_coord
        msgs = [{"role": "user", "content": "kick off a subagent"}]
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=msgs, session_key="chan:a")
        )
        msgs.append({"role": "assistant", "content": "started"})
        msgs.append({"role": "assistant", "content": "[subagent result] done"})
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=list(msgs), session_key="chan:a")
        )
        assert _count(coord, "chan:a")[0] == 1
