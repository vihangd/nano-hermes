"""Tests for the injection-timing fix in NanoHermesHook._inject / drain_injections.

nanobot computes messages_for_model before calling before_iteration, so
hook-injected messages would normally arrive one iteration late. The fix:
- _inject() appends to both the canonical messages list and _pending_injections
- drain_injections() pops the queue so patched _request_model can append them
"""
from __future__ import annotations

from conftest import _make_loop

import nano_hermes


class TestInjectHelper:
    def test_inject_appends_to_messages_and_queue(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        messages: list[dict] = []
        msg = {"role": "system", "content": "hello"}
        hook._inject(messages, msg)

        assert messages == [msg]
        assert hook._pending_injections == [msg]

    def test_inject_queues_multiple_independently(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        messages: list[dict] = []
        m1 = {"role": "system", "content": "first"}
        m2 = {"role": "system", "content": "second"}
        hook._inject(messages, m1)
        hook._inject(messages, m2)

        assert messages == [m1, m2]
        assert hook._pending_injections == [m1, m2]

    def test_drain_returns_and_clears(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        msg = {"role": "system", "content": "injected"}
        hook._pending_injections = [msg]

        drained = hook.drain_injections()

        assert drained == [msg]
        assert hook._pending_injections == []

    def test_drain_empty_returns_empty_list(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        assert hook.drain_injections() == []

    async def test_before_iteration_clears_pending_injections(self, loop):
        """Stale injections from a failed iteration must not leak into the next."""
        hook = nano_hermes.install(loop)
        # Simulate stale entries left over from a previous iteration
        hook._pending_injections = [{"role": "system", "content": "stale"}]

        from nanobot.agent.hook import AgentHookContext

        ctx = AgentHookContext(iteration=0, messages=[])
        await hook.before_iteration(ctx)

        # _pending_injections is cleared at the top of before_iteration
        # (any new injections from this call are fine, but stale ones are gone)
        # The stale entry must not be present (it was cleared before new ones added)
        contents = [m["content"] for m in hook._pending_injections]
        assert "stale" not in contents
