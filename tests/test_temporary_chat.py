"""A chat nanobot does not persist must leave no trace in nano-hermes.

nanobot HEAD marks such turns with ``RequestContext.persist_session = False``
(ephemeral turns, SDK temporary chats, non-persistent session policies).
Extra hooks are skipped on ephemeral turns by default, but plugin tools stay
registered and SDK callers can opt hooks back in.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop


@pytest.fixture
def temporary(monkeypatch):
    import nanobot.agent.tools.context as ctx_mod

    ctx = SimpleNamespace(session_key="sdk:temp", persist_session=False)
    monkeypatch.setattr(ctx_mod, "current_request_context", lambda: ctx)


async def test_a_temporary_turn_is_not_archived(tmp_path, temporary):
    hook = nano_hermes.install(_make_loop(tmp_path), config={})
    msgs = [{"role": "user", "content": "secret plans"}]
    ctx = AgentHookContext(iteration=0, messages=msgs, session_key="sdk:temp")
    await hook.before_iteration(ctx)
    msgs.append({"role": "assistant", "content": "noted"})
    await hook.after_iteration(ctx)
    assert hook.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 0
    assert hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


async def test_write_tools_refuse_in_a_temporary_chat(tmp_path, temporary):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config={})
    before = loop.context.memory.read_memory()
    out = await loop.tools.get("memory_patch").execute(
        slot="memory", action="add", content="durable fact"
    )
    assert out.startswith("skipped")
    assert loop.context.memory.read_memory() == before
    out = await loop.tools.get("reflect").execute(content="a lesson")
    assert out.startswith("skipped")
    assert hook.db.execute("SELECT COUNT(*) FROM reflections").fetchone()[0] == 0


async def test_a_persisted_turn_is_unaffected(tmp_path, monkeypatch):
    import nanobot.agent.tools.context as ctx_mod

    ctx = SimpleNamespace(session_key="cli:1", persist_session=True)
    monkeypatch.setattr(ctx_mod, "current_request_context", lambda: ctx)
    hook = nano_hermes.install(_make_loop(tmp_path), config={})
    msgs = [{"role": "user", "content": "hello"}]
    await hook.before_iteration(
        AgentHookContext(iteration=0, messages=msgs, session_key="cli:1")
    )
    assert hook.db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 1


async def test_a_turn_the_hook_never_saw_is_not_persisted(tmp_path):
    """nanobot 0.2.2 has no persist_session field, but an ephemeral turn there
    skips extra hooks while plugin tools stay registered. A tool running in a
    turn whose request context the hook never saw must not write."""
    from nanobot.agent.tools.context import (
        RequestContext,
        bind_request_context,
        reset_request_context,
    )

    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config={})
    before = loop.context.memory.read_memory()
    token = bind_request_context(
        RequestContext(channel="cli", chat_id="1", session_key="cli:1")
    )
    try:
        # Ephemeral: the hook never ran for this turn.
        out = await loop.tools.get("memory_patch").execute(
            slot="memory", action="add", content="durable fact"
        )
        assert out.startswith("skipped")
        assert loop.context.memory.read_memory() == before
        # A normal turn: the hook ran first, so the same call writes.
        await hook.before_iteration(
            AgentHookContext(
                iteration=0,
                messages=[{"role": "user", "content": "hi"}],
                session_key="cli:1",
            )
        )
        out = await loop.tools.get("memory_patch").execute(
            slot="memory", action="add", content="durable fact"
        )
        assert out.startswith("ok"), out
    finally:
        reset_request_context(token)
