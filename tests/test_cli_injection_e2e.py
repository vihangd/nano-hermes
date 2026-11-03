"""An injection reaches the model exactly once, through nanobot's real runner.

nanobot 0.2.x/0.3.0 build the request before ``before_iteration``: a copy when
context governance rewrote history (the CLI patch must append injections), the
same list when it changed nothing (the hook's append is already there). 0.3.5+
build it after (the patch must stay out). Driving the installed nanobot end to
end checks whichever version is installed.
"""
from __future__ import annotations

import contextlib
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import nanobot.agent.loop as _loop_mod
from nanobot.agent.hook import AgentHook, CompositeHook
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse, ToolCallRequest

import nano_hermes.cli as cli_mod

_MARK = "INJECTED-RECALL-7f3a"


class _Recorder:
    """Provider that records every request's messages and answers once."""

    def __init__(self) -> None:
        self.requests: list[list[dict]] = []
        self.generation = GenerationSettings()

    async def _answer(self, *args, messages=None, **kwargs):
        self.requests.append(deepcopy(messages or (args[0] if args else [])))
        return LLMResponse(content="done", tool_calls=[], finish_reason="stop")

    chat_with_retry = _answer
    chat_stream_with_retry = _answer

    def get_default_model(self) -> str:
        return "test-model"

    def __getattr__(self, name):  # optional provider attributes newer nanobots probe
        return None


class _Injector(AgentHook):
    def __init__(self, hermes) -> None:
        super().__init__()
        self._hermes = hermes

    async def before_iteration(self, context) -> None:
        self._hermes._inject(context.messages, {"role": "user", "content": _MARK})


def _count(messages: list[dict]) -> int:
    return sum(_MARK in str(m.get("content")) for m in messages)


async def test_injection_reaches_model_exactly_once(tmp_path, restore_cli_patches):
    cli_mod._patch_agent_loop()
    rec = _Recorder()
    with contextlib.ExitStack() as stack:  # real SessionManager; no background workers
        sub = stack.enter_context(patch("nanobot.agent.loop.SubagentManager"))
        sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        for sym in ("Dream", "CronService"):
            if hasattr(_loop_mod, sym):
                stack.enter_context(patch(f"nanobot.agent.loop.{sym}"))
        loop = AgentLoop(bus=MessageBus(), provider=rec, workspace=tmp_path, model="test-model")
    hermes = cli_mod._find_hermes_hook(CompositeHook(loop._extra_hooks))  # installed by the patched __init__
    loop._extra_hooks.append(_Injector(hermes))

    await loop.process_direct("hello", session_key="cli:inj")

    assert rec.requests, "provider was never called"
    assert _count(rec.requests[0]) == 1


class _SystemInjector(AgentHook):
    def __init__(self, hermes) -> None:
        super().__init__()
        self._hermes = hermes

    async def before_iteration(self, context) -> None:
        if context.iteration == 0:
            self._hermes._inject(context.messages, {"role": "system", "content": _MARK})


async def test_injections_are_not_saved_into_history(tmp_path, restore_cli_patches):
    """nanobot persists the run's whole message list; an injection left in it
    was replayed on every later turn, growing the prompt until the chat
    compacted on every message. It must still reach every iteration of its
    own run."""
    cli_mod._patch_agent_loop()
    rec = _Recorder()
    first = True

    async def answer(*args, messages=None, **kwargs):
        nonlocal first
        rec.requests.append(deepcopy(messages or (args[0] if args else [])))
        if first:  # one tool round: the run has a second iteration
            first = False
            return LLMResponse(
                content="", finish_reason="tool_calls",
                tool_calls=[ToolCallRequest(id="c1", name="no_such_tool", arguments={})],
            )
        return LLMResponse(content="done", tool_calls=[], finish_reason="stop")

    rec.chat_with_retry = rec.chat_stream_with_retry = answer
    with contextlib.ExitStack() as stack:
        sub = stack.enter_context(patch("nanobot.agent.loop.SubagentManager"))
        sub.return_value.cancel_by_session = AsyncMock(return_value=0)
        for sym in ("Dream", "CronService"):
            if hasattr(_loop_mod, sym):
                stack.enter_context(patch(f"nanobot.agent.loop.{sym}"))
        loop = AgentLoop(bus=MessageBus(), provider=rec, workspace=tmp_path, model="test-model")
    hermes = cli_mod._find_hermes_hook(CompositeHook(loop._extra_hooks))
    loop._extra_hooks.append(_SystemInjector(hermes))

    for i in range(3):
        await loop.process_direct(f"hello {i}", session_key="cli:hist")

    assert [_count(r) for r in rec.requests] == [1, 1, 1, 1]
    saved = loop.sessions.get_or_create("cli:hist").messages
    assert _count(saved) == 0
    assert [m["content"] for m in saved if m["role"] == "assistant" and not m.get("tool_calls")] \
        == ["done"] * 3
