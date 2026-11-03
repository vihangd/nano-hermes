"""Injections on nanobot's Responses API path (deepseek-v4-flash, OpenAI, ...).

nanobot folds every system message into ``instructions`` and keeps the last,
so an injected system message replaced the whole system prompt; and its
stored state is replayed every later turn.
"""
from __future__ import annotations

import pytest

state = pytest.importorskip("nanobot.providers.openai_responses.state")

from nano_hermes.cli import _drop_injected_at_finish, _keep_injected_system  # noqa: E402

_MESSAGES = [
    {"role": "system", "content": "NANOBOT PROMPT"},
    {"role": "user", "content": "hi"},
    {"role": "system", "content": "## Relevant reflections"},
]


def test_injection_does_not_replace_the_system_prompt():
    instructions, items = _keep_injected_system(state.convert_messages)(_MESSAGES)
    assert instructions == "NANOBOT PROMPT"
    assert [(i["role"], i["content"][0]["text"]) for i in items] == [
        ("user", "hi"), ("system", "## Relevant reflections"),
    ]


def _controller(items):
    cs = pytest.importorskip("nanobot.providers.conversation_state")
    from nanobot.providers.base import ProviderConversationState

    ctl = cs.ProviderConversationStateController.__new__(cs.ProviderConversationStateController)
    ctl._boundary = 0
    ctl._state = ProviderConversationState(
        kind="openai_responses", provider="deepseek", model="deepseek-v4-flash",
        version=1, payload={"items": items},
    )
    return cs, ctl


def test_injection_stays_within_the_run_but_is_not_stored():
    _, items = _keep_injected_system(state.convert_messages)(_MESSAGES)
    cs, ctl = _controller(items)
    # Within the run the replayed state still carries the injection ...
    assert ctl.checkpoint([]).payload["items"] == items
    # ... the stored final state does not.
    final = _drop_injected_at_finish(cs.ProviderConversationStateController.finish)(ctl, [])
    assert [i["role"] for i in final.payload["items"]] == ["user"]
    assert ctl._state is final


def test_patch_wires_both_wrappers(monkeypatch):
    import sys

    import nano_hermes.cli as cli_mod

    conv = sys.modules["nanobot.providers.openai_responses.converters"]
    cs = pytest.importorskip("nanobot.providers.conversation_state")
    original = conv.convert_messages
    for mod in list(sys.modules.values()):
        if getattr(mod, "__dict__", {}).get("convert_messages") is original:
            monkeypatch.setattr(mod, "convert_messages", original)
    ctl = cs.ProviderConversationStateController
    monkeypatch.setattr(ctl, "finish", ctl.finish)
    cli_mod._patch_responses()
    once = ctl.finish
    cli_mod._patch_responses()
    assert ctl.finish is once  # idempotent
    assert state.prepare_responses_input(
        _MESSAGES, state=None, provider="deepseek", model="deepseek-v4-flash",
    )[0] == "NANOBOT PROMPT"
    assert conv.convert_messages._nano_hermes and ctl.finish._nano_hermes
