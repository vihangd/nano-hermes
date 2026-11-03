"""Injections on nanobot's Responses API path (deepseek-v4-flash, OpenAI, ...).

nanobot folds every system message into ``instructions`` and keeps the last,
so an injected system message replaced the whole system prompt; and its
stored state is replayed every later turn.
"""
from __future__ import annotations

import pytest

state = pytest.importorskip("nanobot.providers.openai_responses.state")

from nano_hermes.cli import _drop_injected_from_state, _keep_injected_system  # noqa: E402

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


def test_injection_is_not_stored_in_provider_state():
    _, items = _keep_injected_system(state.convert_messages)(_MESSAGES)
    built = _drop_injected_from_state(state.build_responses_state)(
        provider="deepseek", model="deepseek-v4-flash", input_items=items, output_items=[],
    )
    stored = built.payload["items"]
    assert [i["role"] for i in stored] == ["user"]


def test_patch_wires_both_wrappers(monkeypatch):
    import importlib

    import nano_hermes.cli as cli_mod

    monkeypatch.setattr(state, "convert_messages", state.convert_messages)
    mods = []
    for name in ("openai_responses.state", "openai_compat_provider", "openai_codex_provider",
                 "azure_openai_provider", "openai_responses.parsing"):
        mod = importlib.import_module(f"nanobot.providers.{name}")
        if hasattr(mod, "build_responses_state"):
            monkeypatch.setattr(mod, "build_responses_state", mod.build_responses_state)
            mods.append(mod)
    cli_mod._patch_responses()
    cli_mod._patch_responses()  # idempotent
    assert state.prepare_responses_input(
        _MESSAGES, state=None, provider="deepseek", model="deepseek-v4-flash",
    )[0] == "NANOBOT PROMPT"
    assert all(m.build_responses_state._nano_hermes for m in mods)
