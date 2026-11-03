"""Wrapper CLI that auto-installs nano-hermes on every nanobot AgentLoop.

Usage::

    nano-hermes agent       # nanobot agent, with nano-hermes wired in
    nano-hermes gateway     # nanobot gateway, with nano-hermes wired in
    nano-hermes <anything>  # any other nanobot subcommand

Rationale: nanobot has no plugin entry-point group for agent hooks, so
we can't get ``nanobot agent`` to discover us automatically. This
wrapper monkey-patches ``AgentLoop.__init__`` at import time so every
instance the CLI constructs (main loop, gateway, subagent if it goes
through the class) gets nano-hermes installed right after construction.
Then we delegate argv to nanobot's Typer app unchanged.

When nanobot gains a proper hook entry-point group, this wrapper goes
away and the ``nanobot`` command picks us up via ``pip`` metadata.
"""
from __future__ import annotations

import inspect
import logging
from typing import Any

log = logging.getLogger(__name__)

_PATCHED = False


def _find_hermes_hook(hook: Any) -> Any | None:
    """Return the ``NanoHermesHook`` inside a (possibly composite) hook, or None."""
    from nano_hermes.hook import NanoHermesHook

    if isinstance(hook, NanoHermesHook):
        return hook
    for h in getattr(hook, "_hooks", []):
        if isinstance(h, NanoHermesHook):
            return h
    return None


# Responses API (nanobot 0.3.5+: OpenAI, Azure, Codex, and models a provider
# lists in ``responses_models`` — deepseek-v4-flash/pro among them).
_INJECTION_SENTINEL = "\x00nano-hermes-injection-{}\x00"


def _keep_injected_system(convert: Any) -> Any:
    """Wrap nanobot's Chat→Responses ``convert_messages``.

    It folds every system message into ``instructions`` and keeps only the
    last, so a hook injection replaced nanobot's whole system prompt. The
    leading system message stays the instructions; later ones go in place as
    system input items.
    """
    def convert_messages(messages, *args, **kwargs):  # type: ignore[no-untyped-def]
        # ponytail: "not first" = injected; nanobot builds one leading system
        # message. A resumed delta never starts with an injection (the staged
        # user message comes first).
        texts: dict[str, str] = {}
        swapped = []
        for i, m in enumerate(messages):
            if i and m.get("role") == "system" and isinstance(m.get("content"), str):
                token = _INJECTION_SENTINEL.format(i)
                texts[token] = m["content"]
                m = {"role": "user", "content": token}
            swapped.append(m)
        instructions, items = convert(swapped, *args, **kwargs)
        for item in items:
            content = item.get("content")
            if item.get("role") == "user" and isinstance(content, list) and len(content) == 1:
                text = texts.get(content[0].get("text"))
                if text is not None:
                    item["role"] = "system"
                    content[0]["text"] = text
        return instructions, items

    convert_messages._nano_hermes = True  # type: ignore[attr-defined]
    return convert_messages


def _drop_injected_from_state(build: Any) -> Any:
    """Wrap ``build_responses_state`` so injections are not stored.

    The state is replayed on every later turn; nanobot itself never puts a
    system item in ``input`` (its prompt is ``instructions``), so any there are
    this hook's per-run injections.
    """
    def build_responses_state(*args, input_items, **kwargs):  # type: ignore[no-untyped-def]
        input_items = [
            it for it in input_items
            if not (it.get("type") in (None, "message") and it.get("role") == "system")
        ]
        return build(*args, input_items=input_items, **kwargs)

    build_responses_state._nano_hermes = True  # type: ignore[attr-defined]
    return build_responses_state


def _patch_responses() -> None:
    import importlib

    try:
        state_mod = importlib.import_module("nanobot.providers.openai_responses.state")
    except ImportError:
        return  # nanobot before the Responses path
    if getattr(state_mod.convert_messages, "_nano_hermes", False):
        return  # already patched
    state_mod.convert_messages = _keep_injected_system(state_mod.convert_messages)
    # The state module itself is a caller on HEAD (attach_responses_state);
    # 0.3.5 providers import the function by name.
    for name in (
        "nanobot.providers.openai_responses.state",
        "nanobot.providers.openai_compat_provider",
        "nanobot.providers.openai_codex_provider",
        "nanobot.providers.azure_openai_provider",
        "nanobot.providers.openai_responses.parsing",
    ):
        try:
            mod = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(mod, "build_responses_state"):
            mod.build_responses_state = _drop_injected_from_state(mod.build_responses_state)


def _patch_agent_loop() -> None:
    """Idempotently patch ``AgentLoop.__init__`` and ``AgentRunner._request_model``."""
    global _PATCHED
    if _PATCHED:
        return

    from nanobot.agent.loop import AgentLoop
    from nanobot.agent.runner import AgentRunner

    # --- Patch 1: auto-install nano-hermes on every AgentLoop ---
    original_init = AgentLoop.__init__

    def patched_init(self: AgentLoop, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        original_init(self, *args, **kwargs)
        try:
            from nano_hermes import install

            install(self)
            log.debug("nano-hermes installed on AgentLoop id=%s", id(self))
        except Exception:
            # Never break the host CLI if our install crashes — log and move on.
            log.exception("nano-hermes auto-install failed; continuing without it")

    AgentLoop.__init__ = patched_init  # type: ignore[method-assign]

    # --- Patch 2: drain pending injections into messages_for_model ---
    # nanobot 0.2.x/0.3.0 compute messages_for_model before calling
    # before_iteration. When context governance rewrote history that is a
    # copy, so anything the hook appends to context.messages is one iteration
    # late; when governance changed nothing it is the same list and the append
    # is already there. NanoHermesHook._inject() queues injections in
    # _pending_injections; we drain them here and append the ones the request
    # does not already hold (by identity), so each reaches the LLM once.
    # From 0.3.5 (marked by the ``request_state`` parameter) the request is
    # built from context.messages *after* before_iteration, so injections
    # already reach the model and the patch stays out.
    original_request_model = AgentRunner._request_model
    if "request_state" not in inspect.signature(original_request_model).parameters:

        # 0.3.0 adds keyword-only args (malformed_retry); pass them through.
        async def patched_request_model(self, spec, messages, hook, context, *args, **kwargs):  # type: ignore[no-untyped-def]
            hermes = _find_hermes_hook(hook)
            if hermes is not None:
                held = {id(m) for m in messages}
                pending = [m for m in hermes.drain_injections() if id(m) not in held]
                if pending:
                    messages = list(messages) + pending
            return await original_request_model(self, spec, messages, hook, context, *args, **kwargs)

        AgentRunner._request_model = patched_request_model  # type: ignore[method-assign]

    # --- Patch 3: injections on the Responses API path (see helpers) ---
    _patch_responses()

    _PATCHED = True


def main() -> None:
    # nano-hermes-native subcommand: offline review of write-approval-gated
    # writes. Handled before delegating to nanobot's Typer app.
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "pending":
        from nano_hermes.governance.pending import _run

        raise SystemExit(_run(sys.argv[2:]))

    _patch_agent_loop()
    from nanobot.cli.commands import app

    app()


if __name__ == "__main__":
    main()  # pragma: no cover
