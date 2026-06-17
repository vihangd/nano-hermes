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
    # nanobot computes messages_for_model before calling before_iteration, so
    # anything the hook appends to context.messages is one iteration late.
    # NanoHermesHook._inject() queues injections in _pending_injections; we
    # drain and append them here so they reach the LLM in the same turn.
    original_request_model = AgentRunner._request_model

    async def patched_request_model(self, spec, messages, hook, context):  # type: ignore[no-untyped-def]
        hermes = _find_hermes_hook(hook)
        if hermes is not None:
            pending = hermes.drain_injections()
            if pending:
                messages = list(messages) + pending
        return await original_request_model(self, spec, messages, hook, context)

    AgentRunner._request_model = patched_request_model  # type: ignore[method-assign]

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
    main()
