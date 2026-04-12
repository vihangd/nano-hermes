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

log = logging.getLogger(__name__)

_PATCHED = False


def _patch_agent_loop() -> None:
    """Idempotently patch ``AgentLoop.__init__`` to auto-install us."""
    global _PATCHED
    if _PATCHED:
        return

    from nanobot.agent.loop import AgentLoop

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
    _PATCHED = True


def main() -> None:
    _patch_agent_loop()
    from nanobot.cli.commands import app

    app()


if __name__ == "__main__":
    main()
