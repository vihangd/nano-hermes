"""Filesystem layout — everything under ``<workspace>/nano_hermes/``.

We piggyback on nanobot's workspace concept to avoid collisions with its
own ``workspace/memory/`` and ``workspace/skills/`` subtrees.
"""
from __future__ import annotations

from pathlib import Path


def plugin_root(workspace: Path) -> Path:
    return Path(workspace) / "nano_hermes"


def state_db(workspace: Path) -> Path:
    return plugin_root(workspace) / "state.db"


