"""Goal-state transition tracking (Phase 8).

nanobot's ``/goal`` command (and its ``long_task`` lineage) attaches a
``goal_state`` blob to session metadata; ``goal_state_runtime_lines``
then injects "Goal (active):" lines into the runtime-context system
message whenever a goal is active.

Rather than reach into nanobot's SessionManager (private cache, fragile
across versions), we detect goal state by scanning ``context.messages``
for that visible marker. When the marker disappears between iterations
we treat that as a goal completion and surface it to the reflection
coordinator so the next iteration sees a Reflexion nudge — distilling
the long-running objective into durable memory while it's still fresh.

Read-only. No nanobot imports beyond what's already in this module.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Magic phrase emitted by nanobot.session.goal_state.goal_state_runtime_lines
# when a sustained goal is active. Stable across nanobot versions because
# it's user-facing UI text.
_ACTIVE_MARKER = "Goal (active):"
_OBJECTIVE_MAX_CHARS = 240  # cap for the snapshot stored on transition


def _goal_block_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the text of the first system message carrying the active-goal
    marker, or ``None``. Single pass — both ``goal_active`` and the objective
    parse derive from this so the per-iteration tracker scans the message
    list once, not twice.
    """
    for m in messages:
        if m.get("role") != "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(block.get("text") or "")
                for block in content
                if isinstance(block, dict)
            )
        else:
            continue
        if _ACTIVE_MARKER in text:
            return text
    return None


def _parse_objective(text: str) -> str | None:
    """Parse the clipped objective from a marker-bearing runtime block.

    The block looks like:
        Goal (active):
        <objective text>
        Summary: <hint>
    We take the line(s) after the marker up to a blank line or the
    'Summary:' prefix, clipped to _OBJECTIVE_MAX_CHARS.
    """
    after = text.split(_ACTIVE_MARKER, 1)[1].lstrip("\n")
    lines: list[str] = []
    for line in after.splitlines():
        if not line.strip():
            break
        if line.startswith("Summary:"):
            break
        lines.append(line)
    objective = " ".join(lines).strip()
    if not objective:
        return None
    if len(objective) > _OBJECTIVE_MAX_CHARS:
        objective = objective[:_OBJECTIVE_MAX_CHARS].rstrip() + "…"
    return objective


def goal_active(messages: list[dict[str, Any]]) -> bool:
    """Return True if any system message contains the active-goal marker."""
    return _goal_block_text(messages) is not None


def extract_objective(messages: list[dict[str, Any]]) -> str | None:
    """Return the (clipped) objective text from the runtime context, or None."""
    text = _goal_block_text(messages)
    return _parse_objective(text) if text is not None else None


class GoalTracker:
    """Per-hook tracker that emits "started" / "completed" events.

    Call :meth:`update` once per iteration with ``context.messages``.
    Returns one of:
      - ``None`` if no transition this iteration
      - ``"started"`` if a goal became active
      - ``"completed"`` if the goal disappeared from the runtime context

    The last-seen objective text is preserved on transition so callers
    can include it in a reflection prompt.
    """

    def __init__(self) -> None:
        self.active: bool = False
        self.last_objective: str | None = None

    def update(self, messages: list[dict[str, Any]]) -> str | None:
        # Single scan per iteration: derive both active-ness and the objective
        # from one pass over the message list.
        block = _goal_block_text(messages)
        now_active = block is not None
        if now_active and not self.active:
            self.active = True
            self.last_objective = _parse_objective(block)
            return "started"
        if not now_active and self.active:
            self.active = False
            # last_objective is intentionally NOT cleared so the caller
            # can quote it in the reflection prompt; cleared on next start.
            return "completed"
        if now_active:
            # Keep last_objective current — the objective text can be
            # edited in-place via /goal update commands.
            current = _parse_objective(block)
            if current:
                self.last_objective = current
        return None
