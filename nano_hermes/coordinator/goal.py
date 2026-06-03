"""Goal-state transition tracking (Phase 8).

nanobot's ``/goal`` command (and its ``long_task`` lineage) attaches a
``goal_state`` blob to session metadata; ``goal_state_runtime_lines`` then
emits "Goal (active):" / objective / "Summary:" lines INSIDE the
``[Runtime Context …]`` block that ``ContextBuilder`` appends to the
**current user message** every turn while a goal is active. (It is NOT in
the system prompt — context.py merges the runtime block into the user
content — and it is stripped from persisted history, so it appears only on
the live turn.)

Rather than reach into nanobot's SessionManager (no session_key is exposed
to the hook, and it's fragile across versions), we detect goal state by
scanning ``context.messages`` for that block. We require BOTH the runtime
sentinel and the marker so a user merely typing "Goal (active):" doesn't
trip detection. When the block disappears between turns we treat that as a
goal completion and surface it to the reflection coordinator so the next
iteration sees a Reflexion nudge — distilling the long-running objective
into durable memory while it's still fresh.

Read-only. No nanobot imports beyond what's already in this module.
"""
from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


# Phrase emitted by nanobot.session.goal_state.goal_state_runtime_lines when a
# sustained goal is active, and the opening tag of the Runtime Context block it
# lives inside (context.py ``_RUNTIME_CONTEXT_TAG``). Both are user-facing UI
# text, stable across nanobot versions; we match the tag by prefix so minor
# wording changes after "[Runtime Context" don't break detection.
_ACTIVE_MARKER = "Goal (active):"
# Fallback line nanobot emits when a goal is active but has no objective text
# (goal_state.py). Lets us still detect the active→completed transition even
# though there's no objective to parse.
_ACTIVE_MARKER_NOOBJ = "Goal: active"
_RUNTIME_SENTINEL = "[Runtime Context"
_RUNTIME_END = "[/Runtime Context"  # closes the block; ']' omitted for safety
_OBJECTIVE_MAX_CHARS = 240  # cap for the snapshot stored on transition


def _has_active_marker(text: str) -> bool:
    return _ACTIVE_MARKER in text or _ACTIVE_MARKER_NOOBJ in text


def _goal_block_text(messages: list[dict[str, Any]]) -> str | None:
    """Return the text of the message carrying an active goal inside nanobot's
    Runtime Context block, or ``None``.

    Single pass — both ``goal_active`` and the objective parse derive from this
    so the per-iteration tracker scans the message list once. We don't filter
    by role: the block rides in the current user message today, but requiring
    the runtime sentinel makes detection robust wherever nanobot places it and
    rejects a user who merely types the marker.

    Scanned tail-first and returns on the first match: the runtime block lives
    in the latest user message, so in a long ``/goal`` session this touches one
    or a few messages instead of the whole transcript every iteration.
    """
    for m in reversed(messages):
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
        if _RUNTIME_SENTINEL in text and _has_active_marker(text):
            return text
    return None


def _parse_objective(text: str) -> str | None:
    """Parse the clipped objective from a marker-bearing runtime block.

    The block looks like:
        Goal (active):
        <objective text>
        Summary: <hint>
        [/Runtime Context]
    We take the line(s) after the marker up to a blank line, the 'Summary:'
    prefix, or the Runtime Context closing tag, clipped to
    _OBJECTIVE_MAX_CHARS. (Without a Summary line the objective is followed
    directly by '[/Runtime Context]', which must not bleed into it.)
    """
    if _ACTIVE_MARKER not in text:
        return None  # active-but-no-objective fallback form — nothing to parse
    after = text.split(_ACTIVE_MARKER, 1)[1].lstrip("\n")
    lines: list[str] = []
    for line in after.splitlines():
        stripped = line.strip()
        if not stripped:
            break
        if stripped.startswith("Summary:") or stripped.startswith(_RUNTIME_END):
            break
        lines.append(line)
    objective = " ".join(lines).strip()
    if not objective:
        return None
    if len(objective) > _OBJECTIVE_MAX_CHARS:
        objective = objective[:_OBJECTIVE_MAX_CHARS].rstrip() + "…"
    return objective


def goal_active(messages: list[dict[str, Any]]) -> bool:
    """Return True if a message carries an active goal in its Runtime Context."""
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
