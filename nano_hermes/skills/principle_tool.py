"""The ``record_principle`` agent-facing Tool — stores a generalised
if-then rule derived from experience.

Unlike ``reflect`` (which captures a moment-in-time observation),
principles are reusable generalizations: given a condition (context pattern),
take an action to achieve an expected outcome.  They are retrieved at session
start when the current task matches the principle's condition.

Reference: EvolveR (arXiv 2406.00024, 2024).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..redact import format_redaction_note, redact

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "condition": {
            "type": "string",
            "minLength": 8,
            "maxLength": 500,
            "description": (
                "Describe the situation or context where this principle applies. "
                "E.g. 'When deploying a Python app to a server with systemd' or "
                "'When the user asks to parse a deeply nested JSON structure'."
            ),
        },
        "action": {
            "type": "string",
            "minLength": 8,
            "maxLength": 500,
            "description": (
                "What to do in this situation. Be specific and actionable. "
                "E.g. 'Check the systemd service file exists and is enabled before "
                "running systemctl restart'."
            ),
        },
        "expected_outcome": {
            "type": "string",
            "maxLength": 300,
            "description": (
                "Optional: what outcome this action produces. "
                "E.g. 'Avoids service-not-found errors that halt the deploy'."
            ),
        },
    },
    "required": ["condition", "action"],
}


@tool_parameters(_SCHEMA)
class PrincipleTool(Tool):
    """Record a generalised if-then principle from experience.

    Call this when you've noticed a PATTERN — not just a one-off fix. A
    principle captures: given a certain condition, take a specific action to
    achieve a desired outcome. Good principles are situation-independent enough
    to apply again in future sessions on similar tasks.

    Example:
      condition:  "When calling a REST API that returns paginated results"
      action:     "Always check for a next_page token in the response before
                   assuming the result set is complete"
      expected_outcome: "Avoids silently dropping records on paginated endpoints"

    Principles are retrieved at the start of each new session when the current
    task matches the condition text — they appear as a brief hint before you
    begin work.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "record_principle"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        condition: str = (kwargs.get("condition") or "").strip()
        action: str = (kwargs.get("action") or "").strip()
        expected_outcome: str = (kwargs.get("expected_outcome") or "").strip()

        if not condition:
            return "Error: condition must not be empty."
        if not action:
            return "Error: action must not be empty."

        # Redact secrets
        redaction_note = ""
        if self._hook.config.redact_secrets:
            rc = redact(condition)
            ra = redact(action)
            condition, action = rc.text, ra.text
            if expected_outcome:
                ro = redact(expected_outcome)
                expected_outcome = ro.text
            total = rc.count + ra.count
            if total:
                from ..redact import RedactionResult  # noqa: PLC0415
                aggregate = RedactionResult(
                    text="",
                    count=total,
                    kinds=tuple(sorted(set(rc.kinds) | set(ra.kinds))),
                )
                redaction_note = format_redaction_note(aggregate)

        try:
            cur = self._hook.db.execute(
                "INSERT INTO principles "
                "(condition, action, expected_outcome, confidence, created_at) "
                "VALUES (?, ?, ?, 0.5, ?)",
                (condition, action, expected_outcome or None, time.time()),
            )
            principle_id = int(cur.lastrowid)
            # Insert into FTS5 index
            self._hook.db.execute(
                "INSERT INTO principles_fts (rowid, condition, action, content_id) "
                "VALUES (?, ?, ?, ?)",
                (principle_id, condition, action, principle_id),
            )
            self._hook.db.commit()
        except Exception as e:
            return f"Error: {e}"

        return (
            f"ok: principle #{principle_id} recorded"
            + (f" — {expected_outcome[:60]}" if expected_outcome else "")
            + redaction_note
        )
