"""The ``reflect`` agent-facing Tool — stores a self-critique in the
session-scoped ``reflections`` table.

Reflections live with the current session only. Recent ones get injected
into the system prompt on every ``before_iteration`` after they're
written, so the agent "remembers" what it learned earlier in this
conversation without polluting long-term memory. Cross-session learning
happens via ``memory_patch`` (durable facts) and skills (procedures).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "content": {
            "type": "string",
            "minLength": 8,
            "maxLength": 1000,
            "description": (
                "The reflection itself. 2-4 sentences on what worked, "
                "what didn't, and what you'd do differently. Be concrete "
                "and specific — abstract platitudes help no one."
            ),
        },
    },
    "required": ["content"],
}


@tool_parameters(_SCHEMA)
class ReflectTool(Tool):
    """Store a short self-critique of the current task attempt.

    Call this when:
    - You just recovered from an error and want to note the fix.
    - A user correction taught you something concrete.
    - You burned several tool calls on an approach that didn't pan out.
    - You found an elegant path you'd want to repeat later in this session.

    Reflections are scoped to the CURRENT session only — they'll be
    injected into the system prompt for subsequent iterations in this
    conversation to help you avoid repeating mistakes. For cross-session
    learning, use ``memory_patch`` (durable facts) or authored skills.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "reflect"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        content: str = kwargs["content"]
        session_id = self._hook.current_session_id
        if session_id is None:
            return (
                "Error: no active session — reflections need an archived "
                "session row. If you're seeing this on turn 0, try again "
                "next iteration."
            )
        try:
            self._hook.db.execute(
                "INSERT INTO reflections (session_id, content, created_at) "
                "VALUES (?, ?, ?)",
                (session_id, content.strip(), time.time()),
            )
            self._hook.db.commit()
        except Exception as e:
            return f"Error: {e}"
        return (
            f"ok: reflection saved ({len(content)} chars, session {session_id})"
        )
