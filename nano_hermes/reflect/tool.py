"""The ``reflect`` agent-facing Tool — stores a self-critique in the
session-scoped ``reflections`` table.

In ``reflection_scope="session"`` mode (default), reflections live with
the current session only. In ``reflection_scope="global"`` mode, they are
also embedded asynchronously and searchable across all sessions — the most
relevant past reflections are injected on the first iteration of each new
session via cosine similarity to the current task.

Cross-session learning also happens via ``memory_patch`` (durable facts)
and skills (procedures).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from nanobot.agent.tools.base import Tool, tool_parameters

from ..embedding.chain import AllProvidersFailed
from ..redact import format_redaction_note, redact
from ..session.db import run_vec_write

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


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
        stripped = content.strip()
        if not stripped:
            return "Error: reflection content is empty after stripping whitespace."
        content = stripped

        # Redact secrets BEFORE the DB insert and BEFORE embedding — the
        # masked text is what we persist and what cross-session retrieval
        # later sees.
        redaction_note = ""
        if self._hook.config.redact_secrets:
            r = redact(content)
            content = r.text
            redaction_note = format_redaction_note(r)

        session_id = self._hook.current_session_id
        if session_id is None:
            return (
                "Error: no active session — reflections need an archived "
                "session row. If you're seeing this on turn 0, try again "
                "next iteration."
            )
        try:
            cur = self._hook.db.execute(
                "INSERT INTO reflections (session_id, content, created_at) "
                "VALUES (?, ?, ?)",
                (session_id, content.strip(), time.time()),
            )
            self._hook.db.commit()
            reflection_id = int(cur.lastrowid)
        except Exception as e:
            return f"Error: {e}"

        # In global scope mode, embed this reflection so it can be
        # retrieved across sessions by cosine similarity.
        if self._hook.config.reflection_scope == "global":
            self._schedule_embed(reflection_id, content.strip())

        return (
            f"ok: reflection saved ({len(content)} chars, session {session_id})"
            + redaction_note
        )

    def _schedule_embed(self, reflection_id: int, text: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._embed_and_write(reflection_id, text))
            # Keep a reference to avoid GC cancellation
            self._hook._reflection_embed_tasks.add(task)
            task.add_done_callback(self._hook._reflection_embed_tasks.discard)
        except RuntimeError:
            log.debug("no running loop — reflection embed skipped")

    async def _embed_and_write(self, reflection_id: int, text: str) -> None:
        try:
            async with self._hook.embedder() as chain:
                [vec] = await chain.embed([text])
            blob = vec.astype(np.float32).tobytes()

            def _write(w):
                # vec0 does not support UPSERT; delete first, then insert.
                w.execute(
                    "DELETE FROM reflections_vec WHERE reflection_id = ?",
                    (reflection_id,),
                )
                w.execute(
                    "INSERT INTO reflections_vec (reflection_id, embedding) VALUES (?, ?)",
                    (reflection_id, blob),
                )

            await run_vec_write(self._hook.db, _write)
        except AllProvidersFailed as e:
            log.warning("reflection embed skipped: %s", e)
        except Exception:
            log.exception("reflection embed failed for id=%d", reflection_id)
