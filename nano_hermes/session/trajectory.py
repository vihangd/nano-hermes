"""Write session-level trajectory rows into the ``trajectories`` table.

Called at session boundaries (when a new session starts) so each row
captures the full arc of one conversation: what the task was, which
skills were consulted, whether it succeeded, and any reflections written.

No LLM calls — task is the first user message (500-char cap), outcome is
a heuristic (ok/fail/partial), reflections are the raw reflection texts
stored by the reflect tool.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from typing import Any

from .archiver import _extract_text

log = logging.getLogger(__name__)


class TrajectoryWriter:
    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    def write(
        self,
        *,
        session_id: int,
        messages: list[dict[str, Any]],
        skills_used: list[str],
        reflections: list[str],
        had_errors: bool,
    ) -> int | None:
        """Write one trajectory row. Returns the new row id, or None if
        no task could be extracted (e.g. empty or system-only session).
        """
        task = self._extract_task(messages)
        if not task:
            return None

        outcome = self._outcome(messages, had_errors)
        reflection_text = "\n".join(reflections) if reflections else None

        try:
            cur = self._db.execute(
                "INSERT INTO trajectories "
                "(session_id, task, skills_used, outcome, reflection, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    session_id,
                    task,
                    json.dumps(skills_used),
                    outcome,
                    reflection_text,
                    time.time(),
                ),
            )
            self._db.commit()
            row_id = int(cur.lastrowid)
            log.debug(
                "trajectory written: id=%d session=%d outcome=%s skills=%s",
                row_id,
                session_id,
                outcome,
                skills_used,
            )
            return row_id
        except Exception:
            log.exception("trajectory write failed for session %d", session_id)
            return None

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str | None:
        """Return the first user message text, capped at 500 chars."""
        for msg in messages:
            if msg.get("role") == "user":
                text = _extract_text(msg)
                if text:
                    return text[:500]
        return None

    @staticmethod
    def _outcome(messages: list[dict[str, Any]], had_errors: bool) -> str:
        """Classify outcome as ok / partial / fail.

        - ok: no errors
        - fail: errors and a short session (agent gave up quickly)
        - partial: errors but the agent did substantial work anyway
        """
        if not had_errors:
            return "ok"
        # Count non-system, non-empty messages as a proxy for "work done"
        substantive = sum(
            1
            for m in messages
            if m.get("role") in ("user", "assistant", "tool")
            and _extract_text(m)
        )
        return "partial" if substantive > 4 else "fail"
