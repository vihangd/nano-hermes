"""The ``nano_status`` agent-facing Tool — internal observability snapshot.

Returns a compact, human-readable summary of nano-hermes state so the agent
(or a developer) can inspect what's happening without reading SQLite directly.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..paths import state_db

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "required": [],
}


@tool_parameters(_SCHEMA)
class NanoStatusTool(Tool):
    """Return a snapshot of nano-hermes internal state.

    Useful for checking whether the hook is active, how many turns have been
    archived, what the current salience score is, and how skills are
    distributed across lifecycle stages. No arguments required.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "nano_status"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        hook = self._hook
        session_id = hook.current_session_id

        # Turn count for the current session
        if session_id is not None:
            row = hook.db.execute(
                "SELECT MAX(turn_index) FROM chunks WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            turns = (row[0] or 0) + 1 if row and row[0] is not None else 0
        else:
            turns = 0

        # Reflection count for the current session
        if session_id is not None:
            ref_count = hook.db.execute(
                "SELECT COUNT(*) FROM reflections WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
        else:
            ref_count = 0

        # Skill counts by status
        rows = hook.db.execute(
            "SELECT status, COUNT(*) FROM skill_stats GROUP BY status"
        ).fetchall()
        skill_counts: dict[str, int] = {"draft": 0, "active": 0, "deprecated": 0}
        for status, count in rows:
            if status in skill_counts:
                skill_counts[status] = count

        # DB size on disk
        db_path = state_db(hook.workspace)
        try:
            db_bytes = os.path.getsize(db_path)
            db_size = f"{db_bytes / 1024:.1f} KB"
        except OSError:
            db_size = "unknown"

        nudge = "yes" if hook._nudge_pending else "no"
        session_label = str(session_id) if session_id is not None else "none"

        # A stale FTS index silently drops the lexical half of hybrid search,
        # so surface it rather than letting recall quietly degrade.
        from ..session.db import stale_fts_tables  # noqa: PLC0415

        stale = stale_fts_tables(hook.db)
        fts_line = (
            f"\nfts: DEGRADED — stale index: {', '.join(stale)} "
            "(rebuilt automatically on next start)"
            if stale
            else ""
        )

        return (
            f"session: {session_label}\n"
            f"turns: {turns}\n"
            f"salience: {hook._salience_score:.1f} (nudge pending: {nudge})\n"
            f"reflections: {ref_count}\n"
            f"skills: {skill_counts['draft']} draft, "
            f"{skill_counts['active']} active, "
            f"{skill_counts['deprecated']} deprecated\n"
            f"db size: {db_size}"
            f"{fts_line}"
        )
