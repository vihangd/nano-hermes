"""The ``trajectory_search`` agent-facing Tool.

Semantic search over past trajectories (one row per completed session).
Lets the agent ask "have I done something like this before?" and surface
the outcome and any reflections from similar prior tasks.

Falls back to a LIKE-based text search when all embedding providers fail.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..embedding.chain import AllProvidersFailed

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "Describe the task you're working on. Returns similar past sessions ranked by similarity.",
        },
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Number of past trajectories to return. Defaults to 3.",
        },
    },
    "required": ["query"],
}


@tool_parameters(_SCHEMA)
class TrajectorySearchTool(Tool):
    """Search past session trajectories for tasks similar to the current one.

    Returns distilled lessons from previous sessions: what the task was,
    whether it succeeded, which skills were used, and any reflections you
    wrote. Higher signal than raw session_search for recurring task patterns.

    Degrades to keyword search if every embedding provider is unreachable.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "trajectory_search"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        if not query.strip():
            return "Error: query must not be empty."
        k = int(kwargs.get("k") or 3)

        try:
            results = await self._vec_search(query, k)
        except AllProvidersFailed:
            results = self._fts_fallback(query, k)

        if not results:
            return "No matching past trajectories found."

        lines = []
        now = time.time()
        for row in results:
            traj_id, task, skills_used, outcome, reflection, created_at = row
            age_days = int((now - created_at) / 86400)
            age = f"{age_days}d ago" if age_days > 0 else "today"
            skills = json.loads(skills_used) if skills_used else []
            skill_str = ", ".join(skills) if skills else "none"
            lines.append(
                f"[{outcome.upper()}] {age} — {task[:120]}\n"
                f"  skills: {skill_str}"
            )
            if reflection:
                first_line = reflection.splitlines()[0][:200]
                lines.append(f"  reflection: {first_line}")
        return "\n".join(lines)

    async def _vec_search(self, query: str, k: int) -> list[tuple]:
        import numpy as np

        async with self._hook.embedder() as chain:
            [vec] = await chain.embed([query])

        vec_blob = vec.astype(np.float32).tobytes()
        rows = self._hook.db.execute(
            "SELECT trajectory_id, distance FROM trajectories_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, k),
        ).fetchall()
        if not rows:
            return []

        placeholders = ",".join("?" * len(rows))
        id_map = {r[0]: r[1] for r in rows}  # trajectory_id → distance
        traj_rows = self._hook.db.execute(
            f"SELECT id, task, skills_used, outcome, reflection, created_at "
            f"FROM trajectories WHERE id IN ({placeholders})",
            list(id_map.keys()),
        ).fetchall()

        # Sort by original vec distance order
        traj_rows.sort(key=lambda r: id_map.get(r[0], 9999))
        return traj_rows

    def _fts_fallback(self, query: str, k: int) -> list[tuple]:
        """Simple LIKE-based fallback when embedding is unavailable."""
        # Use the first significant word from the query
        words = [w for w in query.split() if len(w) > 3][:3]
        if not words:
            return []
        pattern = "%" + "%".join(words[:2]) + "%"
        return self._hook.db.execute(
            "SELECT id, task, skills_used, outcome, reflection, created_at "
            "FROM trajectories WHERE task LIKE ? ORDER BY created_at DESC LIMIT ?",
            (pattern, k),
        ).fetchall()
