"""The ``trajectory_search`` agent-facing Tool.

Semantic search over past trajectories (one row per completed session).
Lets the agent ask "have I done something like this before?" and surface
the outcome and any reflections from similar prior tasks.

Falls back to a LIKE-based text search when all embedding providers fail.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..embedding.chain import AllProvidersFailed
from .search import _contains_cjk, _like_escape, reciprocal_rank_fusion

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
            results = await self._hybrid_search(query, k)
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

    async def _hybrid_search(self, query: str, k: int) -> list[tuple]:
        """Fuse dense (trajectories_vec) and lexical (trajectories_fts)
        rankings via RRF.

        Dense retrieval misses exact identifiers — tool names, error codes,
        file paths — that recur verbatim in task descriptions. The BM25
        channel catches those; RRF merges the two rank lists without needing
        to calibrate their incomparable score scales.
        """
        import numpy as np

        async with self._hook.embedder() as chain:
            [vec] = await chain.embed([query])

        # Widen each channel's pool so fusion has signal beyond the top-k.
        pool = max(k * 4, 20)

        vec_blob = vec.astype(np.float32).tobytes()
        vec_rows = self._hook.db.execute(
            "SELECT trajectory_id FROM trajectories_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, pool),
        ).fetchall()
        vec_ids = [r[0] for r in vec_rows]

        fts_ids = self._fts_ids(query, pool)
        if not (vec_ids or fts_ids):
            return []

        rrf_k = self._hook.config.retrieval.rrf_k
        fused = reciprocal_rank_fusion(fts_ids, vec_ids, rrf_k)
        ordered = [
            cid for cid, _ in sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ][:k]
        if not ordered:
            return []

        placeholders = ",".join("?" * len(ordered))
        traj_rows = self._hook.db.execute(
            f"SELECT id, task, skills_used, outcome, reflection, created_at "
            f"FROM trajectories WHERE id IN ({placeholders})",
            ordered,
        ).fetchall()

        rank = {tid: i for i, tid in enumerate(ordered)}
        traj_rows.sort(key=lambda r: rank.get(r[0], 9999))
        return traj_rows

    def _fts_ids(self, query: str, limit: int) -> list[int]:
        """BM25-ranked trajectory ids for *query*, or [] on an unparsable
        FTS expression (a malformed query just drops the lexical channel —
        dense still runs)."""
        try:
            rows = self._hook.db.execute(
                "SELECT rowid FROM trajectories_fts "
                "WHERE trajectories_fts MATCH ? ORDER BY rank LIMIT ?",
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        ids = [r[0] for r in rows]
        # FTS5's tokenizer doesn't segment CJK; fall back to a LIKE scan.
        if not ids and _contains_cjk(query):
            like_rows = self._hook.db.execute(
                "SELECT id FROM trajectories WHERE task LIKE ? ESCAPE '\\' LIMIT ?",
                (f"%{_like_escape(query)}%", limit),
            ).fetchall()
            ids = [r[0] for r in like_rows]
        return ids

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
