"""ReflectionCoordinator — extracted from NanoHermesHook.

Manages salience scoring, Reflexion nudge delivery, session-scoped
reflection injection, and global (cross-session) reflection retrieval.

The coordinator produces injection content (list[dict] or dict|None).
The hook is responsible for calling ``self._inject(messages, msg)`` on
whatever the coordinator returns.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Callable

from ..reflect.salience import (
    correction_score,
    error_score,
    tool_burst_score,
)

if TYPE_CHECKING:
    from ..config import NanoHermesConfig
    from ..embedding.chain import EmbeddingChain

log = logging.getLogger(__name__)

_NUDGE_TEXT = (
    "You've just had a tool-heavy or error-prone iteration. When you "
    "have a moment, call reflect(content='...') with a concise takeaway "
    "(2-4 sentences): what worked, what didn't, and what you'd do "
    "differently. This helps you avoid repeating the pattern in the "
    "same session."
)


class ReflectionCoordinator:
    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        config: "NanoHermesConfig",
        embedder_factory: "Callable[[], EmbeddingChain]",
    ) -> None:
        self._db = db
        self._config = config
        self._embedder_factory = embedder_factory
        # Salience state
        self._salience_score: float = 0.0
        self._nudge_pending: bool = False
        # Watermark: highest reflection id injected per session.
        # Bounded by on_new_session pruning completed sessions.
        self._last_injected_reflection_id: dict[int, int] = {}
        # Watermark for global (cross-session) reflections.
        self._last_injected_global_reflection_id: int = 0

    # ------------------------------------------------------------------
    # Salience
    # ------------------------------------------------------------------

    def record_tool_burst(self, tool_count: int) -> None:
        """Accumulate tool-burst salience score per before_execute_tools call.

        Called once per batch (not once per iteration) to preserve the
        original per-batch accumulation semantics of the hook.
        """
        self._salience_score += tool_burst_score(tool_count)

    def score_iteration(
        self, had_error: bool, user_text: str | None
    ) -> None:
        """Accumulate error and correction salience for an iteration.

        Call once per iteration in after_iteration. Tool-burst salience is
        accumulated separately via record_tool_burst in before_execute_tools.
        Sets _nudge_pending when the threshold is crossed and resets score.
        """
        self._salience_score += error_score(had_error)
        self._salience_score += correction_score(user_text)

        if self._salience_score >= self._config.reflection.threshold:
            self._nudge_pending = True
            self._salience_score = 0.0

    # ------------------------------------------------------------------
    # Nudge delivery
    # ------------------------------------------------------------------

    def take_nudge(self) -> dict | None:
        """Return a nudge message dict and clear the flag, or return None."""
        if not self._nudge_pending:
            return None
        self._nudge_pending = False
        return {"role": "system", "content": _NUDGE_TEXT}

    # ------------------------------------------------------------------
    # Session-scoped reflections
    # ------------------------------------------------------------------

    def get_session_injections(self, session_id: int | None) -> list[dict]:
        """Return new reflection messages for this session since last watermark.

        Updates the watermark internally. Returns a list (may be empty).
        """
        if session_id is None:
            return []
        last_seen = self._last_injected_reflection_id.get(session_id, 0)
        limit = self._config.reflection.recent_limit
        rows = self._db.execute(
            "SELECT id, content FROM reflections "
            "WHERE session_id = ? AND id > ? "
            "ORDER BY id "
            "LIMIT ?",
            (session_id, last_seen, limit),
        ).fetchall()
        if not rows:
            return []
        rows = [(int(r[0]), r[1]) for r in rows]
        self._last_injected_reflection_id[session_id] = max(r[0] for r in rows)
        contents = [r[1] for r in rows]
        return [
            {
                "role": "system",
                "content": _format_reflection_reminder(contents),
            }
        ]

    # ------------------------------------------------------------------
    # Session boundary
    # ------------------------------------------------------------------

    def on_new_session(self, completed_session_id: int) -> None:
        """Handle a completed session: prune watermark dict and reset global watermark.

        Prunes _last_injected_reflection_id to bound its growth — once a
        session is done, its watermark entry is no longer needed.
        """
        self._last_injected_reflection_id.pop(completed_session_id, None)
        self._last_injected_global_reflection_id = 0

    # ------------------------------------------------------------------
    # Global (cross-session) reflections
    # ------------------------------------------------------------------

    async def get_global_injections(
        self, messages: list[dict], session_id: int | None
    ) -> list[dict]:
        """Embed first user message and find relevant cross-session reflections.

        Only called on iteration 0 of a new session when reflection_scope='global'.
        Returns a list of system message dicts (may be empty).
        """
        from ..session.archiver import _extract_text

        task_text = next(
            (
                _extract_text(m)
                for m in messages
                if m.get("role") == "user" and _extract_text(m)
            ),
            None,
        )
        if not task_text:
            return []
        try:
            limit = self._config.reflection.recent_limit
            rows = await self._fetch_global_reflections(task_text, limit)
            if not rows:
                return []
            fresh = [
                (rid, content)
                for rid, content, sid in rows
                if sid != session_id
                and rid > self._last_injected_global_reflection_id
            ]
            if not fresh:
                return []
            context_contents = [content for _, content in fresh]
            self._last_injected_global_reflection_id = max(rid for rid, _ in fresh)
            log.debug("global reflections injected: %d entries", len(fresh))
            return [
                {
                    "role": "system",
                    "content": (
                        "## Relevant reflections from past sessions\n"
                        "These were written in previous conversations on similar tasks — "
                        "use them to avoid repeating known pitfalls:\n"
                        + "\n".join(f"- {c}" for c in context_contents)
                    ),
                }
            ]
        except Exception:
            log.debug("global reflection injection failed", exc_info=True)
            return []

    async def _fetch_global_reflections(
        self, task_text: str, limit: int
    ) -> list[tuple[int, str, int]]:
        """Embed task_text and return top-limit reflections by similarity.

        Returns list of (reflection_id, content, session_id) tuples.
        """
        import numpy as np

        async with self._embedder_factory() as chain:
            [query_vec] = await chain.embed([task_text])

        vec_blob = query_vec.astype(np.float32).tobytes()
        fetch_k = min(limit * 4, 50)
        vec_rows = self._db.execute(
            "SELECT reflection_id, distance FROM reflections_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, fetch_k),
        ).fetchall()
        if not vec_rows:
            return []

        placeholders = ",".join("?" * len(vec_rows))
        id_to_distance = {r[0]: r[1] for r in vec_rows}
        ref_rows = self._db.execute(
            f"SELECT id, content, session_id FROM reflections WHERE id IN ({placeholders})",
            [r[0] for r in vec_rows],
        ).fetchall()

        results = sorted(
            [(r[0], r[1], r[2]) for r in ref_rows],
            key=lambda x: id_to_distance.get(x[0], 999.0),
        )
        return results[:limit]

    # ------------------------------------------------------------------
    # Trajectory injection
    # ------------------------------------------------------------------

    async def get_trajectory_injection(self, messages: list[dict]) -> dict | None:
        """Embed the first user message and return a matching past trajectory message.

        Returns None when no match is found or similarity is below threshold.
        """
        from ..session.archiver import _extract_text

        task_text = next(
            (
                _extract_text(m)
                for m in messages
                if m.get("role") == "user" and _extract_text(m)
            ),
            None,
        )
        if not task_text:
            return None
        try:
            import json as _json

            import numpy as np

            async with self._embedder_factory() as chain:
                [vec] = await chain.embed([task_text])

            vec_blob = vec.astype(np.float32).tobytes()
            rows = self._db.execute(
                "SELECT trajectory_id, distance FROM trajectories_vec "
                "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
                (vec_blob,),
            ).fetchall()
            if not rows:
                return None

            traj_id, distance = rows[0]
            similarity = 1.0 - float(distance)
            if similarity < self._config.trajectory.inject_min_similarity:
                return None

            row = self._db.execute(
                "SELECT task, skills_used, outcome, reflection FROM trajectories WHERE id = ?",
                (traj_id,),
            ).fetchone()
            if not row:
                return None

            task, skills_used_json, outcome, reflection = row
            skills = _json.loads(skills_used_json) if skills_used_json else []
            skill_str = ", ".join(skills) if skills else "none"

            lines = [
                "## Relevant past session",
                f"A similar task previously ended with outcome: {outcome}.",
                f"Task: {task[:200]}",
                f"Skills used: {skill_str}",
            ]
            if reflection:
                lines.append(f"Reflection: {reflection.splitlines()[0][:300]}")

            log.debug(
                "trajectory context injected: id=%d similarity=%.3f", traj_id, similarity
            )
            return {"role": "system", "content": "\n".join(lines)}
        except Exception:
            log.debug("trajectory context injection failed", exc_info=True)
            return None


def _format_reflection_reminder(contents: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in contents)
    return (
        "## Reflections from earlier in this session\n"
        "Notes you wrote down earlier in this conversation — use "
        "them to avoid repeating mistakes and carry lessons forward:\n"
        f"{bullets}"
    )
