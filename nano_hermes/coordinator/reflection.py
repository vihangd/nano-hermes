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
        # Queued skill-specific reflection suggestions.
        self._skill_suggestions: list[str] = []
        # Watermark: highest reflection id injected per session.
        # Bounded by on_new_session pruning completed sessions.
        self._last_injected_reflection_id: dict[int, int] = {}
        # Watermark for global (cross-session) reflections.
        self._last_injected_global_reflection_id: int = 0
        # MemRL: set of reflection IDs injected during the current session.
        # Populated by get_session_injections / get_global_injections.
        # Back-propagated at session boundary via back_propagate_utility().
        self._injected_reflection_ids: set[int] = set()

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

    def queue_skill_suggestions(self, suggestions: list[str]) -> None:
        """Queue skill-quality reflection suggestions for next injection."""
        self._skill_suggestions.extend(suggestions)

    def take_skill_suggestions(self) -> dict | None:
        """Return queued skill suggestions as a system message, or None."""
        if not self._skill_suggestions:
            return None
        bullets = "\n".join(f"- {s}" for s in self._skill_suggestions)
        self._skill_suggestions = []
        return {
            "role": "system",
            "content": (
                "## Skill quality signals\n"
                "One or more skills you use have a mixed success rate. "
                "When convenient, use reflect() to note what works and what doesn't:\n"
                + bullets
            ),
        }

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
        for rid, _ in rows:
            self._injected_reflection_ids.add(rid)
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
        self._injected_reflection_ids.clear()

    def back_propagate_utility(self, had_errors: bool) -> None:
        """MemRL: update utility scores for reflections injected this session.

        Called at session boundary with the session's outcome. Applies a
        simple temporal-difference update:
          - Success (no errors): utility += α * (1.0 - utility)
          - Failure (had errors): utility += α * (0.0 - utility)
        α = 0.1 (conservative step size to avoid oscillation).

        Also records co-activation edges between every pair of reflections
        injected together in this session (associative graph, item 10).
        """
        if not self._injected_reflection_ids:
            return
        alpha = 0.1
        reward = 0.0 if had_errors else 1.0
        placeholders = ",".join("?" * len(self._injected_reflection_ids))
        try:
            rows = self._db.execute(
                f"SELECT id, utility FROM reflections WHERE id IN ({placeholders})",
                list(self._injected_reflection_ids),
            ).fetchall()
            updates = [
                (current_utility + alpha * (reward - current_utility), rid)
                for rid, current_utility in rows
            ]
            self._db.executemany(
                "UPDATE reflections SET utility = ? WHERE id = ?",
                updates,
            )
            self._db.commit()
            log.debug(
                "utility back-propagated: %d reflections reward=%.1f",
                len(rows),
                reward,
            )
        except Exception:
            log.debug("utility back-propagation failed", exc_info=True)

        self._record_coactivations()

    _MAX_COACTIVATION_IDS = 20  # cap to keep O(N²) bounded per session

    def _record_coactivations(self) -> None:
        """Record pairwise co-activation edges for all injected reflection IDs.

        Single upsert per pair via ON CONFLICT DO UPDATE — avoids the
        two-statement INSERT OR IGNORE + UPDATE pattern.  Pairs stored with
        smaller ID first so each undirected edge is stored once.
        """
        # Cap to prevent O(N²) blow-up on very long sessions.
        ids = sorted(self._injected_reflection_ids)[-self._MAX_COACTIVATION_IDS:]
        if len(ids) < 2:
            return
        import itertools
        import time as _time
        now = _time.time()
        try:
            self._db.executemany(
                "INSERT INTO reflection_coactivations "
                "(reflection_a_id, reflection_b_id, coactivation_count, last_at) "
                "VALUES (?, ?, 1, ?) "
                "ON CONFLICT(reflection_a_id, reflection_b_id) DO UPDATE SET "
                "coactivation_count = coactivation_count + 1, last_at = excluded.last_at",
                ((a, b, now) for a, b in itertools.combinations(ids, 2)),
            )
            self._db.commit()
            log.debug("co-activation edges recorded: %d pairs", len(ids) * (len(ids) - 1) // 2)
        except Exception:
            log.debug("co-activation recording failed", exc_info=True)

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
            for rid, _ in fresh:
                self._injected_reflection_ids.add(rid)
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
            f"SELECT id, content, session_id, utility FROM reflections WHERE id IN ({placeholders})",
            [r[0] for r in vec_rows],
        ).fetchall()

        min_sim = getattr(self._config.reflection, "global_inject_min_similarity", 0.60)
        utility_weight = 0.3
        scored = []
        for r in ref_rows:
            rid, content, sid, utility = r[0], r[1], r[2], r[3] if len(r) > 3 else 0.5
            cosine_sim = 1.0 - id_to_distance.get(rid, 999.0)
            if cosine_sim < min_sim:
                continue
            # Combined score: cosine similarity + utility bonus.
            combined = cosine_sim + utility_weight * utility
            scored.append((rid, content, sid, combined))

        results = sorted(scored, key=lambda x: -x[3])
        return [(r[0], r[1], r[2]) for r in results[:limit]]

    # ------------------------------------------------------------------
    # Trajectory injection
    # ------------------------------------------------------------------

    async def get_trajectory_injection(self, messages: list[dict]) -> dict | None:
        """Embed the first user message and return up to 2 relevant past trajectories.

        Uses MMR (Maximal Marginal Relevance) to select a diverse pair:
        - d1 = most similar to current task
        - d2 = highest MMR score among remaining candidates (relevance - diversity penalty)
        Diversity is measured by Jaccard similarity on tokenized task text, requiring
        no extra embedding round-trips.

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
            mmr_lambda = self._config.retrieval.mmr_lambda
            min_sim = self._config.trajectory.inject_min_similarity
            fetch_k = 6

            rows = self._db.execute(
                "SELECT trajectory_id, distance FROM trajectories_vec "
                "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
                (vec_blob, fetch_k),
            ).fetchall()
            if not rows:
                return None

            # Filter by minimum similarity threshold.
            candidates = [
                (int(r[0]), 1.0 - float(r[1]))
                for r in rows
                if (1.0 - float(r[1])) >= min_sim
            ]
            if not candidates:
                return None

            # Load trajectory details for all candidates.
            placeholders = ",".join("?" * len(candidates))
            traj_rows = self._db.execute(
                f"SELECT id, task, skills_used, outcome, reflection FROM trajectories "
                f"WHERE id IN ({placeholders})",
                [c[0] for c in candidates],
            ).fetchall()
            detail = {r[0]: r[1:] for r in traj_rows}

            # d1 = highest similarity candidate.
            d1_id, d1_sim = candidates[0]
            if d1_id not in detail:
                return None

            selected = [d1_id]
            d1_task = detail[d1_id][0] or ""

            # d2 = best MMR score among remaining candidates.
            d1_tokens = set(d1_task.lower().split())
            best_mmr, best_id = -1.0, None
            for cid, csim in candidates[1:]:
                if cid not in detail:
                    continue
                c_task = detail[cid][0] or ""
                c_tokens = set(c_task.lower().split())
                union = d1_tokens | c_tokens
                jaccard = len(d1_tokens & c_tokens) / len(union) if union else 0.0
                mmr_score = mmr_lambda * csim - (1.0 - mmr_lambda) * jaccard
                if mmr_score > best_mmr:
                    best_mmr, best_id = mmr_score, cid
            if best_id is not None:
                selected.append(best_id)

            # Build injection message.
            sections: list[str] = []
            for i, tid in enumerate(selected, start=1):
                task_txt, skills_used_json, outcome, reflection = detail[tid]
                skills = _json.loads(skills_used_json) if skills_used_json else []
                skill_str = ", ".join(skills) if skills else "none"
                label = "Most similar" if i == 1 else "Contrasting"
                lines = [
                    f"### Past session {i} ({label})",
                    f"Outcome: {outcome}  |  Skills: {skill_str}",
                    f"Task: {task_txt[:200]}",
                ]
                if reflection:
                    lines.append(f"Reflection: {reflection.splitlines()[0][:300]}")
                sections.append("\n".join(lines))

            log.debug(
                "trajectory context injected: %d sessions (MMR)", len(selected)
            )
            header = "## Relevant past sessions"
            return {"role": "system", "content": header + "\n" + "\n\n".join(sections)}
        except Exception:
            log.debug("trajectory context injection failed", exc_info=True)
            return None


    # ------------------------------------------------------------------
    # Principle injection (EvolveR)
    # ------------------------------------------------------------------

    def get_principle_injections(self, messages: list[dict], limit: int = 3) -> list[dict]:
        """Return stored principles whose condition FTS5-matches the current task.

        Called on iteration 0 of each session. Uses keyword search (no embedding)
        so it never blocks on provider availability.  Returns a list of system
        message dicts (may be empty).
        """
        import re as _re  # noqa: PLC0415
        from ..session.archiver import _extract_text  # noqa: PLC0415

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

        # Build an FTS5 OR query from the unique substantive words in the task
        # text (>= 4 chars), so partial matches surface relevant principles.
        words = _re.findall(r"[a-zA-Z][a-zA-Z0-9_]*", task_text[:500])
        terms = list(dict.fromkeys(w.lower() for w in words if len(w) >= 4))[:12]
        if not terms:
            return []
        fts_query = " OR ".join(terms)
        try:
            rows = self._db.execute(
                "SELECT p.id, p.condition, p.action, p.expected_outcome "
                "FROM principles_fts pf "
                "JOIN principles p ON p.id = pf.content_id "
                "WHERE principles_fts MATCH ? "
                "ORDER BY pf.rank "
                "LIMIT ?",
                (fts_query, limit),
            ).fetchall()
        except Exception:
            log.debug("principle injection FTS5 query failed", exc_info=True)
            return []

        if not rows:
            return []

        # Increment use_count for matched principles
        ids = [r[0] for r in rows]
        try:
            ph = ",".join("?" * len(ids))
            self._db.execute(
                f"UPDATE principles SET use_count = use_count + 1 WHERE id IN ({ph})",
                ids,
            )
            self._db.commit()
        except Exception:
            pass

        lines: list[str] = []
        for _, condition, action, expected_outcome in rows:
            parts = [f"• If: {condition}", f"  Then: {action}"]
            if expected_outcome:
                parts.append(f"  Outcome: {expected_outcome}")
            lines.append("\n".join(parts))

        return [
            {
                "role": "system",
                "content": (
                    "## Relevant principles\n"
                    "These if-then rules apply to your current task:\n"
                    + "\n\n".join(lines)
                ),
            }
        ]


def _format_reflection_reminder(contents: list[str]) -> str:
    bullets = "\n".join(f"- {c}" for c in contents)
    return (
        "## Reflections from earlier in this session\n"
        "Notes you wrote down earlier in this conversation — use "
        "them to avoid repeating mistakes and carry lessons forward:\n"
        f"{bullets}"
    )
