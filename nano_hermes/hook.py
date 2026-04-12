"""NanoHermesHook — attached to nanobot's AgentLoop via ``install()``.

Responsibilities:

- ``before_iteration``:
    1. Reset per-iteration counters.
    2. Lazy-bootstrap the current archive session (on the very first
       iteration of a run we don't have one yet, and the reflect tool
       needs it).
    3. Inject any NEW reflections written earlier in this session, plus
       a pending Reflexion nudge if salience crossed the threshold last
       iteration.
- ``before_execute_tools``: score tool-call bursts for salience.
- ``after_iteration``: archive new messages, score errors and user
  corrections, update :attr:`current_session_id`, flip
  :attr:`_nudge_pending` when salience crosses the threshold.

What this hook still does NOT do: inject memory or skills into the
system prompt. Nanobot's ``ContextBuilder`` already does that via
``MemoryStore.get_memory_context()`` and
``SkillsLoader.build_skills_summary()``. We only add reflections and
salience nudges — both transient, per-session.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING

from nanobot.agent.hook import AgentHook, AgentHookContext

from .config import NanoHermesConfig
from .embedding.chain import EmbeddingChain
from .memory.budgets import BudgetedMemory
from .reflect.salience import (
    correction_score,
    error_score,
    last_user_text,
    tool_burst_score,
)
from .session.archiver import SessionArchiver
from .session.db import open_db, purge_older_than
from .session.trajectory import TrajectoryWriter
from .skills.indexer import SkillIndexer

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

log = logging.getLogger(__name__)

_NUDGE_TEXT = (
    "You've just had a tool-heavy or error-prone iteration. When you "
    "have a moment, call reflect(content='...') with a concise takeaway "
    "(2-4 sentences): what worked, what didn't, and what you'd do "
    "differently. This helps you avoid repeating the pattern in the "
    "same session."
)


class NanoHermesHook(AgentHook):
    def __init__(self, *, config: NanoHermesConfig, loop: "AgentLoop") -> None:
        # reraise=False so errors inside our hook are caught + logged by
        # nanobot's CompositeHook. Flip to True when debugging.
        # Guard against older nanobot versions where AgentHook.__init__
        # doesn't accept reraise yet.
        try:
            super().__init__(reraise=False)
        except TypeError:
            super().__init__()
        self.config = config
        self.workspace = loop.workspace
        self.budgeted_memory = BudgetedMemory(
            store=loop.context.memory,
            budgets=config.memory,
        )
        self.db: sqlite3.Connection = open_db(
            loop.workspace, config.embedding.target_dims
        )
        self.archiver = SessionArchiver(
            db=self.db,
            embedder_factory=self.embedder,
            target_dims=config.embedding.target_dims,
        )
        self.skill_indexer = SkillIndexer(
            db=self.db,
            skills_loader=loop.context.skills,
            embedder_factory=self.embedder,
            stats_config=config.skill_stats,
        )
        self.trajectory_writer = TrajectoryWriter(
            db=self.db,
            embedder_factory=self.embedder,
        )
        # Reflexion state
        self.current_session_id: int | None = None
        self._salience_score: float = 0.0
        self._nudge_pending: bool = False
        # Highest reflection id we've already injected, keyed by session
        # so reflections aren't replayed into the prompt every iteration.
        self._last_injected_reflection_id: dict[int, int] = {}
        # Per-iteration counters
        self._tool_calls = 0
        self._errors = 0
        # Phase 2: skill candidate tracking (reset each iteration, trajectory only)
        self._candidate_skills: list[str] = []
        # Phase 7: observed skill reads — tracks which skills the agent actually
        # loaded via read_file(skills/<name>/SKILL.md) this iteration.
        # Maps skill_name → tool_calls index so we can scope error attribution.
        self._loaded_skills: dict[str, int] = {}
        # Phase 2: session-level accumulators (reset at session boundary)
        self._session_skills_used: set[str] = set()
        self._session_had_errors: bool = False
        # Phase 5: async tasks for reflection embedding (global scope mode)
        self._reflection_embed_tasks: set = set()
        # Phase 5: track which global reflections we've already injected
        # (across all sessions) to avoid re-injecting on every iteration.
        self._last_injected_global_reflection_id: int = 0

    def embedder(self) -> EmbeddingChain:
        return EmbeddingChain(self.config.embedding)

    def record_skill_candidates(self, names: list[str]) -> None:
        """Called by SkillSearchTool to register skills returned this iteration.

        These are recorded for trajectory tracking (which skills the agent
        searched for), NOT for stat crediting. Stat crediting is based on
        directly-observed read_file calls on skills/*/SKILL.md paths.

        Extends (not replaces) so multiple skill_search calls in one iteration
        are all captured for trajectory purposes.
        """
        self._candidate_skills.extend(names)

    # ------------------------------------------------------------------
    # AgentHook lifecycle
    # ------------------------------------------------------------------

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._tool_calls = 0
        self._errors = 0
        self._candidate_skills = []
        self._loaded_skills = {}

        # Run retention purge once per session (first iteration only)
        if context.iteration == 0:
            try:
                purge_older_than(self.db, self.config.trajectory_retention_days)
            except Exception:
                log.exception("nano-hermes purge failed")

        # Keep current_session_id in sync with the archiver, and
        # lazy-bootstrap a session row on the very first iteration of a
        # new messages list so the reflect tool has something to attach
        # to from iteration 0 onwards.
        self._sync_session_id(context.messages)

        # Phase 3: inject a matching past trajectory on the first iteration
        # of a new session (opt-in via config).
        if context.iteration == 0 and self.config.trajectory.inject_context:
            await self._maybe_inject_trajectory(context.messages)

        # Phase 5: inject relevant cross-session reflections on first iteration
        # of a new session when global scope is enabled.
        if (
            context.iteration == 0
            and self.config.reflection_scope == "global"
            and self.current_session_id is not None
        ):
            await self._maybe_inject_global_reflections(context.messages)

        # Inject any new reflections written since the last iteration
        # (always session-scoped so the agent gets immediate feedback).
        if self.current_session_id is not None:
            new_reflections = self._unseen_reflections(self.current_session_id)
            if new_reflections:
                context.messages.append(
                    {
                        "role": "system",
                        "content": self._format_reflection_reminder(
                            [r[1] for r in new_reflections]
                        ),
                    }
                )
                self._last_injected_reflection_id[self.current_session_id] = (
                    max(r[0] for r in new_reflections)
                )

        # Deliver the Reflexion nudge if salience crossed the threshold
        # on the previous iteration.
        if self._nudge_pending:
            context.messages.append(
                {"role": "system", "content": _NUDGE_TEXT}
            )
            self._nudge_pending = False

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        tool_count = len(context.tool_calls)
        self._tool_calls += tool_count
        self._salience_score += tool_burst_score(tool_count)

        # Detect which skills the agent actually loads.
        # nanobot's skill template tells the agent: "To use a skill, read its
        # SKILL.md file using the read_file tool." So a read_file call on
        # skills/<name>/SKILL.md is the directly-observable usage signal.
        # Use getattr() defensively: Mock's constructor reserves 'name' and
        # 'arguments' may not be accessible via normal attribute lookup in tests.
        for i, tc in enumerate(context.tool_calls):
            if getattr(tc, "name", None) != "read_file":
                continue
            args = getattr(tc, "arguments", None) or {}
            path = args.get("path", "")
            skill_name = self._extract_skill_name_from_path(path)
            if skill_name:
                # Store the position so we can scope error attribution later.
                self._loaded_skills[skill_name] = i

    async def after_iteration(self, context: AgentHookContext) -> None:
        if context.error:
            self._errors += 1
            self._session_had_errors = True
        try:
            self.archiver.archive_and_embed(context.messages)
        except Exception:
            log.exception("nano-hermes archive failed")
        self._sync_session_id(context.messages)

        # Phase 2: credit candidate skills with usage if tool work happened
        self._update_skill_stats(context)

        # Salience from errors and user corrections this iteration.
        self._salience_score += error_score(context.error is not None)
        self._salience_score += correction_score(last_user_text(context.messages))

        if self._salience_score >= self.config.reflection.threshold:
            self._nudge_pending = True
            self._salience_score = 0.0

        log.debug(
            "after_iteration: iter=%d tool_calls=%d errors=%d salience=%.1f nudge=%s session=%s",
            context.iteration,
            self._tool_calls,
            self._errors,
            self._salience_score,
            self._nudge_pending,
            self.current_session_id,
        )

    # ------------------------------------------------------------------
    # Internals — reflection retrieval / formatting
    # ------------------------------------------------------------------

    async def _maybe_inject_global_reflections(self, messages: list[dict]) -> None:
        """Inject cross-session reflections relevant to the current task.

        Only fires on iteration 0 of a new session when
        ``reflection_scope="global"``. Embeds the first user message and
        searches ``reflections_vec`` for the top-N most relevant reflections
        from ANY past session, excluding the current one.
        """
        from .session.archiver import _extract_text

        task_text = next(
            (
                _extract_text(m)
                for m in messages
                if m.get("role") == "user" and _extract_text(m)
            ),
            None,
        )
        if not task_text:
            return
        try:
            limit = self.config.reflection.recent_limit
            rows = await self._global_reflections(task_text, limit)
            if not rows:
                return
            # Filter out reflections from the current session (those are
            # handled by _unseen_reflections) and already-injected ones.
            current_session = self.current_session_id
            fresh = [
                (rid, content)
                for rid, content, sid in rows
                if sid != current_session
                and rid > self._last_injected_global_reflection_id
            ]
            if not fresh:
                return
            context_contents = [content for _, content in fresh]
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "## Relevant reflections from past sessions\n"
                        "These were written in previous conversations on similar tasks — "
                        "use them to avoid repeating known pitfalls:\n"
                        + "\n".join(f"- {c}" for c in context_contents)
                    ),
                }
            )
            self._last_injected_global_reflection_id = max(rid for rid, _ in fresh)
            log.debug(
                "global reflections injected: %d entries", len(fresh)
            )
        except Exception:
            log.debug("global reflection injection failed", exc_info=True)

    async def _global_reflections(
        self, task_text: str, limit: int
    ) -> list[tuple[int, str, int]]:
        """Embed *task_text* and return top-*limit* reflections by similarity.

        Returns list of ``(reflection_id, content, session_id)`` tuples,
        ordered by embedding distance (closest first).
        Raises if embedding fails — caller logs and skips.
        """
        import numpy as np

        async with self.embedder() as chain:
            [query_vec] = await chain.embed([task_text])

        vec_blob = query_vec.astype(np.float32).tobytes()
        # Fetch a wider pool then slice — vec0 k parameter must be a literal
        fetch_k = min(limit * 4, 50)
        vec_rows = self.db.execute(
            "SELECT reflection_id, distance FROM reflections_vec "
            "WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (vec_blob, fetch_k),
        ).fetchall()
        if not vec_rows:
            return []

        placeholders = ",".join("?" * len(vec_rows))
        id_to_distance = {r[0]: r[1] for r in vec_rows}
        ref_rows = self.db.execute(
            f"SELECT id, content, session_id FROM reflections WHERE id IN ({placeholders})",
            [r[0] for r in vec_rows],
        ).fetchall()

        results = sorted(
            [(r[0], r[1], r[2]) for r in ref_rows],
            key=lambda x: id_to_distance.get(x[0], 999.0),
        )
        return results[:limit]

    async def _maybe_inject_trajectory(self, messages: list[dict]) -> None:
        """Embed the first user message and inject the most similar past trajectory."""
        from .session.archiver import _extract_text

        task_text = next(
            (
                _extract_text(m)
                for m in messages
                if m.get("role") == "user" and _extract_text(m)
            ),
            None,
        )
        if not task_text:
            return
        try:
            import numpy as np
            async with self.embedder() as chain:
                [vec] = await chain.embed([task_text])

            vec_blob = vec.astype(np.float32).tobytes()
            rows = self.db.execute(
                "SELECT trajectory_id, distance FROM trajectories_vec "
                "WHERE embedding MATCH ? AND k = 1 ORDER BY distance",
                (vec_blob,),
            ).fetchall()
            if not rows:
                return

            traj_id, distance = rows[0]
            similarity = 1.0 - float(distance)
            if similarity < self.config.trajectory.inject_min_similarity:
                return

            row = self.db.execute(
                "SELECT task, skills_used, outcome, reflection FROM trajectories WHERE id = ?",
                (traj_id,),
            ).fetchone()
            if not row:
                return

            task, skills_used_json, outcome, reflection = row
            import json as _json
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

            messages.append({"role": "system", "content": "\n".join(lines)})
            log.debug(
                "trajectory context injected: id=%d similarity=%.3f", traj_id, similarity
            )
        except Exception:
            log.debug("trajectory context injection failed", exc_info=True)

    def _sync_session_id(self, messages: list[dict]) -> None:
        existing = self.archiver.current_session_id(messages)
        if existing is None:
            try:
                self.archiver.archive_and_embed(messages)
                existing = self.archiver.current_session_id(messages)
            except Exception:
                log.exception("nano-hermes session bootstrap failed")

        prev_session = self.current_session_id
        # Session boundary: close the previous session and finalize trajectory.
        if existing is not None and existing != prev_session and prev_session is not None:
            # Mark the old session as ended so purge_older_than can find it.
            try:
                self.db.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                    (time.time(), prev_session),
                )
                self.db.commit()
            except Exception:
                log.exception("failed to set sessions.ended_at for session %d", prev_session)
            self._finalize_trajectory(prev_session)
            # Reset global reflection watermark for the new session.
            self._last_injected_global_reflection_id = 0

        self.current_session_id = existing

    def _update_skill_stats(self, context: AgentHookContext) -> None:
        """Track skills used this iteration for trajectory recording.

        This is trajectory-only — no DB writes, no stat crediting. The
        observed-use detection (read_file on SKILL.md) tells us which skills
        the agent consulted this iteration; we accumulate those into
        _session_skills_used so trajectories capture a complete skill picture.

        Stat crediting (use_count, success_count) and lifecycle transitions
        are handled exclusively by SkillRateTool, which the agent calls
        voluntarily with an explicit outcome judgment.
        """
        self._session_skills_used.update(self._candidate_skills)
        self._session_skills_used.update(self._loaded_skills.keys())

    def record_skill_rating(self, name: str) -> None:
        """Called by SkillRateTool to register a rated skill for trajectory."""
        self._session_skills_used.add(name)

    def _extract_skill_name_from_path(self, path: str) -> str | None:
        """Extract skill name from a path ending in skills/<name>/SKILL.md.

        Handles absolute paths, relative paths, and paths with './' prefixes.
        Returns None for any path that doesn't match the expected pattern.
        """
        try:
            parts = Path(path).parts
        except (TypeError, ValueError):
            return None
        for i, part in enumerate(parts):
            if part == "skills" and i + 2 < len(parts) and parts[i + 2] == "SKILL.md":
                return parts[i + 1]
        return None

    def _check_promotions(self, names: list[str]) -> None:
        """Promote draft skills to active, or deprecate skills with low success."""
        cfg = self.config.skill_stats
        try:
            with self.db:
                for name in names:
                    row = self.db.execute(
                        "SELECT status, use_count, success_count FROM skill_stats WHERE name = ?",
                        (name,),
                    ).fetchone()
                    if not row:
                        continue
                    status, use_count, success_count = row

                    # Promotion: draft -> active after enough successes
                    if status == "draft" and success_count >= cfg.promotion_threshold:
                        self.db.execute(
                            "UPDATE skill_stats SET status = 'active' WHERE name = ?",
                            (name,),
                        )
                        log.info(
                            "skill '%s' promoted draft -> active (success_count=%d)",
                            name, success_count,
                        )
                        status = "active"

                    # Deprecation: any non-deprecated skill with chronic low success rate
                    if (
                        status != "deprecated"
                        and use_count >= cfg.deprecation_min_uses
                        and use_count > 0
                        and success_count / use_count < cfg.deprecation_max_success_rate
                    ):
                        self.db.execute(
                            "UPDATE skill_stats SET status = 'deprecated' WHERE name = ?",
                            (name,),
                        )
                        log.info(
                            "skill '%s' deprecated (success_rate=%.2f after %d uses)",
                            name, success_count / use_count, use_count,
                        )
        except Exception:
            log.exception("skill promotion check failed")

    def _finalize_trajectory(self, session_id: int) -> None:
        """Write a trajectory row for a completed session and reset accumulators."""
        try:
            # Fallback: ensure ended_at is set even if _sync_session_id missed it.
            self.db.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), session_id),
            )
            self.db.commit()

            reflections = [
                r[1]
                for r in self.db.execute(
                    "SELECT id, content FROM reflections WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            ]
            chunks = self.db.execute(
                "SELECT role, content FROM chunks WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            messages = [{"role": r, "content": c} for r, c in chunks]
            self.trajectory_writer.write(
                session_id=session_id,
                messages=messages,
                skills_used=list(self._session_skills_used),
                reflections=reflections,
                had_errors=self._session_had_errors,
            )
        except Exception:
            log.exception("trajectory finalization failed for session %d", session_id)
        finally:
            self._session_skills_used = set()
            self._session_had_errors = False

    def _unseen_reflections(self, session_id: int) -> list[tuple[int, str]]:
        last_seen = self._last_injected_reflection_id.get(session_id, 0)
        limit = self.config.reflection.recent_limit
        rows = self.db.execute(
            "SELECT id, content FROM reflections "
            "WHERE session_id = ? AND id > ? "
            "ORDER BY id "
            "LIMIT ?",
            (session_id, last_seen, limit),
        ).fetchall()
        return [(int(r[0]), r[1]) for r in rows]

    @staticmethod
    def _format_reflection_reminder(contents: list[str]) -> str:
        bullets = "\n".join(f"- {c}" for c in contents)
        return (
            "## Reflections from earlier in this session\n"
            "Notes you wrote down earlier in this conversation — use "
            "them to avoid repeating mistakes and carry lessons forward:\n"
            f"{bullets}"
        )
