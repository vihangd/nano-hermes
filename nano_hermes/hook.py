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
        # Phase 2: skill candidate tracking (reset each iteration)
        self._candidate_skills: list[str] = []
        # Phase 2: session-level accumulators (reset at session boundary)
        self._session_skills_used: set[str] = set()
        self._session_had_errors: bool = False

    def embedder(self) -> EmbeddingChain:
        return EmbeddingChain(self.config.embedding)

    def record_skill_candidates(self, names: list[str]) -> None:
        """Called by SkillSearchTool to register skills returned this iteration."""
        self._candidate_skills = list(names)

    # ------------------------------------------------------------------
    # AgentHook lifecycle
    # ------------------------------------------------------------------

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._tool_calls = 0
        self._errors = 0
        self._candidate_skills = []

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

        # Inject any new reflections written since the last iteration.
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
        # Session boundary: finalize the previous session's trajectory
        if existing is not None and existing != prev_session and prev_session is not None:
            self._finalize_trajectory(prev_session)

        self.current_session_id = existing

    def _update_skill_stats(self, context: AgentHookContext) -> None:
        """Credit candidate skills with a use if the agent made tool calls."""
        if not self._candidate_skills or not context.tool_calls:
            return
        had_error = context.error is not None or any(
            ev.get("status") == "error" for ev in (context.tool_events or [])
        )
        now = time.time()
        session_id = self.current_session_id
        try:
            with self.db:
                for name in self._candidate_skills:
                    self.db.execute(
                        "UPDATE skill_stats SET "
                        "use_count = use_count + 1, "
                        "success_count = success_count + CASE WHEN ? THEN 1 ELSE 0 END, "
                        "last_used_at = ?, "
                        "provenance = json_insert(COALESCE(provenance, '[]'), '$[#]', ?) "
                        "WHERE name = ?",
                        (not had_error, now, session_id, name),
                    )
        except Exception:
            log.exception("skill_stats update failed")
        # Accumulate for the session-level trajectory
        self._session_skills_used.update(self._candidate_skills)

    def _finalize_trajectory(self, session_id: int) -> None:
        """Write a trajectory row for a completed session and reset accumulators."""
        try:
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
