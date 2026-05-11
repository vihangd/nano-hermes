"""NanoHermesHook — attached to nanobot's AgentLoop via ``install()``.

Thin orchestrator that delegates to three coordinator modules:
- SkillUsageTracker      (coordinator/skills.py)
- ReflectionCoordinator  (coordinator/reflection.py)
- SessionCoordinator     (coordinator/session.py)

Public API is unchanged — all attributes and methods that tools and tests
rely on remain on this class, delegating to coordinators as needed.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from typing import TYPE_CHECKING

from nanobot.agent.hook import AgentHook, AgentHookContext

from .config import NanoHermesConfig
from .coordinator.reflection import ReflectionCoordinator
from .coordinator.session import SessionCoordinator
from .coordinator.skills import SkillUsageTracker
from .embedding.chain import EmbeddingChain
from .memory.budgets import BudgetedMemory
from .reflect.salience import last_user_text
from .session.archiver import SessionArchiver
from .session.db import open_db, purge_older_than
from .session.trajectory import TrajectoryWriter
from .skills.indexer import SkillIndexer

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

log = logging.getLogger(__name__)

# Re-export for backward compatibility with any code that imports from here.
from .coordinator.reflection import _NUDGE_TEXT  # noqa: F401, E402


class NanoHermesHook(AgentHook):
    def __init__(self, *, config: NanoHermesConfig, loop: "AgentLoop") -> None:
        try:
            super().__init__(reraise=False)
        except TypeError:
            super().__init__()
        self.config = config
        self._loop = loop
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
            redact_secrets=config.redact_secrets,
        )
        self.skill_indexer = SkillIndexer(
            db=self.db,
            skills_loader=loop.context.skills,
            embedder_factory=self.embedder,
            stats_config=config.skill_stats,
            external_dirs=config.skills.external_dirs,
        )
        self.trajectory_writer = TrajectoryWriter(
            db=self.db,
            embedder_factory=self.embedder,
        )

        # Tracks which skills have already had a reflection suggestion injected.
        self._skill_reflection_triggered: set[str] = set()

        # Coordinators
        self._skill_tracker = SkillUsageTracker(
            db=self.db,
            config=config.skill_stats,
        )
        self._reflection_coord = ReflectionCoordinator(
            db=self.db,
            config=config,
            embedder_factory=self.embedder,
        )
        self._session_coord = SessionCoordinator(
            archiver=self.archiver,
            db=self.db,
            trajectory_writer=self.trajectory_writer,
        )

        # Per-iteration tool call counter (accumulated across before_execute_tools).
        self._tool_calls = 0

        # Injection timing: system messages queued for current iteration's model call.
        self._pending_injections: list[dict] = []

        # Background purge task handle (retained so tests can await it).
        self._purge_task: asyncio.Task | None = None

        # Pending reflection embedding tasks — reflect/tool.py registers tasks here
        # so asyncio doesn't GC them before they complete.
        self._reflection_embed_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API surface (unchanged for external callers)
    # ------------------------------------------------------------------

    @property
    def current_session_id(self) -> int | None:
        return self._session_coord.current_session_id

    @current_session_id.setter
    def current_session_id(self, value: int | None) -> None:
        self._session_coord.current_session_id = value

    def embedder(self) -> EmbeddingChain:
        return EmbeddingChain(self.config.embedding)

    # ------------------------------------------------------------------
    # Backward-compat proxy properties (tools and tests access these
    # directly; they delegate to the coordinator that owns the state)
    # ------------------------------------------------------------------

    @property
    def _nudge_pending(self) -> bool:
        return self._reflection_coord._nudge_pending

    @_nudge_pending.setter
    def _nudge_pending(self, value: bool) -> None:
        self._reflection_coord._nudge_pending = value

    @property
    def _salience_score(self) -> float:
        return self._reflection_coord._salience_score

    @property
    def _loaded_skills(self) -> dict:
        return self._skill_tracker._loaded_skills

    @_loaded_skills.setter
    def _loaded_skills(self, value: dict) -> None:
        self._skill_tracker._loaded_skills = value

    @property
    def _candidate_skills(self) -> list:
        return self._skill_tracker._candidate_skills

    @_candidate_skills.setter
    def _candidate_skills(self, value: list) -> None:
        self._skill_tracker._candidate_skills = value

    @property
    def _session_skills_used(self) -> set:
        return self._skill_tracker._session_skills_used

    def _extract_skill_name_from_path(self, path: str) -> str | None:
        return SkillUsageTracker.extract_skill_name_from_path(path)

    def record_skill_candidates(self, names: list[str]) -> None:
        """Called by SkillSearchTool to register skills returned this iteration."""
        self._skill_tracker.record_candidates(names)

    def record_skill_rating(self, name: str) -> None:
        """Called by SkillRateTool to register a rated skill for trajectory."""
        self._skill_tracker.record_rating(name)

    def _check_promotions(self, names: list[str]) -> None:
        """Delegate to SkillUsageTracker. Called by rate_tool.py."""
        self._skill_tracker.check_promotions(names)
        from .skills.reflection_trigger import check_skill_reflection_triggers  # noqa: PLC0415
        suggestions = check_skill_reflection_triggers(
            self.db, names, self.config.skill_stats, self._skill_reflection_triggered
        )
        if suggestions:
            self._reflection_coord.queue_skill_suggestions(suggestions)

    # ------------------------------------------------------------------
    # Injection timing mechanism (unchanged)
    # ------------------------------------------------------------------

    def _inject(self, messages: list, msg: dict) -> None:
        """Append msg to canonical messages and queue it for this iteration."""
        messages.append(msg)
        self._pending_injections.append(msg)

    def drain_injections(self) -> list[dict]:
        """Return and clear pending injections for the current iteration."""
        pending = self._pending_injections
        self._pending_injections = []
        return pending

    # ------------------------------------------------------------------
    # AgentHook lifecycle
    # ------------------------------------------------------------------

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._tool_calls = 0
        self._skill_tracker.reset_iteration()
        self._pending_injections = []  # clear stale entries from failed iteration

        # Schedule retention purge as a non-blocking background task on first iteration.
        if context.iteration == 0:
            self._purge_task = asyncio.get_event_loop().create_task(
                self._background_purge()
            )

        # Keep current_session_id in sync with archiver (lazy-bootstrap on first call).
        self._sync_session(context.messages)

        # Inject a matching past trajectory on iteration 0 (opt-in via config).
        if context.iteration == 0 and self.config.trajectory.inject_context:
            msg = await self._reflection_coord.get_trajectory_injection(
                context.messages
            )
            if msg:
                self._inject(context.messages, msg)

        # Inject cross-session reflections on iteration 0 when global scope is enabled.
        if (
            context.iteration == 0
            and self.config.reflection_scope == "global"
            and self.current_session_id is not None
        ):
            for msg in await self._reflection_coord.get_global_injections(
                context.messages, self.current_session_id
            ):
                self._inject(context.messages, msg)

        # Inject new session-scoped reflections written since last iteration.
        for msg in self._reflection_coord.get_session_injections(
            self.current_session_id
        ):
            self._inject(context.messages, msg)

        # Deliver Reflexion nudge if salience crossed threshold last iteration.
        nudge = self._reflection_coord.take_nudge()
        if nudge:
            self._inject(context.messages, nudge)

        # Deliver queued skill-quality reflection suggestions.
        skill_suggestions = self._reflection_coord.take_skill_suggestions()
        if skill_suggestions:
            self._inject(context.messages, skill_suggestions)

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        tool_count = len(context.tool_calls)
        self._tool_calls += tool_count
        # Accumulate tool-burst salience per-batch (preserves original semantics).
        self._reflection_coord.record_tool_burst(tool_count)

        # Detect which skills the agent actually loads via read_file on SKILL.md.
        for i, tc in enumerate(context.tool_calls):
            if getattr(tc, "name", None) != "read_file":
                continue
            args = getattr(tc, "arguments", None) or {}
            path = args.get("path", "")
            skill_name = SkillUsageTracker.extract_skill_name_from_path(path)
            if skill_name:
                self._skill_tracker.record_read(skill_name, i)

    async def after_iteration(self, context: AgentHookContext) -> None:
        if context.error:
            self._skill_tracker.record_error()

        try:
            self.archiver.archive_and_embed(context.messages)
        except Exception:
            log.exception("nano-hermes archive failed")

        self._sync_session(context.messages)

        # Accumulate skill observations into session-level state.
        self._skill_tracker.update_accumulators()

        # Score error and correction salience (tool-burst already accumulated
        # in before_execute_tools via record_tool_burst).
        self._reflection_coord.score_iteration(
            had_error=context.error is not None,
            user_text=last_user_text(context.messages),
        )

        log.debug(
            "after_iteration: iter=%d tool_calls=%d session=%s",
            context.iteration,
            self._tool_calls,
            self.current_session_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _background_purge(self) -> None:
        """Run retention purge. Scheduled as a background task (non-blocking)."""
        try:
            purge_older_than(self.db, self.config.trajectory_retention_days)
        except Exception:
            log.exception("nano-hermes purge failed")

    def _sync_session(self, messages: list) -> None:
        """Sync session state and handle boundary crossings."""
        _existing, completed_id = self._session_coord.sync(messages)

        if completed_id is not None:
            # Session boundary: finalize trajectory, reset skill state, prune dicts.
            skills_used, had_errors = self._skill_tracker.reset_session()
            if len(skills_used) >= 2:
                from .skills.composition import record_composition  # noqa: PLC0415
                record_composition(self.db, skills_used)
            self._session_coord.finalize(completed_id, skills_used, had_errors)
            self._reflection_coord.on_new_session(completed_id)
            self.archiver.prune_session_by_id(completed_id)
