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
from .paths import state_db
from .session.db import evict_low_value_facts, open_db, purge_older_than
from .session.trajectory import TrajectoryWriter
from .skills.indexer import SkillIndexer

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

log = logging.getLogger(__name__)

# Meta-KV key tracking the last full VACUUM, for the purge cooldown gate.
_META_LAST_VACUUM = "vacuum.last_run_at"

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
            loop.workspace,
            config.embedding.target_dims,
            busy_timeout_ms=config.sqlite_busy_timeout_ms,
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
        # Tracks nanobot /goal transitions; emits "started" / "completed"
        # events from observable runtime-context markers (no SessionManager
        # internals — survives nanobot version drift).
        from .coordinator.goal import GoalTracker  # noqa: PLC0415
        self._goal_tracker = GoalTracker()

        # Per-iteration tool call counter (accumulated across before_execute_tools).
        self._tool_calls = 0

        # Injection timing: system messages queued for current iteration's model call.
        self._pending_injections: list[dict] = []

        # Background purge task handle (retained so tests can await it).
        self._purge_task: asyncio.Task | None = None

        # Evolution task handle — GEPA + rewriter, scheduled at session boundaries.
        self._evolution_task: asyncio.Task | None = None

        # Completed-session counter for evolution trigger cadence.
        self._completed_session_count: int = 0

        # Pending reflection embedding tasks — reflect/tool.py registers tasks here
        # so asyncio doesn't GC them before they complete.
        self._reflection_embed_tasks: set[asyncio.Task] = set()

        # Cache for last_user_text: (id(messages), len(messages), result).
        # Avoids a full reverse scan every after_iteration when nothing changed.
        self._last_user_text_cache: tuple[int, int, str | None] = (0, 0, None)
        # Count of user-role messages seen — drives the save-nudge cadence by
        # detecting new turns via list growth (text equality misses repeated
        # identical prompts like "go", "again").
        self._last_user_msg_count: int = 0

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

    async def _check_promotions(self, names: list[str]) -> None:
        """Delegate to SkillUsageTracker. Called by rate_tool.py."""
        if self.config.skill_stats.reconstruction_check_enabled:
            names = await self._filter_reconstruction_blocked(names)
        self._skill_tracker.check_promotions(names)
        from .skills.reflection_trigger import check_skill_reflection_triggers  # noqa: PLC0415
        suggestions = check_skill_reflection_triggers(
            self.db, names, self.config.skill_stats, self._skill_reflection_triggered
        )
        if suggestions:
            self._reflection_coord.queue_skill_suggestions(suggestions)

    async def _filter_reconstruction_blocked(self, names: list[str]) -> list[str]:
        """Remove names whose draft skill body doesn't match their description.

        Only runs the check for draft skills at or above the promotion threshold
        — passing skills and active/deprecated skills are returned unchanged.
        """
        from .skills.reconstruction import check_reconstruction  # noqa: PLC0415
        import asyncio  # noqa: PLC0415

        if not names:
            return []

        cfg = self.config.skill_stats

        # Batch fetch statuses for all candidates at once.
        placeholders = ",".join("?" * len(names))
        rows = self.db.execute(
            f"SELECT name, status, success_count FROM skill_stats WHERE name IN ({placeholders})",
            names,
        ).fetchall()
        status_map = {r[0]: (r[1], r[2]) for r in rows}

        passed: list[str] = []
        for name in names:
            info = status_map.get(name)
            if info is None or info[0] != "draft" or info[1] < cfg.promotion_threshold:
                passed.append(name)
                continue

            skill_path = self.workspace / "skills" / name / "SKILL.md"
            if not skill_path.exists():
                passed.append(name)
                continue

            raw = await asyncio.to_thread(skill_path.read_text, encoding="utf-8")
            # Extract frontmatter description and body.
            # If frontmatter is malformed (missing closing ---), skip the check
            # rather than passing an empty description that biases LLM toward YES.
            description = ""
            body = raw
            if raw.startswith("---"):
                end = raw.find("\n---", 3)
                if end == -1:
                    log.warning(
                        "hook: %s — malformed SKILL.md frontmatter, skipping reconstruction check",
                        name,
                    )
                    passed.append(name)
                    continue
                fm_text = raw[4:end]
                for line in fm_text.splitlines():
                    if line.startswith("description:"):
                        description = line.partition(":")[2].strip()
                body = raw[end + len("\n---"):].lstrip("\n")

            ok = await check_reconstruction(
                self,
                skill_name=name,
                description=description,
                body=body,
            )
            if ok:
                passed.append(name)
            else:
                log.info(
                    "hook: reconstruction check blocked promotion of '%s'", name
                )
        return passed

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
        self._reflection_coord.reset_iteration_citations()

        # Schedule retention purge as a non-blocking background task on first iteration.
        if context.iteration == 0:
            self._purge_task = asyncio.create_task(self._background_purge())
            self._curator_task = asyncio.create_task(self._background_curator())

        # Keep current_session_id in sync with archiver (lazy-bootstrap on first call).
        self._sync_session(context.messages)

        # Inject matching principles on iteration 0 (pure FTS5, always fast).
        if context.iteration == 0:
            for msg in self._reflection_coord.get_principle_injections(context.messages):
                self._inject(context.messages, msg)
            # Memory-save nudge hydration: count prior user turns in the
            # resumed conversation so the cadence picks up where it left
            # off rather than firing instantly on restart.
            recent_user_turns = sum(
                1 for m in context.messages if m.get("role") == "user"
            )
            self._last_user_msg_count = recent_user_turns
            self._reflection_coord.hydrate_save_counter_from_history(
                recent_user_turns=recent_user_turns,
            )

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

        # Deliver memory-save cadence nudge if armed last iteration.
        save_nudge = self._reflection_coord.take_save_nudge()
        if save_nudge:
            self._inject(context.messages, save_nudge)

        # Deliver goal-completion nudge if a /goal wrapped up last iteration.
        goal_nudge = self._reflection_coord.take_goal_completion()
        if goal_nudge:
            self._inject(context.messages, goal_nudge)

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
        msgs = context.messages
        cache_key = (id(msgs), len(msgs))
        if (self._last_user_text_cache[0], self._last_user_text_cache[1]) != cache_key:
            self._last_user_text_cache = (*cache_key, last_user_text(msgs))
        current_user_text = self._last_user_text_cache[2]
        self._reflection_coord.score_iteration(
            had_error=context.error is not None,
            user_text=current_user_text,
        )

        # Memory-save cadence: count fresh user turns by user-message count
        # growth, not text equality — two identical "go" messages should
        # advance the counter by two, not one.
        user_msg_count = sum(1 for m in msgs if m.get("role") == "user")
        new_user_turns = user_msg_count - self._last_user_msg_count
        if new_user_turns > 0:
            for _ in range(new_user_turns):
                self._reflection_coord.note_user_turn()
            self._last_user_msg_count = user_msg_count

        # RMM: bump view/cite counters for reflections injected this iteration
        # against the assistant response text.
        response_text = context.final_content or ""
        self._reflection_coord.record_iteration_citations(response_text)

        # /goal transitions: when a sustained goal completes, the runtime
        # context drops the "Goal (active):" marker. Promote that into a
        # reflection nudge so the agent distils what was learned while
        # the objective is still fresh in the conversation.
        transition = self._goal_tracker.update(context.messages)
        if transition == "completed":
            objective = self._goal_tracker.last_objective or "(objective text unavailable)"
            log.info("nano-hermes: /goal completed — queuing goal-completion nudge")
            # Push enough salience to arm the Reflexion nudge on the next
            # iteration's score_iteration call even on an otherwise quiet turn.
            self._reflection_coord.add_salience(
                self.config.reflection.threshold + 1.0
            )
            self._reflection_coord.queue_goal_completion(objective)

        log.debug(
            "after_iteration: iter=%d tool_calls=%d session=%s",
            context.iteration,
            self._tool_calls,
            self.current_session_id,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _maybe_schedule_evolution(self) -> None:
        """Schedule a GEPA+rewriter background task if the cadence config fires."""
        interval = self.config.skill_stats.rewrite_session_interval
        if interval <= 0:
            return
        if self._completed_session_count % interval != 0:
            return
        # Don't stack tasks — wait for the previous cycle to finish first.
        if self._evolution_task and not self._evolution_task.done():
            log.debug("evolution cycle still running — skipping this session boundary")
            return
        self._evolution_task = asyncio.create_task(self._run_evolution_cycle())

    async def _run_evolution_cycle(self) -> None:
        """Run GEPA (if enabled) then the failure-driven rewriter, non-blocking."""
        from .skills.gepa import run_gepa  # noqa: PLC0415
        from .skills.rewriter import run_rewriter  # noqa: PLC0415

        gepa_evolved: list[str] = []
        try:
            gepa_evolved = await run_gepa(self)
            if gepa_evolved:
                log.info("evolution cycle: GEPA updated %d skill(s): %s", len(gepa_evolved), gepa_evolved)
        except Exception:
            log.exception("evolution cycle: GEPA failed")

        try:
            rewritten = await run_rewriter(self, skip=frozenset(gepa_evolved))
            if rewritten:
                log.info("evolution cycle: rewriter updated %d skill(s): %s", len(rewritten), rewritten)
        except Exception:
            log.exception("evolution cycle: rewriter failed")

    async def _background_curator(self) -> None:
        """Run the curator on the main loop after a short delay.

        sqlite3 connections aren't thread-safe by default so we stay on
        the event loop; the curator is pure SQL (one SELECT, a few
        UPDATEs) and takes sub-millisecond time even on Pi 3B+.
        """
        await asyncio.sleep(3)  # let iteration 0's session sync finish
        try:
            from .skills.curator import run_curator  # noqa: PLC0415
            archived = run_curator(self)
            if archived:
                log.info("curator: archived %d stale skill(s): %s", len(archived), archived)
        except Exception:
            log.exception("nano-hermes curator failed")

    async def _background_purge(self) -> None:
        """Run retention purge off the event loop via a short-lived DB connection."""
        # Defer briefly so iteration 0's session sync finishes before the
        # follow-up VACUUM (when needed) takes its short exclusive lock.
        await asyncio.sleep(2)
        db_path = state_db(self.workspace)
        days = self.config.trajectory_retention_days
        vacuum_interval_s = self.config.vacuum_min_interval_days * 86400
        busy_timeout_ms = self.config.sqlite_busy_timeout_ms
        decay = self.config.decay

        def _run() -> None:
            import sqlite3  # noqa: PLC0415
            import time  # noqa: PLC0415

            import sqlite_vec  # noqa: PLC0415

            from .skills.curator import meta_get, meta_set  # noqa: PLC0415

            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(f"PRAGMA busy_timeout = {int(busy_timeout_ms)}")
            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
            try:
                result = purge_older_than(conn, days)
                # Evict low-value semantic_facts — the one store retention
                # purge doesn't touch, so it grows without bound on a Pi.
                evicted = 0
                if decay.enabled:
                    evicted = evict_low_value_facts(
                        conn,
                        retention_days=decay.fact_retention_days,
                        importance_floor=decay.fact_evict_importance_floor,
                        superseded_grace_days=decay.superseded_grace_days,
                        max_per_run=decay.max_evictions_per_run,
                    )
                # Reclaim space when rows were actually removed. FTS5 and vec0
                # shadow tables don't return pages without an explicit VACUUM.
                # Gate behind a cooldown: once data ages past the retention
                # window every startup removes a fresh day of rows, so an
                # unconditional VACUUM would rewrite the whole DB on each boot,
                # taking an exclusive lock that contends with the live archiver.
                if result["sessions"] or result["trajectories"] or evicted:
                    now = time.time()
                    last_raw = meta_get(conn, _META_LAST_VACUUM)
                    last = float(last_raw) if last_raw else 0.0
                    if now - last >= vacuum_interval_s:
                        conn.isolation_level = None  # VACUUM cannot run in a txn
                        conn.execute("VACUUM")
                        conn.isolation_level = ""  # restore default for meta_set
                        meta_set(conn, _META_LAST_VACUUM, str(now))
                        log.info(
                            "nano-hermes purge: %d sessions, %d trajectories, "
                            "%d facts — VACUUMed",
                            result["sessions"],
                            result["trajectories"],
                            evicted,
                        )
                    else:
                        log.info(
                            "nano-hermes purge: %d sessions, %d trajectories, "
                            "%d facts — VACUUM skipped (cooldown)",
                            result["sessions"],
                            result["trajectories"],
                            evicted,
                        )
            finally:
                conn.close()

        try:
            await asyncio.to_thread(_run)
        except Exception:
            log.exception("nano-hermes purge failed")

    def _sync_session(self, messages: list) -> None:
        """Sync session state and handle boundary crossings."""
        _existing, completed_id = self._session_coord.sync(messages)

        if completed_id is not None:
            # Session boundary: finalize trajectory, reset skill state, prune dicts.
            skills_used, skills_loaded, had_errors = self._skill_tracker.reset_session()
            if len(skills_loaded) >= 2:
                from .skills.composition import record_composition  # noqa: PLC0415
                record_composition(self.db, skills_loaded)
            self._reflection_coord.back_propagate_utility(had_errors)
            self._session_coord.finalize(completed_id, skills_used, had_errors)
            self._reflection_coord.on_new_session(completed_id)
            self.archiver.prune_session_by_id(completed_id)
            self._completed_session_count += 1
            self._maybe_schedule_evolution()
