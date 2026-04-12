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
from .session.db import open_db
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
        super().__init__(reraise=False)
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

    def embedder(self) -> EmbeddingChain:
        return EmbeddingChain(self.config.embedding)

    # ------------------------------------------------------------------
    # AgentHook lifecycle
    # ------------------------------------------------------------------

    async def before_iteration(self, context: AgentHookContext) -> None:
        self._tool_calls = 0
        self._errors = 0

        # Keep current_session_id in sync with the archiver, and
        # lazy-bootstrap a session row on the very first iteration of a
        # new messages list so the reflect tool has something to attach
        # to from iteration 0 onwards.
        self._sync_session_id(context.messages)

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
        try:
            self.archiver.archive_and_embed(context.messages)
        except Exception:
            log.exception("nano-hermes archive failed")
        self._sync_session_id(context.messages)

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

    def _sync_session_id(self, messages: list[dict]) -> None:
        existing = self.archiver.current_session_id(messages)
        if existing is None:
            try:
                self.archiver.archive_and_embed(messages)
                existing = self.archiver.current_session_id(messages)
            except Exception:
                log.exception("nano-hermes session bootstrap failed")
        self.current_session_id = existing

    def _unseen_reflections(self, session_id: int) -> list[tuple[int, str]]:
        last_seen = self._last_injected_reflection_id.get(session_id, 0)
        rows = self.db.execute(
            "SELECT id, content FROM reflections "
            "WHERE session_id = ? AND id > ? "
            "ORDER BY id",
            (session_id, last_seen),
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
