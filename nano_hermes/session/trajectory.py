"""Write session-level trajectory rows into the ``trajectories`` table.

Called at session boundaries (when a new session starts) so each row
captures the full arc of one conversation: what the task was, which
skills were consulted, whether it succeeded, and any reflections written.

No LLM calls — task is the first user message (500-char cap), outcome is
a heuristic (ok/fail/partial), reflections are the raw reflection texts
stored by the reflect tool.

Phase 3: also embeds the task text and writes to ``trajectories_vec``
for semantic retrieval of similar past tasks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import time
from typing import Any, Callable

import numpy as np

from ..embedding.chain import AllProvidersFailed, EmbeddingChain
from .archiver import _extract_text
from .db import fts_guarded_write
from .db import run_vec_write

log = logging.getLogger(__name__)


class TrajectoryWriter:
    def __init__(
        self,
        db: sqlite3.Connection,
        embedder_factory: Callable[[], EmbeddingChain] | None = None,
    ) -> None:
        self._db = db
        self._embedder_factory = embedder_factory
        self._embed_tasks: set[asyncio.Task] = set()

    def write(
        self,
        *,
        session_id: int,
        messages: list[dict[str, Any]],
        skills_used: list[str],
        reflections: list[str],
        had_errors: bool,
    ) -> int | None:
        """Write one trajectory row. Returns the new row id, or None if
        no task could be extracted (e.g. empty or system-only session).

        If an embedder_factory is set, schedules async embedding of the
        task text into ``trajectories_vec``.
        """
        task = self._extract_task(messages)
        if not task:
            return None

        outcome = self._outcome(messages, had_errors)
        reflection_text = "\n".join(reflections) if reflections else None

        try:
            # trajectories_ai mirrors this row into trajectories_fts; a
            # corrupt index must not cost the canonical trajectory.
            cur = fts_guarded_write(
                self._db,
                "trajectories_fts",
                lambda: self._db.execute(
                    "INSERT INTO trajectories "
                    "(session_id, task, skills_used, outcome, reflection, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        session_id,
                        task,
                        json.dumps(skills_used),
                        outcome,
                        reflection_text,
                        time.time(),
                    ),
                ),
            )
            self._db.commit()
            row_id = int(cur.lastrowid)
            log.debug(
                "trajectory written: id=%d session=%d outcome=%s skills=%s",
                row_id,
                session_id,
                outcome,
                skills_used,
            )
            # Schedule async task embedding if embedder is available
            if self._embedder_factory is not None:
                self._schedule_embed(row_id, task)
            return row_id
        except Exception:
            log.exception("trajectory write failed for session %d", session_id)
            return None

    def _schedule_embed(self, trajectory_id: int, task: str) -> None:
        try:
            loop = asyncio.get_running_loop()
            t = loop.create_task(self._embed_and_write(trajectory_id, task))
            self._embed_tasks.add(t)
            t.add_done_callback(self._embed_tasks.discard)
        except RuntimeError:
            log.debug("no running event loop — skipping trajectory embed")

    async def _embed_and_write(self, trajectory_id: int, task: str) -> None:
        try:
            async with self._embedder_factory() as chain:
                [vec] = await chain.embed([task])
        except AllProvidersFailed as e:
            log.warning("trajectory embed skipped (id=%d): %s", trajectory_id, e)
            return
        except Exception:
            log.exception("trajectory embed crashed (id=%d)", trajectory_id)
            return
        blob = vec.astype(np.float32).tobytes()
        try:
            await run_vec_write(
                self._db,
                lambda w: w.execute(
                    "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
                    (trajectory_id, blob),
                ),
            )
        except Exception:
            log.exception("trajectory vec write failed (id=%d)", trajectory_id)

    async def drain(self, timeout: float | None = 5.0) -> None:
        """Wait for in-flight embedding tasks (for tests)."""
        if not self._embed_tasks:
            return
        tasks = list(self._embed_tasks)
        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.wait(tasks, timeout=timeout)

    @staticmethod
    def _extract_task(messages: list[dict[str, Any]]) -> str | None:
        """Return the first user message text, capped at 500 chars."""
        for msg in messages:
            if msg.get("role") == "user":
                text = _extract_text(msg)
                if text:
                    return text[:500]
        return None

    @staticmethod
    def _outcome(messages: list[dict[str, Any]], had_errors: bool) -> str:
        """Classify outcome as ok / partial / fail.

        - ok: no errors
        - fail: errors and a short session (agent gave up quickly)
        - partial: errors but the agent did substantial work anyway
        """
        if not had_errors:
            return "ok"
        # Count non-system, non-empty messages as a proxy for "work done"
        substantive = sum(
            1
            for m in messages
            if m.get("role") in ("user", "assistant", "tool")
            and _extract_text(m)
        )
        return "partial" if substantive > 4 else "fail"
