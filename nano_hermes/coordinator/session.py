"""SessionCoordinator — extracted from NanoHermesHook.

Owns session ID tracking, boundary detection, ended_at stamping, and
trajectory finalization at session boundaries.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..session.archiver import SessionArchiver
    from ..session.trajectory import TrajectoryWriter

log = logging.getLogger(__name__)


class SessionCoordinator:
    def __init__(
        self,
        *,
        archiver: "SessionArchiver",
        db: sqlite3.Connection,
        trajectory_writer: "TrajectoryWriter",
    ) -> None:
        self._archiver = archiver
        self._db = db
        self._trajectory_writer = trajectory_writer
        self.current_session_id: int | None = None

    def sync(self, messages: list) -> tuple[int | None, int | None]:
        """Sync current_session_id from archiver state.

        Lazy-bootstraps a session row on first call so the reflect tool
        has something to attach to from iteration 0.

        Returns (current_session_id, completed_session_id_or_None).
        completed_session_id is non-None only when a session boundary was
        crossed (old session ended, new session started).
        """
        if not messages:
            return self.current_session_id, None

        existing = self._archiver.current_session_id(messages)
        if existing is None:
            try:
                existing = self._archiver.ensure_session(messages)
            except Exception:
                log.exception("nano-hermes session bootstrap failed")

        prev_session = self.current_session_id
        completed_session_id: int | None = None

        if existing is not None and existing != prev_session and prev_session is not None:
            # Mark the old session as ended so purge_older_than can find it.
            try:
                self._db.execute(
                    "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                    (time.time(), prev_session),
                )
                self._db.commit()
            except Exception:
                log.exception(
                    "failed to set sessions.ended_at for session %d", prev_session
                )
            completed_session_id = prev_session

        self.current_session_id = existing
        return existing, completed_session_id

    def finalize(
        self,
        session_id: int,
        skills_used: set[str],
        had_errors: bool,
    ) -> None:
        """Write a trajectory row for a completed session.

        Ensures ended_at is set (fallback in case sync() missed it).
        """
        try:
            self._db.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), session_id),
            )
            self._db.commit()

            reflections = [
                r[1]
                for r in self._db.execute(
                    "SELECT id, content FROM reflections "
                    "WHERE session_id = ? ORDER BY id",
                    (session_id,),
                ).fetchall()
            ]
            chunks = self._db.execute(
                "SELECT role, content FROM chunks WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
            messages = [{"role": r, "content": c} for r, c in chunks]
            self._trajectory_writer.write(
                session_id=session_id,
                messages=messages,
                skills_used=list(skills_used),
                reflections=reflections,
                had_errors=had_errors,
            )
        except Exception:
            log.exception("trajectory finalization failed for session %d", session_id)
