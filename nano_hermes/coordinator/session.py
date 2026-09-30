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

# Cap on per-conversation boundary state, so a long-lived process serving many
# channels does not accumulate one entry per channel forever.
_MAX_TRACKED_CONVERSATIONS = 32


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
        # Most recently synced session — what the reflect/rate/status tools
        # attach to. Deliberately a scalar: those tools act on the turn in
        # flight.
        self.current_session_id: int | None = None
        # Boundary detection is per session_key, not on the scalar above.
        # Interleaved chats alternate session ids (1, 2, 1, 2, ...), so
        # comparing against a single previous value fires a bogus boundary on
        # every switch — finalizing 1-turn "sessions", re-scoring principle
        # trust, and advancing the evolution/curation cadence each turn.
        # Under a stable key the boundary fires only when the *same*
        # conversation gets a new session id: /new, or in-place compaction.
        self._sessions_by_key: dict[str, int] = {}
        # Keyless runs get their own "previous", never ``current_session_id``:
        # that scalar was last written by whichever *keyed* conversation synced
        # most recently, so comparing against it ended and pruned a live keyed
        # session, re-archived its window, and wrote its trajectory twice.
        self._keyless_session: int | None = None

    def sync(
        self, messages: list, session_key: str | None = None
    ) -> tuple[int | None, int | None]:
        """Sync current_session_id from archiver state.

        Lazy-bootstraps a session row on first call so the reflect tool
        has something to attach to from iteration 0.

        Returns (current_session_id, completed_session_id_or_None).
        completed_session_id is non-None only when a session boundary was
        crossed (old session ended, new session started).
        """
        if not messages:
            return self.current_session_id, None

        existing = self._archiver.current_session_id(messages, session_key)
        if existing is None:
            try:
                existing = self._archiver.ensure_session(messages, session_key)
            except Exception:
                log.exception("nano-hermes session bootstrap failed")

        if not session_key:
            # Falsy, not just None — the archiver keys on ``session_key or
            # id(messages)``, so an empty key must resolve the same way here.
            # No stable key, so the list object is the only conversation
            # identity available and a different list means the previous
            # conversation is over. Reachable in production: nanobot's
            # ``AgentRunSpec.session_key`` is optional and loop.py passes
            # ``session.key if session else None`` into the hook context.
            # Keyless interleaving still over-counts — nothing distinguishes it
            # from two consecutive chats. See
            # ``test_keyless_interleaving_still_over_counts``.
            prev_session = self._keyless_session
        else:
            prev_session = self._sessions_by_key.get(session_key)
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

        if existing is not None and not session_key:
            self._keyless_session = existing
        if existing is not None and session_key:
            # Pop before insert so a re-synced key moves to the end: dict order
            # is insertion order, and without this the eviction below would
            # drop the oldest-*inserted* key even if it is the busiest one,
            # swallowing a genuine boundary and leaving ended_at NULL forever.
            self._sessions_by_key.pop(session_key, None)
            self._sessions_by_key[session_key] = existing
            while len(self._sessions_by_key) > _MAX_TRACKED_CONVERSATIONS:
                # Least recently synced conversation. Losing an entry only
                # costs one missed boundary.
                self._sessions_by_key.pop(next(iter(self._sessions_by_key)))
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
