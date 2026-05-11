"""Turn archival — persist ``context.messages`` into chunks + chunks_vec.

Runs from ``NanoHermesHook.after_iteration``:

1. Synchronously inserts any newly-appended messages into ``chunks``.
   The FTS5 trigger fires on insert, so keyword search stays current
   with zero latency.
2. Fires a background asyncio task to embed the new chunks via the
   provider chain and write them to ``chunks_vec``. The loop does not
   wait on the network.

Session boundaries are keyed on ``id(messages_list)``. A new list (new
run) means a new ``sessions`` row. This is pragmatic but imperfect:
Python can reuse ``id()`` values after GC, which would glue unrelated
runs into one synthetic session. v0.2 should graduate to real session
tracking once nanobot exposes the ``Session`` object to hooks (it's
currently passed to the runner via ``AgentRunSpec.session_key`` but not
into ``AgentHookContext``).
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from typing import Any, Callable

import numpy as np

from ..embedding.chain import AllProvidersFailed, EmbeddingChain
from ..redact import redact

log = logging.getLogger(__name__)

_ARCHIVABLE_ROLES = frozenset({"user", "assistant", "tool"})


def _extract_text(msg: dict[str, Any]) -> str | None:
    """Return a searchable string for one OpenAI-format message, or None
    if the message has no textual content (e.g. an assistant message
    that only emitted ``tool_calls``)."""
    content = msg.get("content")
    if content is None:
        return None
    if isinstance(content, str):
        text = content.strip()
        return text or None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            block_text = block.get("text")
            if block_text:
                parts.append(str(block_text))
        joined = "\n".join(parts).strip()
        return joined or None
    text = str(content).strip()
    return text or None


class SessionArchiver:
    def __init__(
        self,
        *,
        db: sqlite3.Connection,
        embedder_factory: Callable[[], EmbeddingChain],
        target_dims: int,
        redact_secrets: bool = True,
    ) -> None:
        self._db = db
        self._embedder_factory = embedder_factory
        self._target_dims = target_dims
        self._redact_secrets = redact_secrets
        # id(messages_list) → sessions.id and high-water message index.
        self._session_ids: dict[int, int] = {}
        self._watermarks: dict[int, int] = {}
        # Pending background embedding tasks — retained so asyncio
        # doesn't GC them mid-flight.
        self._embed_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def archive_and_embed(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[int], asyncio.Task | None]:
        """Insert new chunks synchronously and schedule embedding.

        Must be called from within a running asyncio event loop (the
        embedding step is scheduled as a background task on that loop).
        Returns ``(new_chunk_ids, embed_task_or_none)``. Callers don't
        need to await the task; tests can via :meth:`drain`.
        """
        chunk_ids, texts = self._archive(messages)
        if not chunk_ids:
            return [], None
        task = self._schedule_embed(chunk_ids, texts)
        return chunk_ids, task

    async def drain(self, timeout: float | None = 5.0) -> None:
        """Wait for in-flight embedding tasks.

        Used by tests and graceful shutdown. ``timeout=None`` waits
        indefinitely; otherwise, unfinished tasks are abandoned.
        """
        if not self._embed_tasks:
            return
        tasks = list(self._embed_tasks)
        if timeout is None:
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.wait(tasks, timeout=timeout)

    def current_session_id(self, messages: list[dict[str, Any]]) -> int | None:
        """Return the ``sessions.id`` this messages list is mapped to, or
        ``None`` if ``archive_and_embed`` hasn't run against it yet.

        Exposed for :class:`NanoHermesHook` so the reflect tool can
        attach reflections to the correct session without duplicating
        the id-based bookkeeping.
        """
        return self._session_ids.get(id(messages))

    def ensure_session(self, messages: list[dict[str, Any]]) -> int | None:
        """Bootstrap a session row and archive any new chunks without embedding.

        Safe to call from synchronous code — no asyncio event loop required.
        Subsequent :meth:`archive_and_embed` calls use the watermark to avoid
        double-inserting chunks already written here.
        """
        if not messages:
            return self._session_ids.get(id(messages))
        self._archive(messages)
        return self._session_ids.get(id(messages))

    def prune_session_by_id(self, session_id: int) -> None:
        """Remove all internal bookkeeping entries that map to session_id.

        Called when a session boundary is crossed to bound dict growth.
        Keyed by id(messages_list), not by session_id directly.
        """
        keys_to_remove = [k for k, v in self._session_ids.items() if v == session_id]
        for k in keys_to_remove:
            self._session_ids.pop(k, None)
            self._watermarks.pop(k, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _archive(
        self, messages: list[dict[str, Any]]
    ) -> tuple[list[int], list[str]]:
        key = id(messages)
        watermark = self._watermarks.get(key, 0)
        # If the list got shorter since last time, treat it as a fresh
        # list that happens to share the same id() — start a new session.
        if watermark > len(messages):
            self._session_ids.pop(key, None)
            watermark = 0

        session_id = self._session_ids.get(key)
        if session_id is None:
            session_id = self._start_session(key)

        new_ids: list[int] = []
        new_texts: list[str] = []
        now = time.time()
        cur = self._db.cursor()
        for idx in range(watermark, len(messages)):
            msg = messages[idx]
            if msg.get("role") not in _ARCHIVABLE_ROLES:
                continue
            text = _extract_text(msg)
            if text is None:
                continue
            # Redact secret-shaped strings so they neither land in chunks
            # nor reach the embedding provider over the wire. The same
            # masked text feeds both the INSERT and the background embed.
            if self._redact_secrets:
                text = redact(text).text
            cur.execute(
                "INSERT INTO chunks "
                "(session_id, turn_index, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, idx, msg["role"], text, now),
            )
            new_ids.append(int(cur.lastrowid))
            new_texts.append(text)
        self._watermarks[key] = len(messages)
        if new_ids:
            self._db.commit()
        return new_ids, new_texts

    def _start_session(self, key: int) -> int:
        cur = self._db.cursor()
        cur.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            (f"loop:{int(time.time() * 1000)}", time.time()),
        )
        self._db.commit()
        session_id = int(cur.lastrowid)
        self._session_ids[key] = session_id
        return session_id

    def _schedule_embed(
        self, chunk_ids: list[int], texts: list[str]
    ) -> asyncio.Task:
        # Raises RuntimeError if no running loop — that's the contract.
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._embed_and_write(chunk_ids, texts))
        self._embed_tasks.add(task)
        task.add_done_callback(self._embed_tasks.discard)
        return task

    async def _embed_and_write(
        self, chunk_ids: list[int], texts: list[str]
    ) -> None:
        try:
            async with self._embedder_factory() as chain:
                vecs = await chain.embed(texts)
        except AllProvidersFailed as e:
            log.warning("embed batch (%d chunks) skipped: %s", len(chunk_ids), e)
            return
        except Exception:
            log.exception("embed batch (%d chunks) crashed", len(chunk_ids))
            return

        try:
            for cid, vec in zip(chunk_ids, vecs):
                self._db.execute(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
                    (cid, vec.astype(np.float32).tobytes()),
                )
            self._db.commit()
        except Exception:
            log.exception("vec write failed for %d chunks", len(chunk_ids))
