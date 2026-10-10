"""Turn archival — persist ``context.messages`` into chunks + chunks_vec.

Runs from ``NanoHermesHook.after_iteration``:

1. Synchronously inserts any newly-appended messages into ``chunks``.
   The FTS5 trigger fires on insert, so keyword search stays current
   with zero latency.
2. Fires a background asyncio task to embed the new chunks via the
   provider chain and write them to ``chunks_vec``. The loop does not
   wait on the network.

Conversation identity comes from nanobot's ``AgentHookContext.session_key``
(a stable ``channel:chat`` string) when the caller supplies it, falling back
to ``id(messages_list)`` when it doesn't — direct API users and older tests.
The fallback is imperfect: Python reuses ``id()`` values after GC, and two
interleaved conversations get distinct ids, so every switch between them
looks like a new session.

nanobot hands the hook a **bounded sliding window**, not a monotonically
growing list: ``context.messages`` wraps ``Session.get_history(max_messages=...,
max_tokens=...)``, which slices the unconsolidated tail and re-trims it to a
legal user-turn start every turn. So a message index is only meaningful within
one turn, and no single message's text is identity either — a window full of
``"ok"`` replies made a one-message anchor match the wrong occurrence and stop
archiving for good (48 messages emitted, 12 stored, nothing logged).

Position therefore rides a **block alignment**: the archiver keeps an ordered
list of fingerprints of what it has already stored for a conversation and each
pass finds the longest suffix of that list which is also a prefix of this
window. Everything past the overlap is new. A run of identical replies can no
longer mislead it, because the unique turns bracketing the run are part of the
matched block. ``overlap == 0`` against a non-empty list *is* the
new-conversation signal — there is no length rule left.

Fingerprints are taken from the window, i.e. from messages that have already
been through ``get_history`` and nanobot's replay sanitizer. Fingerprinting the
raw mid-turn form instead is what made every window re-archive from zero when a
reply held a line the sanitizer drops (an image breadcrumb or a tool-call echo:
12 messages emitted, 27 chunks, growing quadratically). The
``[Message Time: ...]`` prefix is deliberately *kept* in the fingerprint — it is
the only per-message identity nanobot offers, and stripping it made two
identical prompts on different days compare equal, so a daily cron job archived
its first day and nothing after.

*Residual, version-dependent:* the prefix is what makes two identical prompts
distinguishable, and nanobot HEAD no longer emits it (nothing there produces
``[Message Time: ...]``; only the sanitizer still strips it, and neither version
keeps ``timestamp`` on a history entry). Where it is absent, a ``/new`` whose
first message is byte-identical to the last one archived under the same key
still continues the old session. Narrow, and there is nothing in
``AgentHookContext`` to break the tie; revisit if upstream ever exposes a
per-message or per-session-instance id.

Two message shapes cannot be fingerprinted from the window. The first is the
reply appended *during* the turn, which ``after_iteration`` archives before any
``get_history`` has rewritten it. The second is the window's **last entry on
every pass**: ``ContextBuilder.build_messages`` appends the current message plus
the runtime-context block, so its live text is
``"u1\n\n[Runtime Context …]"`` against a replayed
``"[Message Time: t]\nu1"``. Confirming that fingerprint made every single turn
take the new-conversation branch: one ``sessions`` row and one trajectory per
turn, the whole ``session_key`` scheme inert.

That entry is *appended*, not merged, on an ordinary turn:
``AgentLoop._state_build`` reads ``ctx.history = session.get_history(...)``
before it calls ``_persist_user_message_early``, so the roles never line up
(``loop.py:1465`` / ``:1478`` on the pinned nanobot-ai 0.2.2; the same order holds
at HEAD ``b46d4dfd``, where the two sit at ``:2049`` / ``:2122``). The merge
branch at the end of ``ContextBuilder.build_messages`` fires only when the
history already ends with a ``user`` entry — right after in-place compaction,
whose summary is stored as one, or when an interrupted turn is retried. This
premise has been wrong twice; the symbols are cited so the next reader can
re-check it instead of inferring it.

Both shapes are recorded as *pending* by role only; the next pass skips them by
role and adopts their replayed fingerprints. Losing that would mean losing the
last reply of every session. The live entry's *text* is cleaned before the
insert (``_clean_live_tail``) so the retrieval corpus doesn't carry a metadata
block per turn — cleaning cannot rescue the fingerprint, only the corpus.

A merged entry holds two messages in one row, and the window that replays them
separately archives both again: two duplicate rows per compaction. Deliberate —
keeping the unresolved pending items instead measured 29 chunks for 24 messages
on a post-compaction tool turn, because a kept item goes on matching later
entries by role and shifts every window after it.

``session_key`` is conversation identity, not session-instance identity, so a
zero overlap under an unchanged key is what opens a new ``sessions`` row. It
fires for both ``/new`` and in-place history compaction, and
``AgentHookContext`` exposes nothing that tells them apart. That conflation is
load-bearing rather than a defect: a session boundary is the only thing that
writes a trajectory, so a long-running conversation that is never `/new`-ed
still gets one at each compaction.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import sqlite3
import time
from typing import Any, Callable, NamedTuple

import numpy as np

from ..embedding.chain import AllProvidersFailed, EmbeddingChain
from ..embedding.contextual import add_context_preamble
from ..lru import MAX_TRACKED_CONVERSATIONS, touch
from ..redact import redact
from .db import fts_guarded_write, run_vec_write

log = logging.getLogger(__name__)

_ARCHIVABLE_ROLES = frozenset({"user", "assistant", "tool"})

# How many fingerprints to retain per conversation, at minimum. Only an overlap
# with the current window matters; 512 covers ``max_messages`` at its default of
# 120 several times over, and the trim below also keeps twice the current
# window, because ``max_messages`` is user config with no upper bound — a list
# shorter than the window would drop its head, find no overlap, and declare a
# new conversation on every turn.
_MAX_FINGERPRINTS = 512

# How many conversations to keep alignment state for — the cap every
# per-conversation dict shares (see ``nano_hermes.lru``).
_MAX_TRACKED_CONVERSATIONS = MAX_TRACKED_CONVERSATIONS

# Opening tag of the runtime-context block ``build_messages`` merges into the
# window's last message (``context.py`` ``_RUNTIME_CONTEXT_TAG``). Matched on the
# prefix so upstream wording changes after it don't matter, and duplicated
# rather than imported from ``coordinator/goal.py`` to keep this module out of
# the coordinator import cycle.
_RUNTIME_SENTINEL = "[Runtime Context"

# Prefix ``get_history`` stamps onto a replayed message. A window entry that
# carries it came from history; the live current message does not, because
# ``build_messages`` appends ``current_message`` verbatim. That is the only
# evidence available for "this entry is the merge of a history message and the
# current one" — see :func:`_clean_live_tail`.
_MESSAGE_TIME_PREFIX = "[Message Time:"


def _clean_live_tail(text: str, *, role: str = "user", merged: bool = False) -> str:
    """Strip the runtime-context block, and a merge duplicate when there is one.

    The usual live entry is *appended*, not merged: ``_state_build`` reads the
    history before ``_persist_user_message_early`` (see the module docstring),
    so the roles do not line up and ``build_messages`` adds a
    fresh ``current_message + [Runtime Context …]`` entry. Archiving it verbatim
    puts a per-turn metadata block (time, cwd, channel) into every user chunk —
    that is, into FTS5 and into the embedder.

    The merge branch of ``build_messages`` only fires when the history
    already ends with a ``user`` entry: right after in-place compaction, whose
    summary is stored as one, or after an interrupted turn. Then the text is
    ``"[Message Time: t]\n<history>" + "\n\n" + "<current>"`` and the
    paragraph dedup below applies. It is gated on *merged*, which the caller
    derives from the ``[Message Time:`` stamp, because an unstamped live entry
    is an unmerged one and a real message may legitimately end with a repeated
    paragraph (``"A\n\nB\n\nA\n\nB"`` must survive whole).

    The fingerprint is deliberately *not* taken from the cleaned text: no
    amount of cleaning makes the live form equal the replayed form, which is
    why a live entry is pending-by-role rather than confirmed.

    The runtime-context cut applies to ``user`` entries only, and only to a
    block that starts a line — the one shape nanobot writes. A tool
    result or reply that merely *mentions* the tag (reading ``context.py``, say)
    is content, and cutting it lost everything after the tag from chunks, FTS5
    and the embedder. Not gated on liveness: nanobot HEAD persists the block
    into the stored user message and ``get_history`` replays it by default
    (``include_runtime_context=True``), so a replayed user entry archived after
    a restart carries it too.
    """
    if role == "user":
        # One newline, not two: a multimodal message's text parts are joined
        # with "\n" (``_extract_text``), so its block follows a single one.
        cut = text.find("\n" + _RUNTIME_SENTINEL)
        if cut > 0:
            text = text[:cut].rstrip()
    if not merged:
        return text
    # The merge repeats the history entry's own text when the interrupted turn
    # is being retried: ``"[Message Time: t]\nu1" + "\n\n" + "u1"``. Leftmost
    # split wins, so a prompt that itself contains a blank line is de-duplicated
    # whole rather than one paragraph at a time. A post-compaction merge
    # (``"<summary>\n\n<current>"``) matches nothing here and is kept whole.
    # The history half must *equal* the current message once its stamp line is
    # set aside — a suffix test dropped a genuine reply ("yes") whenever the
    # interrupted message happened to end with it ("should I say yes").
    at = text.find("\n\n")
    while at != -1:
        head, tail = text[:at], text[at + 2:]
        if tail and head.partition("\n")[2] == tail:
            return head
        at = text.find("\n\n", at + 1)
    return text


def _fingerprint(role: str, text: str) -> str:
    """Identity of one message, as it appears in a replayed window.

    Hashed rather than stored verbatim so a saturated window's worth of
    fingerprints costs kilobytes, not megabytes, per tracked conversation.
    """
    h = hashlib.blake2b(digest_size=16)
    h.update(role.encode())
    h.update(b"\x00")
    h.update(text.encode())
    return h.hexdigest()


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


def _archivable(
    messages: list[dict[str, Any]],
    cache: dict[int, tuple] | None = None,
) -> list[tuple[dict[str, Any], str, str]]:
    """The subsequence of *messages* this archiver stores, with fingerprints.

    Filtered identically to the insert loop, so a fingerprint list and a window
    are always comparable: a system prompt or a tool-calls-only assistant
    message shifts neither.

    Runs over the whole window on every pass — the overlap search needs the
    window's head — and there is one pass per hook call, so a turn with eight
    tool calls hashes the window nine times. Within a turn the runner appends
    to one list of the same message dicts, so *cache* (``id(msg)`` → the
    message, its content object, and the result) answers the repeats. Each
    entry holds the message itself, so an id cannot be recycled while cached,
    and a replaced ``content`` misses. The caller replaces the cache with each
    call's results, so it never outlives one window.
    """
    out: list[tuple[dict[str, Any], str, str]] = []
    fresh: dict[int, tuple] = {}
    for msg in messages:
        role = msg.get("role")
        if role not in _ARCHIVABLE_ROLES:
            continue
        content = msg.get("content")
        hit = cache.get(id(msg)) if cache else None
        if hit is not None and hit[0] is msg and hit[1] is content and hit[2] == role:
            text, fp = hit[3], hit[4]
        else:
            text = _extract_text(msg)
            fp = None if text is None else _fingerprint(str(role), text)
        if cache is not None:
            fresh[id(msg)] = (msg, content, role, text, fp)
        if text is None:
            continue
        out.append((msg, text, fp))
    if cache is not None:
        cache.clear()
        cache.update(fresh)
    return out


def _overlap(confirmed: list[str], window: list[str]) -> int:
    """How many of *window*'s leading messages are already archived.

    The window's first message is looked up in the archived sequence and the
    match extended forward. A match counts only if it runs to the end of one
    side or the other: to the end of *confirmed* (the normal case — the window
    continues where archiving stopped) or to the end of *window* (the whole
    window is already archived, which is what consolidation dropping messages
    out of later windows looks like). Anything shorter is a coincidental
    interior hit.

    Both acceptance conditions are end-anchored, so an accepted match at a
    lower starting position is never the shorter one and taking the longest is
    the same as taking the first — ``best`` only makes that explicit. What the
    anchoring buys is that a run of identical replies resolves to the alignment
    which also matches the unique turns around it, instead of to the first
    coincidence. Only the window's first fingerprint is ever looked up, so the
    candidates come from one filtered scan; building a full fingerprint index
    per call was a fifth of the hook's time on a saturated window.
    """
    if not confirmed or not window:
        return 0
    first = window[0]
    best = 0
    for i in [i for i, fp in enumerate(confirmed) if fp == first]:
        k = 0
        while i + k < len(confirmed) and k < len(window) and confirmed[i + k] == window[k]:
            k += 1
        if k > best and (i + k == len(confirmed) or k == len(window)):
            best = k
    return best


class _Alignment(NamedTuple):
    start: int            # first index of *entries* not yet archived
    adopted: list[int]    # indices whose replayed fingerprint to confirm
    keep: list[tuple[str, str]]   # pending items still unreplayed
    new_conversation: bool


def _align(
    confirmed: list[str],
    pending: list[tuple[str, str]] | None,
    entries: list[tuple[dict[str, Any], str, str]],
    *,
    mid_turn: bool = False,
) -> _Alignment:
    """Locate *entries* against what this conversation has already archived.

    ``confirmed`` are fingerprints of messages stored in their replayed form —
    the ones a window can be matched against directly. ``pending`` are
    ``(role, fingerprint)`` pairs for messages stored in a form the window will
    not repeat: this turn's raw reply, and the live merged tip (see the module
    docstring). Each pending item resolves one of three ways:

    * it matches by fingerprint or by role, and this window is one a later
      window will repeat (a replayed window, and not its final entry) — adopt
      the fingerprint; the message is confirmed from here on.
    * it matches, but this window won't be repeated: the same list handed to
      another pass (there are several per turn: ``before_iteration`` per
      iteration, ``after_iteration``, plus
      ``current_session_id``/``ensure_session``), or the live final entry.
      Already archived, so skip it — and leave it pending.
    * it matches neither, so the run stops. That keeps a message the sanitizer dropped
      entirely from shifting the rest of the run and swallowing a genuinely new
      message. Known cost: if nanobot drops a pending message during
      persistence instead of rewriting it (an orphan tool result,
      ``AgentLoop._save_turn``), the rest no longer lines up and those messages are
      archived a second time. One duplicate row on a path that logs a warning
      upstream, versus swallowing a real message.

    Pure, so :meth:`_archive` and :meth:`_is_new_conversation` cannot disagree
    about the same window.
    """
    overlap = _overlap(confirmed, [e[2] for e in entries])
    start = overlap
    last = len(entries) - 1
    adopted: list[int] = []
    keep: list[tuple[str, str]] = []
    for role, raw_fp in pending or ():
        if start > last:
            break
        if entries[start][2] != raw_fp and (
            start == last or entries[start][0].get("role") != role
        ):
            # The final entry of a window is always freshly appended — the live
            # current message on an iteration-0 pass, this turn's raw reply on a
            # mid-turn one — never the replay of a pending message. Consuming it
            # by role is how a new conversation that repeats an old prompt gets
            # swallowed. The remainder of the run is dropped rather than kept:
            # a kept item the window never replays goes on matching later
            # entries by role and shifts every window after it (measured: 29
            # chunks for 24 messages on a post-compaction tool turn, versus 2
            # duplicate rows from dropping it).
            break
        if mid_turn or start == last:
            # This pass is looking at the same unreplayed form again: already
            # archived, but its fingerprint still isn't one a later window will
            # repeat, so it stays pending.
            keep.append((role, raw_fp))
        else:
            adopted.append(start)
        start += 1
    if start == 0 and (confirmed or (pending and not mid_turn)):
        # This window shares nothing with what the key has archived — neither a
        # fingerprint overlap nor the pending run (which is tried from index 0
        # too, because the window can slide until it *starts* at a message that
        # is still pending). So it is a different conversation under the same
        # key: /new in the same channel, a recycled id(), or a daily cron job
        # repeating yesterday's prompt with a new ``[Message Time:]``. In-place
        # compaction looks the same from here and also starts a new session;
        # see the module docstring. A mid-turn pass is never judged on pending
        # alone: it legitimately holds the raw list.
        return _Alignment(0, [], [], True)
    return _Alignment(start, adopted, keep, False)


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
        # Conversation key (session_key, else id(messages_list)) → sessions.id,
        # the ordered fingerprints of what has already been archived, and the
        # roles of messages archived mid-turn whose fingerprints are not yet
        # trustworthy. See the module docstring — the fingerprint list is what
        # positions every archive pass.
        self._session_ids: dict[str | int, int] = {}
        self._fingerprints: dict[str | int, list[str]] = {}
        # (role, fingerprint) per message archived from a form the window
        # will not repeat — see _align.
        self._pending: dict[str | int, list[tuple[str, str]]] = {}
        # sessions.id → next turn_index. Per session and monotonic, because
        # ``chunks.turn_index`` is read back as transcript order
        # (``session/browse.py``) and as a turn count (``status/tool.py``); the
        # window index it used to hold ran 1..cap then repeated.
        self._turn_counts: dict[int, int] = {}
        # The window the last successful pass archived: (list, length, key).
        # The hook asks current_session_id about that same list straight after
        # archiving it, and re-deriving the verdict cost a full-window hash plus
        # an alignment — about a quarter of the hook's per-turn time. Holding
        # the list itself, not its id(), so a recycled address cannot hit.
        self._last_pass: tuple[list, int, str | int] | None = None
        # Per-message fingerprints of the last window seen, for _archivable.
        self._entry_cache: dict[int, tuple] = {}
        # session_id → first user message text (task context for embeddings).
        self._session_tasks: dict[int, str] = {}
        # Pending background embedding tasks — retained so asyncio
        # doesn't GC them mid-flight.
        self._embed_tasks: set[asyncio.Task] = set()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def archive_and_embed(
        self,
        messages: list[dict[str, Any]],
        session_key: str | None = None,
        *,
        mid_turn: bool = False,
    ) -> tuple[list[int], asyncio.Task | None]:
        """Insert new chunks synchronously and schedule embedding.

        Must be called from within a running asyncio event loop (the
        embedding step is scheduled as a background task on that loop).
        Returns ``(new_chunk_ids, embed_task_or_none)``. Callers don't
        need to await the task; tests can via :meth:`drain`.

        ``mid_turn=True`` says the tail of *messages* was appended during this
        turn and so has not been through ``get_history`` — its fingerprints are
        not usable for the next pass. ``after_iteration`` is the only such
        caller; see the module docstring.
        """
        chunk_ids, texts, session_id = self._archive(
            messages, session_key, mid_turn=mid_turn
        )
        if not chunk_ids:
            return [], None
        task_ctx = self._session_tasks.get(session_id) if session_id is not None else None
        task = self._schedule_embed(chunk_ids, texts, task_ctx)
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

    def current_session_id(
        self, messages: list[dict[str, Any]], session_key: str | None = None
    ) -> int | None:
        """Return the ``sessions.id`` this messages list is mapped to, or
        ``None`` if ``archive_and_embed`` hasn't run against it yet.

        Exposed for :class:`NanoHermesHook` so the reflect tool can
        attach reflections to the correct session without duplicating
        the key-based bookkeeping.

        Returns ``None`` when this window is a new conversation under a reused
        key, because the mapped id belongs to the *previous* one. Reporting it
        would attach this turn's reflections to a session that is about to be
        ended and purged.
        """
        key = session_key or id(messages)
        last = self._last_pass
        if last is not None and last[0] is messages and last[1:] == (len(messages), key):
            # Just archived: aligned by construction, so not a new conversation.
            return self._session_ids.get(key)
        if self._is_new_conversation(key, messages):
            return None
        return self._session_ids.get(key)

    def ensure_session(
        self, messages: list[dict[str, Any]], session_key: str | None = None
    ) -> int | None:
        """Bootstrap a session row and archive any new chunks without embedding.

        Safe to call from synchronous code — no asyncio event loop required.
        Subsequent :meth:`archive_and_embed` calls align on the fingerprints
        recorded here, so chunks already written are not inserted twice.
        """
        key = session_key or id(messages)
        if not messages:
            return self._session_ids.get(key)
        self._archive(messages, session_key)
        return self._session_ids.get(key)

    def prune_session_by_id(self, session_id: int) -> None:
        """Remove all internal bookkeeping entries that map to session_id.

        Called when a session boundary is crossed to bound dict growth.
        Keyed by conversation key, not by session_id directly.
        """
        keys_to_remove = [k for k, v in self._session_ids.items() if v == session_id]
        for k in keys_to_remove:
            self._session_ids.pop(k, None)
            self._fingerprints.pop(k, None)
            self._pending.pop(k, None)
        self._session_tasks.pop(session_id, None)
        self._turn_counts.pop(session_id, None)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _archive(
        self,
        messages: list[dict[str, Any]],
        session_key: str | None = None,
        *,
        mid_turn: bool = False,
    ) -> tuple[list[int], list[str], int | None]:
        key = session_key or id(messages)
        entries = _archivable(messages, self._entry_cache)
        if not entries:
            # A window with nothing archivable in it (system prompt only) says
            # nothing about conversation identity — in particular it must not
            # look like a zero overlap and start a session.
            return [], [], self._session_ids.get(key)

        aligned = _align(
            self._fingerprints.get(key, []),
            self._pending.get(key),
            entries,
            mid_turn=mid_turn,
        )
        start = aligned.start
        if aligned.new_conversation:
            # Nothing this window shares with what we stored: a different
            # conversation under the same key — /new in the same channel, or a
            # recycled id(). (In-place compaction looks the same from here and
            # also starts a new session; see the module docstring.)
            old_sid = self._session_ids.pop(key, None)
            if old_sid is not None:
                self._session_tasks.pop(old_sid, None)
                self._turn_counts.pop(old_sid, None)
                self._end_session(old_sid)
            self._fingerprints.pop(key, None)
            self._pending.pop(key, None)
            start = 0
        # Fingerprints this pass confirms, applied only once every insert has
        # landed: committing them earlier left a failed pass's pending items
        # adoptable a second time, so the next window aligned against a doubled
        # block and re-archived as duplicates.
        new_confirmed: list[str] = []
        if not aligned.new_conversation and not mid_turn:
            # This window replayed those pending messages, so their real
            # fingerprints are finally knowable — adopt them. A mid-turn pass
            # must not: that list is the raw one the runner is still appending
            # to, and its tail will be rewritten before it comes back.
            new_confirmed.extend(entries[i][2] for i in aligned.adopted)

        session_id = self._session_ids.get(key)
        if session_id is None:
            session_id = self._start_session(key, session_key)
        turn = self._turn_counts.get(session_id, 0)

        new_ids: list[int] = []
        new_texts: list[str] = []
        now = time.time()
        cur = self._db.cursor()
        archived_roles: list[tuple[str, str]] = []
        last_index = len(entries) - 1
        try:
            for index, (msg, text, fp) in enumerate(entries[start:], start=start):
                # The window's last entry on a non-mid-turn pass is the live
                # current message: stored in a form no later window repeats, so its
                # fingerprint must not be confirmed. It is *merged* with the entry
                # before it only when the history already ended with a user turn,
                # and the ``[Message Time:`` stamp is the evidence — an unstamped
                # live entry came from ``current_message`` verbatim.
                live = not mid_turn and index == last_index
                merged = live and text.startswith(_MESSAGE_TIME_PREFIX)
                text = _clean_live_tail(text, role=str(msg.get("role")), merged=merged)
                # Redact secret-shaped strings so they neither land in chunks
                # nor reach the embedding provider over the wire. The same
                # masked text feeds both the INSERT and the background embed.
                if self._redact_secrets:
                    text = redact(text).text
                # Capture first user message as task context for embeddings.
                if msg.get("role") == "user" and session_id not in self._session_tasks:
                    self._session_tasks[session_id] = text[:500]
                turn += 1
                # Guarded per row, not per batch: a batch-level retry would
                # re-insert the rows that already landed before the failure. If
                # chunks_fts is corrupt the trigger would otherwise fail this
                # canonical INSERT and the hook's archive-level except would
                # swallow it, silently losing the transcript.
                fts_guarded_write(
                    self._db,
                    "chunks_fts",
                    lambda: cur.execute(
                        "INSERT INTO chunks "
                        "(session_id, turn_index, role, content, created_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (session_id, turn, msg["role"], text, now),
                    ),
                )
                new_ids.append(int(cur.lastrowid))
                new_texts.append(text)
                archived_roles.append((str(msg["role"]), fp))
                if not mid_turn and not live:
                    new_confirmed.append(fp)
        except BaseException:
            # All or nothing for chunks and alignment state: rows from a
            # half-done pass would sit in chunks while the alignment state says
            # they were never archived, and the next pass would insert them a
            # second time. The sessions row is the deliberate exception —
            # _start_session committed it already, so the mapping stays and the
            # retry archives into it rather than leaving chunks pointing at a
            # rolled-back row. _session_tasks may keep the failed pass's first
            # user text; it is only embedding context for that same session.
            self._db.rollback()
            if new_ids:
                # Rollback is not enough on its own: detaching a corrupt
                # chunks_fts mid-pass commits (``db.detach_fts``), which makes
                # the rows inserted before it durable. Remove them explicitly.
                try:
                    marks = ",".join("?" * len(new_ids))
                    self._db.execute(f"DELETE FROM chunks WHERE id IN ({marks})", new_ids)
                    self._db.commit()
                except sqlite3.Error:
                    log.exception("could not remove a failed archive pass's rows")
            raise
        confirmed = self._fingerprints.setdefault(key, [])
        confirmed.extend(new_confirmed)
        self._turn_counts[session_id] = turn
        # What stays pending: the items this window did not replay, plus
        # everything just archived in a form the next window will not repeat —
        # all of it on a mid-turn pass, and the live current message otherwise.
        pending = list(aligned.keep)
        if mid_turn:
            pending += archived_roles
        elif start <= last_index:
            pending.append(
                (str(entries[last_index][0]["role"]), entries[last_index][2])
            )
        if pending:
            self._pending[key] = pending
        else:
            self._pending.pop(key, None)
        del confirmed[: -max(_MAX_FINGERPRINTS, 2 * len(entries))]
        self._evict_idle_conversations(key)
        if new_ids:
            self._db.commit()
        self._last_pass = (messages, len(messages), key)
        return new_ids, new_texts, session_id

    def release_window(self, key: str | int | None) -> None:
        """Drop the references to *key*'s last window once its turn is over.

        ``_last_pass`` and ``_entry_cache`` hold that window's list and every
        message dict in it — large tool outputs, base64 image parts — and would
        otherwise keep them alive until some later pass replaced them. Another
        conversation's window is left alone: its turn may still be running.
        """
        last = self._last_pass
        if last is None:
            return
        # A keyless window is keyed by its list's id(); its turn has no session
        # key to match on, so any keyless turn's end releases it.
        if last[2] == key or (key is None and isinstance(last[2], int)):
            self._last_pass = None
            self._entry_cache.clear()

    def _evict_idle_conversations(self, key: str | int) -> None:
        """Bound per-conversation state to the busiest _MAX_TRACKED_CONVERSATIONS.

        Nothing else releases it: ``prune_session_by_id`` runs at a boundary, by
        which point :meth:`_archive` has already popped and remapped the key, so
        it matches nothing — and a channel that is simply never spoken to again
        crosses no boundary at all. Without this a long-lived process serving
        many channels keeps a fingerprint list per channel forever.

        Everything goes together, session id included: dropping the
        fingerprints alone would leave the next window aligning against an empty
        list inside a live session and re-archiving it as duplicates. Dropping
        the id too makes that window open a fresh session instead — one bogus
        boundary for a conversation that has been idle for 32 others. The
        session-id-keyed state goes with it, since that is the last reference to
        the id and nothing else would ever release it.
        """
        for idle in touch(self._fingerprints, key):
            self._pending.pop(idle, None)
            idle_sid = self._session_ids.pop(idle, None)
            if idle_sid is not None:
                self._session_tasks.pop(idle_sid, None)
                self._turn_counts.pop(idle_sid, None)
                self._end_session(idle_sid)

    def _end_session(self, session_id: int) -> None:
        """Stamp ``ended_at`` on a session this archiver will never write again.

        Dropping the mapping is what ends a session: the conversation's next
        pass opens a new row. The coordinator's boundary also stamps it, but
        only when it still remembers the conversation — past its own 32-key
        cap it does not, and a session left with ``ended_at`` NULL is one
        ``purge_older_than`` can never delete.
        """
        try:
            self._db.execute(
                "UPDATE sessions SET ended_at = ? WHERE id = ? AND ended_at IS NULL",
                (time.time(), session_id),
            )
            self._db.commit()
        except sqlite3.Error:
            log.exception("failed to set ended_at for session %d", session_id)

    def _is_new_conversation(
        self, key: str | int, messages: list[dict[str, Any]]
    ) -> bool:
        """True when *messages* shares nothing with what this key has archived.

        Read-only mirror of the branch in :meth:`_archive` — the same
        :func:`_align` call, so the two cannot drift apart.
        """
        entries = _archivable(messages, self._entry_cache)
        if not entries:
            return False
        return _align(
            self._fingerprints.get(key, []),
            self._pending.get(key),
            entries,
        ).new_conversation

    def _start_session(self, key: str | int, session_key: str | None = None) -> int:
        cur = self._db.cursor()
        if session_key:
            # A row for this conversation still open here was left by a previous
            # process (in this one, dropping the mapping already ended it):
            # nothing will ever write to it again, and ``purge_older_than`` only
            # deletes ended sessions.
            cur.execute(
                "UPDATE sessions SET ended_at = ? WHERE session_key = ? AND ended_at IS NULL",
                (time.time(), session_key),
            )
        cur.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
            (session_key or f"loop:{int(time.time() * 1000)}", time.time()),
        )
        self._db.commit()
        session_id = int(cur.lastrowid)
        self._session_ids[key] = session_id
        return session_id

    def _schedule_embed(
        self, chunk_ids: list[int], texts: list[str], task_ctx: str | None = None
    ) -> asyncio.Task:
        # Raises RuntimeError if no running loop — that's the contract.
        loop = asyncio.get_running_loop()
        task = loop.create_task(self._embed_and_write(chunk_ids, texts, task_ctx))
        self._embed_tasks.add(task)
        task.add_done_callback(self._embed_tasks.discard)
        return task

    async def _embed_and_write(
        self, chunk_ids: list[int], texts: list[str], task_ctx: str | None = None
    ) -> None:
        embed_texts = [add_context_preamble(t, task_ctx) for t in texts]
        try:
            async with self._embedder_factory() as chain:
                vecs = await chain.embed(embed_texts)
        except AllProvidersFailed as e:
            log.warning("embed batch (%d chunks) skipped: %s", len(chunk_ids), e)
            return
        except Exception:
            log.exception("embed batch (%d chunks) crashed", len(chunk_ids))
            return

        rows = [
            (cid, vec.astype(np.float32).tobytes()) for cid, vec in zip(chunk_ids, vecs)
        ]
        try:
            await run_vec_write(
                self._db,
                lambda w: w.executemany(
                    "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)", rows
                ),
            )
        except Exception:
            log.exception("vec write failed for %d chunks", len(chunk_ids))
