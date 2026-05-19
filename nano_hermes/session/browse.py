"""``session_browse`` — session-arc navigation with bookends.

Two modes designed for reading session history with full context, unlike
``session_search`` which returns isolated snippets.

DISCOVERY:  FTS5 keyword search that deduplicates by session lineage and
            returns each matching session as a structured arc:
              • bookend_start  — first N chunks (conversation opener)
              • anchor_window  — ±window chunks around the best match
              • bookend_end    — last N chunks (conversation closer)

EXPAND:     window slice around a specific chunk_id (no FTS5) — used to
            drill into a session whose chunk_id was surfaced by DISCOVERY
            or session_search.

No LLM calls — pure SQL.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_BOOKEND_N = 3   # chunks shown at session start/end
_WINDOW_DEFAULT = 5  # chunks on each side of anchor


@dataclass
class _Chunk:
    chunk_id: int
    turn_index: int
    role: str
    content: str


@dataclass
class _SessionArc:
    session_id: int
    anchor_chunk_id: int
    bookend_start: list[_Chunk]
    anchor_window: list[_Chunk]
    bookend_end: list[_Chunk]


def _fetch_session_chunks(
    conn: sqlite3.Connection, session_id: int
) -> list[_Chunk]:
    rows = conn.execute(
        "SELECT id, turn_index, role, content FROM chunks "
        "WHERE session_id = ? ORDER BY turn_index, id",
        (session_id,),
    ).fetchall()
    return [_Chunk(chunk_id=r[0], turn_index=r[1], role=r[2], content=r[3]) for r in rows]


def _build_arc(
    chunks: list[_Chunk], anchor_id: int, window: int
) -> _SessionArc:
    """Assemble a session arc given ordered chunks and the anchor chunk_id."""
    session_id = 0  # caller fills this in

    if not chunks:
        return _SessionArc(
            session_id=session_id,
            anchor_chunk_id=anchor_id,
            bookend_start=[],
            anchor_window=[],
            bookend_end=[],
        )

    anchor_pos = next(
        (i for i, c in enumerate(chunks) if c.chunk_id == anchor_id),
        len(chunks) // 2,
    )

    # bookend_start: first _BOOKEND_N chunks
    bookend_start = chunks[:_BOOKEND_N]

    # anchor_window: ±window around anchor, but don't duplicate bookend_start
    win_lo = max(0, anchor_pos - window)
    win_hi = min(len(chunks), anchor_pos + window + 1)
    anchor_window = chunks[win_lo:win_hi]

    # bookend_end: last _BOOKEND_N chunks
    bookend_end = chunks[max(0, len(chunks) - _BOOKEND_N):]

    return _SessionArc(
        session_id=session_id,
        anchor_chunk_id=anchor_id,
        bookend_start=bookend_start,
        anchor_window=anchor_window,
        bookend_end=bookend_end,
    )


def fts_discovery(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 3,
    window: int = _WINDOW_DEFAULT,
) -> list[_SessionArc]:
    """Run FTS5 search and return one arc per matching session.

    Deduplicates by session: only the best (FTS rank) match per session is
    used as the anchor; remaining matches in the same session are ignored.
    """
    rows = conn.execute(
        "SELECT chunks.id, chunks.session_id, chunks_fts.rank "
        "FROM chunks_fts "
        "JOIN chunks ON chunks.id = chunks_fts.rowid "
        "WHERE chunks_fts MATCH ? "
        "ORDER BY chunks_fts.rank "
        "LIMIT ?",
        (query, limit * 10),  # fetch extras to cover multi-match sessions
    ).fetchall()

    # Deduplicate by session_id — keep best rank (smallest abs(rank), FTS5 uses negative ranks)
    seen_sessions: dict[int, tuple[int, float]] = {}  # session_id -> (chunk_id, rank)
    for chunk_id, session_id, rank in rows:
        if session_id not in seen_sessions or rank < seen_sessions[session_id][1]:
            seen_sessions[session_id] = (chunk_id, rank)

    # Sort by best rank (FTS5 uses negative BM25 scores — smaller is better)
    # before slicing so the highest-ranked sessions are always selected.
    top_sessions = sorted(seen_sessions.items(), key=lambda kv: kv[1][1])[:limit]

    arcs: list[_SessionArc] = []
    for session_id, (anchor_id, _rank) in top_sessions:
        chunks = _fetch_session_chunks(conn, session_id)
        arc = _build_arc(chunks, anchor_id, window)
        arc.session_id = session_id
        arcs.append(arc)
    return arcs


def expand_around(
    conn: sqlite3.Connection,
    chunk_id: int,
    *,
    window: int = _WINDOW_DEFAULT,
) -> _SessionArc | None:
    """Return a window of chunks centered on *chunk_id*."""
    row = conn.execute(
        "SELECT session_id FROM chunks WHERE id = ?", (chunk_id,)
    ).fetchone()
    if row is None:
        return None
    session_id = row[0]
    chunks = _fetch_session_chunks(conn, session_id)
    arc = _build_arc(chunks, chunk_id, window)
    arc.session_id = session_id
    return arc


def _format_chunk(c: _Chunk) -> str:
    content = c.content
    if len(content) > 400:
        content = content[:397] + "…"
    return f"  [{c.role}] {content}"


def _format_arc(arc: _SessionArc, label: str | None = None) -> str:
    parts: list[str] = []
    header = f"Session {arc.session_id}"
    if label:
        header += f" ({label})"
    parts.append(header)

    def _section(name: str, chunks: list[_Chunk], anchor_id: int) -> None:
        if not chunks:
            return
        parts.append(f"  [{name}]")
        for c in chunks:
            marker = " ◄" if c.chunk_id == anchor_id else ""
            content = c.content[:400] + ("…" if len(c.content) > 400 else "")
            parts.append(f"    [{c.role}]{marker} {content}")

    _section("start", arc.bookend_start, arc.anchor_chunk_id)

    # Avoid repeating chunks that are already in bookend_start
    start_ids = {c.chunk_id for c in arc.bookend_start}
    end_ids = {c.chunk_id for c in arc.bookend_end}
    mid = [c for c in arc.anchor_window if c.chunk_id not in start_ids and c.chunk_id not in end_ids]
    if mid:
        _section("context", mid, arc.anchor_chunk_id)

    _section("end", arc.bookend_end, arc.anchor_chunk_id)
    return "\n".join(parts)


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mode": {
            "type": "string",
            "enum": ["discovery", "expand"],
            "description": (
                "'discovery': FTS5 keyword search returning one session arc "
                "per matching session (opener + match context + closer). "
                "'expand': show chunks around a specific chunk_id surfaced by "
                "session_search or a prior discovery call."
            ),
        },
        "query": {
            "type": "string",
            "description": "FTS5 keyword query. Required for mode='discovery'.",
        },
        "chunk_id": {
            "type": "integer",
            "description": "Chunk ID to expand around. Required for mode='expand'.",
        },
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": (
                "Max sessions to return in discovery mode. Default 3."
            ),
        },
        "window": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": (
                f"Chunks on each side of the anchor to include. Default {_WINDOW_DEFAULT}."
            ),
        },
    },
    "required": ["mode"],
}


@tool_parameters(_SCHEMA)
class SessionBrowseTool(Tool):
    """Browse past session content with full conversational context.

    Two modes:

    **discovery** — FTS5 keyword search. Returns one arc per matching session:
    session opener (first 3 turns), the part around the keyword match, and
    the session closer (last 3 turns). Useful for finding what happened in a
    session that dealt with a specific topic.

    **expand** — show chunks surrounding a specific ``chunk_id`` (from
    ``session_search`` or a prior discovery call). Use when you need more
    context around a match you found elsewhere.

    No embedding calls — pure SQL, always fast.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "session_browse"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        mode = kwargs.get("mode", "")
        if mode == "discovery":
            return self._do_discovery(kwargs)
        if mode == "expand":
            return self._do_expand(kwargs)
        return "Error: mode must be 'discovery' or 'expand'."

    def _do_discovery(self, kwargs: dict) -> str:
        query = (kwargs.get("query") or "").strip()
        if not query:
            return "Error: query is required for mode='discovery'."
        limit = int(kwargs.get("limit") or 3)
        window = int(kwargs.get("window") or _WINDOW_DEFAULT)

        try:
            arcs = fts_discovery(self._hook.db, query, limit=limit, window=window)
        except Exception as e:
            return f"Error: FTS5 search failed: {e}"

        if not arcs:
            return "no matches"

        return "\n\n".join(
            _format_arc(arc, label=f"result {i + 1}")
            for i, arc in enumerate(arcs)
        )

    def _do_expand(self, kwargs: dict) -> str:
        chunk_id = kwargs.get("chunk_id")
        if chunk_id is None:
            return "Error: chunk_id is required for mode='expand'."
        window = int(kwargs.get("window") or _WINDOW_DEFAULT)

        try:
            arc = expand_around(self._hook.db, int(chunk_id), window=window)
        except Exception as e:
            return f"Error: expand failed: {e}"

        if arc is None:
            return f"no chunk found with id={chunk_id}"

        return _format_arc(arc, label="expanded")
