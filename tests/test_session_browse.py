"""Tests for session_browse — FTS5 discovery + expand modes."""
from __future__ import annotations

import time

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.session.browse import (
    SessionBrowseTool,
    _build_arc,
    _Chunk,
    expand_around,
    fts_discovery,
)


def _seed_session(hook, chunks: list[str]) -> int:
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s_{time.time()}", time.time()),
    )
    session_id = cur.lastrowid
    for i, content in enumerate(chunks):
        hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (session_id, i, content, time.time()),
        )
    hook.db.commit()
    return session_id


class TestBuildArc:
    def _chunks(self, n: int) -> list[_Chunk]:
        return [_Chunk(chunk_id=i, turn_index=i, role="user", content=f"msg {i}") for i in range(n)]

    def test_anchor_in_middle(self):
        chunks = self._chunks(10)
        arc = _build_arc(chunks, anchor_id=5, window=2)
        assert arc.anchor_chunk_id == 5
        assert len(arc.bookend_start) == 3  # first 3
        assert len(arc.bookend_end) == 3    # last 3
        assert any(c.chunk_id == 5 for c in arc.anchor_window)

    def test_anchor_is_first_chunk(self):
        chunks = self._chunks(5)
        arc = _build_arc(chunks, anchor_id=0, window=2)
        assert arc.anchor_chunk_id == 0

    def test_empty_chunks(self):
        arc = _build_arc([], anchor_id=99, window=2)
        assert arc.bookend_start == []
        assert arc.anchor_window == []
        assert arc.bookend_end == []

    def test_single_chunk(self):
        chunks = [_Chunk(chunk_id=42, turn_index=0, role="user", content="only")]
        arc = _build_arc(chunks, anchor_id=42, window=2)
        assert arc.anchor_chunk_id == 42
        assert len(arc.bookend_start) == 1


class TestFtsDiscovery:
    def test_returns_arc_for_matching_session(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        sid = _seed_session(hook, [
            "hello world",
            "deploy the application to production",
            "done",
        ])

        arcs = fts_discovery(hook.db, "deploy production", limit=3, window=2)
        assert len(arcs) == 1
        assert arcs[0].session_id == sid

    def test_deduplicates_by_session(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Two sessions matching the same query
        sid1 = _seed_session(hook, ["deploy app", "deploy again"])
        sid2 = _seed_session(hook, ["deploy another thing"])
        # Single session that appears twice in FTS — should only get one arc
        arcs = fts_discovery(hook.db, "deploy", limit=5, window=2)
        session_ids = [a.session_id for a in arcs]
        # No duplicates
        assert len(session_ids) == len(set(session_ids))
        assert sid1 in session_ids
        assert sid2 in session_ids

    def test_empty_when_no_match(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        _seed_session(hook, ["unrelated content here"])

        arcs = fts_discovery(hook.db, "xyzzy_not_in_db", limit=3, window=2)
        assert arcs == []

    def test_respects_limit(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        for _ in range(5):
            _seed_session(hook, ["deploy something"])

        arcs = fts_discovery(hook.db, "deploy", limit=2, window=2)
        assert len(arcs) <= 2


class TestExpandAround:
    def test_returns_window_around_chunk(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        sid = _seed_session(hook, [f"message {i}" for i in range(10)])
        # Grab a chunk ID from the middle
        rows = hook.db.execute(
            "SELECT id FROM chunks WHERE session_id = ? ORDER BY turn_index",
            (sid,),
        ).fetchall()
        mid_id = rows[5][0]

        arc = expand_around(hook.db, mid_id, window=2)
        assert arc is not None
        assert arc.session_id == sid
        assert arc.anchor_chunk_id == mid_id
        # window=2 means ±2 around position 5 → positions 3..7
        assert any(c.chunk_id == mid_id for c in arc.anchor_window)

    def test_returns_none_for_missing_chunk(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        arc = expand_around(hook.db, 999999, window=3)
        assert arc is None


class TestSessionBrowseTool:
    async def test_discovery_returns_string(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_session(hook, ["deploy the server", "it worked"])
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="discovery", query="deploy")
        assert "Session" in result
        assert "deploy" in result.lower()

    async def test_discovery_no_match_returns_no_matches(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="discovery", query="xyzzy_nonexistent")
        assert "no matches" in result

    async def test_discovery_requires_query(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="discovery")
        assert result.startswith("Error")

    async def test_expand_requires_chunk_id(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="expand")
        assert result.startswith("Error")

    async def test_expand_missing_chunk(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="expand", chunk_id=999999)
        assert "no chunk found" in result

    async def test_expand_returns_context(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        sid = _seed_session(hook, [f"turn {i}" for i in range(8)])
        rows = hook.db.execute(
            "SELECT id FROM chunks WHERE session_id = ? ORDER BY turn_index",
            (sid,),
        ).fetchall()
        mid_id = rows[4][0]

        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="expand", chunk_id=mid_id, window=2)
        assert f"Session {sid}" in result
        assert "turn 4" in result

    async def test_invalid_mode_returns_error(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        tool = SessionBrowseTool(hook=hook)
        result = await tool.execute(mode="invalid")
        assert result.startswith("Error")
