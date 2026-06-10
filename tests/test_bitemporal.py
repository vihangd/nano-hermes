"""Bi-temporal fact invalidation (Zep/Graphiti variant).

A newly distilled fact can supersede a near-duplicate prior fact. One LLM
call (gated by a cosine prefilter) decides; superseded facts get invalid_at
stamped non-destructively and are filtered from live views.
"""
from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np

import json as _json

import nano_hermes
from conftest import _FAKE_VEC_SEARCH, _make_loop, _patch_embedding
from nano_hermes.memory.bitemporal import (
    _parse_index_list,
    invalidate_superseded_facts,
    sweep_contradictions,
)
from nano_hermes.memory.links import link_new_fact, neighbours_of
from nano_hermes.memory.tool import MemoryPatchTool


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = text
    return resp


def _make_hook(tmp_path):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    hook._loop.provider = MagicMock()
    hook._loop.model = "test-model"
    return hook


def _insert_fact(db, content: str) -> int:
    cur = db.execute(
        "INSERT INTO semantic_facts (content, source_chunk_ids, created_at, "
        "keywords, tags, context, importance) "
        "VALUES (?, '[]', ?, '[]', '[]', '', 5)",
        (content, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_vec(db, fact_id: int, vec: np.ndarray) -> None:
    db.execute(
        "INSERT OR REPLACE INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
        (fact_id, vec.astype(np.float32).tobytes()),
    )
    db.commit()


def _unit(axis: int, dims: int = 512) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    v[axis] = 1.0
    return v


def _invalid_at(db, fact_id: int):
    return db.execute(
        "SELECT invalid_at FROM semantic_facts WHERE id = ?", (fact_id,)
    ).fetchone()[0]


class TestParseIndexList:
    def test_plain_array(self):
        assert _parse_index_list("[1, 3]") == [1, 3]

    def test_empty(self):
        assert _parse_index_list("[]") == []

    def test_prose_wrapped(self):
        assert _parse_index_list("The outdated ones are [2].") == [2]

    def test_code_fence(self):
        assert _parse_index_list("```json\n[1]\n```") == [1]

    def test_garbage(self):
        assert _parse_index_list("none of them") == []

    def test_non_int_entries_dropped(self):
        assert _parse_index_list('[1, "x", 2]') == [1, 2]


class TestInvalidateSuperseded:
    async def test_near_duplicate_superseded(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "deploy with `make deploy`")
        _insert_vec(db, old, _unit(0))
        new = _insert_fact(db, "deploy with `make ship` (renamed)")
        _insert_vec(db, new, _unit(0))  # near-identical embedding

        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("[1]")
        )

        result = await invalidate_superseded_facts(hook, new, "deploy with make ship")
        assert result == [old]
        assert _invalid_at(db, old) is not None
        assert _invalid_at(db, new) is None  # the new fact stays current

    async def test_llm_says_none_outdated(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "user prefers Python")
        _insert_vec(db, old, _unit(0))
        new = _insert_fact(db, "user also likes Rust")
        _insert_vec(db, new, _unit(0))

        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("[]")
        )
        result = await invalidate_superseded_facts(hook, new, "user also likes Rust")
        assert result == []
        assert _invalid_at(db, old) is None

    async def test_no_near_duplicate_skips_llm(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "completely unrelated fact")
        _insert_vec(db, old, _unit(1))  # orthogonal — below threshold
        new = _insert_fact(db, "a different topic entirely")
        _insert_vec(db, new, _unit(0))

        hook._loop.provider.chat_with_retry = AsyncMock()
        result = await invalidate_superseded_facts(hook, new, "a different topic")
        assert result == []
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_disabled_is_noop(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "old fact")
        _insert_vec(db, old, _unit(0))
        new = _insert_fact(db, "new fact")
        _insert_vec(db, new, _unit(0))

        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("[1]")
        )
        result = await invalidate_superseded_facts(hook, new, "new fact", enabled=False)
        assert result == []
        hook._loop.provider.chat_with_retry.assert_not_called()
        assert _invalid_at(db, old) is None

    async def test_missing_vector_returns_empty(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        new = _insert_fact(db, "no vec stored")  # no _insert_vec
        hook._loop.provider.chat_with_retry = AsyncMock()
        result = await invalidate_superseded_facts(hook, new, "no vec stored")
        assert result == []
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_never_raises_on_provider_error(self, tmp_path):
        """The docstring promises best-effort: a provider exception must not
        escape into distillation."""
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "old fact")
        _insert_vec(db, old, _unit(0))
        new = _insert_fact(db, "new fact")
        _insert_vec(db, new, _unit(0))
        hook._loop.provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("boom"))
        result = await invalidate_superseded_facts(hook, new, "new fact")
        assert result == []
        assert _invalid_at(db, old) is None

    async def test_out_of_range_and_duplicate_indices_handled(self, tmp_path):
        """LLM returns indices beyond the candidate count and a duplicate —
        only the valid one is invalidated, exactly once."""
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact(db, "the one real candidate")
        _insert_vec(db, old, _unit(0))
        new = _insert_fact(db, "supersedes it")
        _insert_vec(db, new, _unit(0))
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("[1, 5, 1]")  # 5 is out of range; 1 dup
        )
        result = await invalidate_superseded_facts(hook, new, "supersedes it")
        assert result == [old]
        assert _invalid_at(db, old) is not None


class TestDistillEndToEnd:
    async def test_distill_invalidates_superseded_fact(self, tmp_path, monkeypatch):
        """Full memory_patch(action='distill') path: a distilled fact stamps
        invalid_at on a near-duplicate prior fact and surfaces it in the output."""
        from tests.test_episodic_distillation import _seed_two_session_hub  # noqa: PLC0415

        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)

        # Pre-seed a prior fact the new one will supersede, with a vector that
        # matches what the distilled "duckduckgo" fact embeds to.
        old = _insert_fact(hook.db, "duckduckgo search uses the v1 API")
        _insert_vec(hook.db, old, _FAKE_VEC_SEARCH)

        distill_json = _json.dumps({
            "fact": "duckduckgo search now requires the v2 API",
            "keywords": ["duckduckgo", "api"],
            "tags": ["web"],
            "context": "When calling search.",
            "importance": 7,
        })
        # Call 1: distill_hub_to_fact. Call 2: supersession check.
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=[_mock_response(distill_json), _mock_response("[1]")]
        )

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        assert _invalid_at(hook.db, old) is not None, "prior fact not invalidated"
        assert "supersedes" in result
        # The new fact itself stays current.
        new_rows = hook.db.execute(
            "SELECT invalid_at FROM semantic_facts WHERE content LIKE '%v2 API%'"
        ).fetchone()
        assert new_rows is not None and new_rows[0] is None


class TestMigration:
    def test_invalid_at_added_to_legacy_semantic_facts(self, tmp_path):
        """A DB whose semantic_facts predates invalid_at must gain the column
        via _apply_migrations, and the live WHERE invalid_at IS NULL filter
        must then work."""
        import sqlite3

        from nano_hermes.session.db import _apply_migrations

        conn = sqlite3.connect(tmp_path / "legacy.db")
        # Legacy 4-column shape (pre-A-MEM, pre-bitemporal).
        conn.execute(
            "CREATE TABLE semantic_facts ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT,"
            " content TEXT NOT NULL,"
            " source_chunk_ids TEXT NOT NULL,"
            " created_at REAL NOT NULL)"
        )
        conn.execute(
            "INSERT INTO semantic_facts (content, source_chunk_ids, created_at) "
            "VALUES ('legacy fact', '[]', 1.0)"
        )
        conn.commit()

        _apply_migrations(conn)

        cols = [r[1] for r in conn.execute("PRAGMA table_info(semantic_facts)")]
        assert "invalid_at" in cols
        # Existing rows default to NULL = currently valid, so the live filter
        # returns them.
        rows = conn.execute(
            "SELECT content FROM semantic_facts WHERE invalid_at IS NULL"
        ).fetchall()
        assert rows == [("legacy fact",)]
        conn.close()


class TestInvalidFiltering:
    async def test_neighbours_exclude_invalid(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        a = _insert_fact(db, "fact a")
        b = _insert_fact(db, "fact b (will be invalidated)")
        db.execute(
            "INSERT INTO semantic_fact_links (fact_a_id, fact_b_id, similarity, created_at) "
            "VALUES (?, ?, ?, ?)",
            (min(a, b), max(a, b), 0.9, time.time()),
        )
        db.commit()
        assert [n for n, _ in neighbours_of(db, a)] == [b]

        db.execute(
            "UPDATE semantic_facts SET invalid_at = ? WHERE id = ?", (time.time(), b)
        )
        db.commit()
        assert neighbours_of(db, a) == []

    async def test_link_new_fact_skips_invalid(self, tmp_path, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        db = hook.db
        # An existing, already-superseded fact with a near-identical vector.
        stale = _insert_fact(db, "duckduckgo old behaviour")
        _insert_vec(db, stale, _unit(0))
        db.execute(
            "UPDATE semantic_facts SET invalid_at = ? WHERE id = ?", (time.time(), stale)
        )
        db.commit()

        fresh = _insert_fact(db, "duckduckgo new behaviour")
        # link_new_fact embeds "duckduckgo ..." -> _FAKE_VEC_SEARCH == _unit(0),
        # so the stale fact is a geometric neighbour, but it's invalid.
        n_links = await link_new_fact(hook, fresh, "duckduckgo new behaviour")
        assert n_links == 0
        links = db.execute("SELECT COUNT(*) FROM semantic_fact_links").fetchone()[0]
        assert links == 0


class TestSchema:
    def test_invalid_at_column_exists(self, tmp_path):
        hook = _make_hook(tmp_path)
        cols = [
            r[1]
            for r in hook.db.execute("PRAGMA table_info(semantic_facts)").fetchall()
        ]
        assert "invalid_at" in cols


def _insert_fact_at(db, content: str, created_at: float, importance: int = 5) -> int:
    cur = db.execute(
        "INSERT INTO semantic_facts (content, source_chunk_ids, created_at, "
        "keywords, tags, context, importance) VALUES (?, '[]', ?, '[]', '[]', '', ?)",
        (content, created_at, importance),
    )
    db.commit()
    return int(cur.lastrowid)


class TestContradictionSweep:
    async def test_retires_older_superseded_fact(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact_at(db, "deploy with `make deploy`", created_at=100.0)
        _insert_vec(db, old, _unit(0))
        new = _insert_fact_at(db, "deploy with `make ship` (renamed)", created_at=200.0)
        _insert_vec(db, new, _unit(0))  # near-identical embedding
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response("[1]"))

        n = await sweep_contradictions(hook, max_anchors=10)
        assert n == 1
        assert _invalid_at(db, old) is not None   # older retired
        assert _invalid_at(db, new) is None        # anchor (newer) kept

    async def test_noop_when_not_superseded(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        a = _insert_fact_at(db, "user prefers Python", created_at=100.0)
        _insert_vec(db, a, _unit(0))
        b = _insert_fact_at(db, "user also likes Rust", created_at=200.0)
        _insert_vec(db, b, _unit(0))
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response("[]"))

        n = await sweep_contradictions(hook, max_anchors=10)
        assert n == 0
        assert _invalid_at(db, a) is None

    async def test_disabled_returns_zero(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact_at(db, "x", created_at=100.0)
        _insert_vec(db, old, _unit(0))
        new = _insert_fact_at(db, "x updated", created_at=200.0)
        _insert_vec(db, new, _unit(0))
        called = AsyncMock(return_value=_mock_response("[1]"))
        hook._loop.provider.chat_with_retry = called

        n = await sweep_contradictions(hook, enabled=False, max_anchors=10)
        assert n == 0
        assert called.call_count == 0  # no LLM call when disabled

    async def test_audit_action_via_memory_tool(self, tmp_path):
        hook = _make_hook(tmp_path)
        db = hook.db
        old = _insert_fact_at(db, "config lives in settings.py", created_at=100.0)
        _insert_vec(db, old, _unit(0))
        new = _insert_fact_at(db, "config moved to config.toml", created_at=200.0)
        _insert_vec(db, new, _unit(0))
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response("[1]"))

        out = await MemoryPatchTool(hook=hook).execute(action="audit")
        assert out.startswith("ok: audit retired 1"), out
        assert _invalid_at(db, old) is not None
