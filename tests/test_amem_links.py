"""Tests for A-MEM (Phase 6) — keywords/tags/context/importance + linking."""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np

import nano_hermes
from conftest import _make_loop, _patch_embedding
from nano_hermes.memory.consolidation import (
    _clamp_importance,
    distill_hub_to_fact,
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
    return hook


def _insert_fact(db, content: str, *, importance: int = 5) -> int:
    cur = db.execute(
        "INSERT INTO semantic_facts (content, source_chunk_ids, created_at, "
        "keywords, tags, context, importance) "
        "VALUES (?, '[]', ?, '[]', '[]', '', ?)",
        (content, time.time(), importance),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_fact_vec(db, fact_id: int, vec: np.ndarray) -> None:
    db.execute(
        "INSERT OR REPLACE INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
        (fact_id, vec.astype(np.float32).tobytes()),
    )
    db.commit()


def _unit(axis: int, dims: int = 512) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    v[axis] = 1.0
    return v


class TestClampImportance:
    def test_in_range(self):
        assert _clamp_importance(7) == 7

    def test_low_clamped(self):
        assert _clamp_importance(0) == 1
        assert _clamp_importance(-5) == 1

    def test_high_clamped(self):
        assert _clamp_importance(11) == 10
        assert _clamp_importance(999) == 10

    def test_non_numeric_defaults(self):
        assert _clamp_importance("not a number") == 5
        assert _clamp_importance(None) == 5


class TestDistillHubToFactJSON:
    async def test_parses_full_json_payload(self, tmp_path):
        hook = _make_hook(tmp_path)
        payload = json.dumps({
            "fact": "Always validate input before persisting.",
            "keywords": ["validate", "input", "persist"],
            "tags": ["correctness", "safety"],
            "context": "When writing to durable stores.",
            "importance": 8,
        })
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response(payload))
        hook._loop.model = "test-model"
        out = await distill_hub_to_fact(hook, {"samples": ["s1"], "sessions": [1, 2]})
        assert out is not None
        assert out["fact"] == "Always validate input before persisting."
        assert out["keywords"] == ["validate", "input", "persist"]
        assert out["tags"] == ["correctness", "safety"]
        assert out["context"] == "When writing to durable stores."
        assert out["importance"] == 8

    async def test_strips_code_fence(self, tmp_path):
        hook = _make_hook(tmp_path)
        payload = (
            "```json\n"
            + json.dumps({"fact": "F.", "importance": 6})
            + "\n```"
        )
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response(payload))
        hook._loop.model = "test-model"
        out = await distill_hub_to_fact(hook, {"samples": ["s"], "sessions": [1, 2]})
        assert out is not None
        assert out["fact"] == "F."
        assert out["importance"] == 6

    async def test_falls_back_on_non_json(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("A plain English fact.")
        )
        hook._loop.model = "test-model"
        out = await distill_hub_to_fact(hook, {"samples": ["s"], "sessions": [1, 2]})
        assert out is not None
        assert out["fact"] == "A plain English fact."
        assert out["keywords"] == []
        assert out["tags"] == []
        assert out["context"] == ""
        assert out["importance"] == 5  # default

    async def test_empty_response_returns_none(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response(""))
        hook._loop.model = "test-model"
        out = await distill_hub_to_fact(hook, {"samples": ["s"], "sessions": [1, 2]})
        assert out is None


class TestLinkNewFact:
    async def test_writes_vec_and_links_above_threshold(self, tmp_path, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        # Two prior facts with vectors aligned to the "search" axis.
        prior_a = _insert_fact(hook.db, "fact A about web search")
        _insert_fact_vec(hook.db, prior_a, _unit(0))
        prior_b = _insert_fact(hook.db, "fact B about web search")
        _insert_fact_vec(hook.db, prior_b, _unit(0))
        # New fact also matches the search axis.
        new_id = _insert_fact(hook.db, "new fact about duckduckgo search")
        n = await link_new_fact(hook, new_id, "duckduckgo search")
        assert n == 2
        rows = hook.db.execute(
            "SELECT fact_a_id, fact_b_id, similarity FROM semantic_fact_links"
        ).fetchall()
        assert len(rows) == 2
        # All similarities should be >= threshold
        for _, _, sim in rows:
            assert sim >= 0.78

    async def test_no_links_when_no_prior_facts(self, tmp_path, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        fid = _insert_fact(hook.db, "lonely fact")
        n = await link_new_fact(hook, fid, "lonely text")
        assert n == 0

    async def test_no_links_below_threshold(self, tmp_path, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        # Prior fact aligned to "search" axis.
        prior = _insert_fact(hook.db, "search fact")
        _insert_fact_vec(hook.db, prior, _unit(0))
        # New fact aligned to a different axis (unrelated → orthogonal → sim=0).
        new_id = _insert_fact(hook.db, "unrelated topic")
        n = await link_new_fact(hook, new_id, "unrelated text")
        assert n == 0

    async def test_canonical_a_lt_b(self, tmp_path, monkeypatch):
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        prior = _insert_fact(hook.db, "search fact 1")
        _insert_fact_vec(hook.db, prior, _unit(0))
        new_id = _insert_fact(hook.db, "search fact 2")
        await link_new_fact(hook, new_id, "duckduckgo search")
        row = hook.db.execute(
            "SELECT fact_a_id, fact_b_id FROM semantic_fact_links"
        ).fetchone()
        a, b = row
        assert a < b


class TestSemanticFactsVecTrigger:
    def test_delete_fact_removes_vec_row(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        fid = _insert_fact(hook.db, "to be deleted")
        _insert_fact_vec(hook.db, fid, _unit(0))
        # Sanity: vec row exists.
        assert hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts_vec WHERE fact_id = ?", (fid,)
        ).fetchone()[0] == 1
        hook.db.execute("DELETE FROM semantic_facts WHERE id = ?", (fid,))
        hook.db.commit()
        # Trigger should have removed the vec row.
        assert hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts_vec WHERE fact_id = ?", (fid,)
        ).fetchone()[0] == 0


class TestNeighboursOf:
    def test_returns_both_sides_of_edge(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        f1 = _insert_fact(hook.db, "fact 1")
        f2 = _insert_fact(hook.db, "fact 2")
        f3 = _insert_fact(hook.db, "fact 3")
        # f1-f2, f1-f3 edges.
        hook.db.execute(
            "INSERT INTO semantic_fact_links VALUES (?, ?, ?, ?)",
            (min(f1, f2), max(f1, f2), 0.9, time.time()),
        )
        hook.db.execute(
            "INSERT INTO semantic_fact_links VALUES (?, ?, ?, ?)",
            (min(f1, f3), max(f1, f3), 0.85, time.time()),
        )
        hook.db.commit()
        nbrs = neighbours_of(hook.db, f1)
        ids = {n[0] for n in nbrs}
        assert ids == {f2, f3}
        # Order by similarity desc.
        assert nbrs[0][1] >= nbrs[1][1]


class TestDistillEndToEndWithLinks:
    async def test_distill_writes_columns_and_attempts_link(self, tmp_path, monkeypatch):
        """Full path: hub → JSON-aware distill → INSERT with new columns → link attempt.

        The embedding is patched, so linking actually runs and is verified.
        """
        _patch_embedding(monkeypatch)
        from tests.test_episodic_distillation import _seed_two_session_hub  # noqa: PLC0415
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        payload = json.dumps({
            "fact": "Recurring lesson: validate the embedding chain on startup.",
            "keywords": ["embedding", "validate", "startup"],
            "tags": ["bootstrap"],
            "context": "When configuring providers.",
            "importance": 9,
        })
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response(payload))

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        row = hook.db.execute(
            "SELECT content, keywords, tags, context, importance FROM semantic_facts"
        ).fetchone()
        content, keywords_json, tags_json, ctx, importance = row
        assert "validate the embedding chain" in content
        assert json.loads(keywords_json) == ["embedding", "validate", "startup"]
        assert json.loads(tags_json) == ["bootstrap"]
        assert ctx == "When configuring providers."
        assert importance == 9
        # The result string surfaces the new metadata.
        assert "importance=9" in result
        assert "bootstrap" in result


def _insert_fact_kt(db, content, *, keywords, tags, importance=5) -> int:
    cur = db.execute(
        "INSERT INTO semantic_facts (content, source_chunk_ids, created_at, "
        "keywords, tags, context, importance) VALUES (?, '[]', ?, ?, ?, '', ?)",
        (content, time.time(), json.dumps(keywords), json.dumps(tags), importance),
    )
    db.commit()
    return int(cur.lastrowid)


class TestNeighbourEvolution:
    async def test_closest_neighbour_gains_new_facts_tags(self, tmp_path, monkeypatch):
        from conftest import _FAKE_DIMS
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        db = hook.db
        vec = np.zeros(_FAKE_DIMS, dtype=np.float32)
        vec[0] = 1.0  # _FAKE_VEC_SEARCH ('duckduckgo')
        nbr = _insert_fact_kt(db, "duckduckgo old", keywords=["old"], tags=["search"])
        _insert_fact_vec(db, nbr, vec)
        new = _insert_fact_kt(db, "duckduckgo new", keywords=["fresh"], tags=["web"])

        written = await link_new_fact(hook, new, "duckduckgo new")
        assert written >= 1
        kw, tags = db.execute(
            "SELECT keywords, tags FROM semantic_facts WHERE id = ?", (nbr,)
        ).fetchone()
        assert "old" in json.loads(kw) and "fresh" in json.loads(kw)   # union
        assert set(json.loads(tags)) == {"search", "web"}

    async def test_disabled_leaves_neighbour_untouched(self, tmp_path, monkeypatch):
        from conftest import _FAKE_DIMS
        _patch_embedding(monkeypatch)
        hook = _make_hook(tmp_path)
        hook.config.memory.amem_evolve_neighbours = False
        db = hook.db
        vec = np.zeros(_FAKE_DIMS, dtype=np.float32)
        vec[0] = 1.0
        nbr = _insert_fact_kt(db, "duckduckgo old", keywords=["old"], tags=["search"])
        _insert_fact_vec(db, nbr, vec)
        new = _insert_fact_kt(db, "duckduckgo new", keywords=["fresh"], tags=["web"])

        await link_new_fact(hook, new, "duckduckgo new")
        kw = db.execute("SELECT keywords FROM semantic_facts WHERE id = ?", (nbr,)).fetchone()[0]
        assert json.loads(kw) == ["old"]  # unchanged

    def test_union_capped_preserves_existing(self):
        from nano_hermes.memory.links import _union_capped
        assert _union_capped(["a", "b"], ["b", "c", "d"], cap=3) == ["a", "b", "c"]
        assert _union_capped(["a"], [], cap=5) == ["a"]
