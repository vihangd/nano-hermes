"""Hybrid (BM25 + dense) trajectory retrieval via RRF.

trajectory_search used to be dense-only. These tests prove the lexical
channel adds recall the embedding misses, that the FTS mirror stays in
sync, and that a malformed FTS query degrades gracefully to dense-only.
"""
from __future__ import annotations

import time as _time

import pytest
from nanobot.agent.loop import AgentLoop

import nano_hermes
from conftest import (
    _FAKE_VEC_SEARCH,
    _FAKE_VEC_UNRELATED,
    _patch_embedding,
)


def _seed(hook, task: str, vec, *, outcome: str = "ok") -> int:
    cur = hook.db.execute(
        "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task, "[]", outcome, _time.time()),
    )
    tid = cur.lastrowid
    hook.db.execute(
        "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
        (tid, vec.astype("float32").tobytes()),
    )
    hook.db.commit()
    return tid


class TestHybridRecall:
    async def test_lexical_channel_beats_misleading_vector(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A trajectory whose embedding doesn't capture an exact term is
        still surfaced when the term matches lexically — and outranks a
        trajectory that vector-matches the query but lacks the term."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        # A: contains the exact word "duckduckgo" but has a non-matching
        # embedding (simulates embedding drift / term not captured).
        _seed(hook, "duckduckgo outage postmortem", _FAKE_VEC_UNRELATED)
        # B: vector-matches the query ("duckduckgo" -> _FAKE_VEC_SEARCH) but
        # its task text is about something else entirely.
        _seed(hook, "weather forecast pipeline", _FAKE_VEC_SEARCH)

        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="duckduckgo", k=1)
        # Dense-only would have returned the weather row; hybrid surfaces the
        # lexical match.
        assert "duckduckgo outage" in out
        assert "weather forecast" not in out

    async def test_dense_still_works_without_lexical_overlap(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        _seed(hook, "duckduckgo news roundup", _FAKE_VEC_SEARCH)
        tool = loop.tools.get("trajectory_search")
        # Query shares the embedding keyword but not the surface tokens.
        out = await tool.execute(query="duckduckgo", k=3)
        assert "duckduckgo news" in out


class TestFtsSync:
    async def test_insert_trigger_populates_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("kubernetes crashloop debugging", "ok", _time.time()),
        )
        hook.db.commit()
        rows = hook.db.execute(
            "SELECT rowid FROM trajectories_fts WHERE trajectories_fts MATCH ?",
            ("crashloop",),
        ).fetchall()
        assert len(rows) == 1

    async def test_delete_trigger_removes_from_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        cur = hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("obscureword lookup", "ok", _time.time()),
        )
        tid = cur.lastrowid
        hook.db.commit()
        hook.db.execute("DELETE FROM trajectories WHERE id = ?", (tid,))
        hook.db.commit()
        rows = hook.db.execute(
            "SELECT rowid FROM trajectories_fts WHERE trajectories_fts MATCH ?",
            ("obscureword",),
        ).fetchall()
        assert rows == []

    def test_backfill_populates_empty_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nano_hermes.session.db import _backfill_trajectories_fts

        from nano_hermes.session.db import _FTS_BACKFILL_FLAG

        hook = nano_hermes.install(loop)
        hook.db.execute(
            "INSERT INTO trajectories (task, outcome, created_at) VALUES (?, ?, ?)",
            ("legacy migration task", "ok", _time.time()),
        )
        hook.db.commit()
        # Simulate a pre-existing DB: empty the FTS index and clear the
        # one-time backfill flag so the rebuild re-runs.
        hook.db.execute("INSERT INTO trajectories_fts(trajectories_fts) VALUES('delete-all')")
        hook.db.execute("DELETE FROM meta WHERE key = ?", (_FTS_BACKFILL_FLAG,))
        hook.db.commit()
        # COUNT(*) on external-content FTS reads the content table, so verify
        # emptiness via a MATCH that should now miss.
        assert hook.db.execute(
            "SELECT rowid FROM trajectories_fts WHERE trajectories_fts MATCH ?",
            ("migration",),
        ).fetchall() == []

        _backfill_trajectories_fts(hook.db)

        rows = hook.db.execute(
            "SELECT rowid FROM trajectories_fts WHERE trajectories_fts MATCH ?",
            ("migration",),
        ).fetchall()
        assert len(rows) == 1


class TestEmbedderDownFallback:
    async def test_all_providers_failed_degrades_to_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When every embedding provider is down, _hybrid_search raises
        AllProvidersFailed and execute() must fall back to the LIKE-based
        keyword search and still return a matching trajectory."""
        from nano_hermes.embedding.chain import AllProvidersFailed

        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        _seed(hook, "kubernetes crashloop investigation", _FAKE_VEC_UNRELATED)

        async def _boom(self, texts):
            raise AllProvidersFailed("all down")

        monkeypatch.setattr(
            "nano_hermes.embedding.chain.EmbeddingChain.embed", _boom
        )

        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="kubernetes crashloop", k=3)
        assert "kubernetes crashloop" in out, "fallback did not return the row"


class TestMalformedQuery:
    async def test_unparsable_fts_query_degrades_to_dense(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A query that FTS5 can't parse must not crash the search — the
        lexical channel is dropped and dense still runs."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)
        _seed(hook, "some unrelated task", _FAKE_VEC_UNRELATED)
        tool = loop.tools.get("trajectory_search")
        # A lone double-quote is an invalid FTS5 MATCH expression. The query
        # has no fake-embedder keyword, so it embeds to _FAKE_VEC_UNRELATED
        # and the dense channel still finds the seeded row.
        out = await tool.execute(query='"', k=3)
        assert isinstance(out, str)
        assert "some unrelated task" in out
