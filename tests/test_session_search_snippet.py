"""Tests for session_search snippet centering and CJK LIKE fallback."""
from __future__ import annotations

import sqlite3

import pytest

from nano_hermes.session.search import (
    _contains_cjk,
    _like_escape,
    _match_centered_snippet,
    sanitize_fts_query,
)


class TestSanitizeFtsQuery:
    def _run(self, sql_query):
        c = sqlite3.connect(":memory:")
        c.execute("CREATE VIRTUAL TABLE t USING fts5(body)")
        c.execute("INSERT INTO t VALUES ('the deployment rollback caused an outage')")
        sq = sanitize_fts_query(sql_query)
        if not sq:
            return "empty"
        return c.execute("SELECT count(*) FROM t WHERE t MATCH ?", (sq,)).fetchone()[0]

    def test_prose_recall_restored(self):
        # Raw prose AND-joins to 0 hits; sanitized OR-query recovers the match.
        assert self._run("what happened with the deployment rollback") == 1

    def test_special_chars_no_syntax_error(self):
        # Unquoted "(prod)?" throws fts5 syntax error; quoting makes it literal.
        assert self._run("rollback (prod)?") == 1

    def test_punctuation_only_is_empty(self):
        assert sanitize_fts_query("?! ...") == ""

    def test_terms_are_quoted_and_or_joined(self):
        assert sanitize_fts_query("alpha beta") == '"alpha" OR "beta"'

    def test_embedded_quote_stripped(self):
        assert sanitize_fts_query('foo"bar') == '"foobar"'

    def test_cjk_kept_as_single_quoted_token(self):
        assert sanitize_fts_query("部署回滚") == '"部署回滚"'


class TestFtsOrderByRank:
    """Regression: the lexical channel must return BM25-ranked ids, not rowid
    order — hybrid_search's FTS subquery relies on `ORDER BY rank`."""

    def _db(self, tmp_path):
        from nano_hermes.session.db import open_db
        db = open_db(str(tmp_path / "state.db"), 512)
        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        return db

    def _add_chunk(self, db, content):
        cur = db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (1, 0, 'user', ?, 0)",
            (content,),
        )
        db.commit()
        return int(cur.lastrowid)

    def test_high_bm25_ranks_ahead_of_lower_rowid(self, tmp_path):
        from nano_hermes.session.search import _fts_rows
        db = self._db(tmp_path)
        # Low BM25 term buried in a long doc, inserted FIRST (lower rowid).
        weak = self._add_chunk(db, "a long note about many topics including rollback and more prose here")
        # High BM25: short doc, term repeated, inserted SECOND (higher rowid).
        strong = self._add_chunk(db, "rollback rollback rollback")
        ids = [
            r[0]
            for r in _fts_rows(
                db,
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                "rollback",
                10,
            )
        ]
        # BM25 order → strong (higher rowid) first; rowid order would put weak first.
        assert ids[0] == strong
        assert set(ids) == {weak, strong}


class TestContainsCjk:
    def test_ascii_false(self):
        assert not _contains_cjk("hello world")

    def test_chinese_true(self):
        assert _contains_cjk("你好世界")

    def test_japanese_hiragana_true(self):
        assert _contains_cjk("こんにちは")

    def test_katakana_true(self):
        assert _contains_cjk("コンピュータ")

    def test_korean_true(self):
        assert _contains_cjk("안녕하세요")

    def test_mixed_ascii_cjk_true(self):
        assert _contains_cjk("query: 人工知能")

    def test_empty_false(self):
        assert not _contains_cjk("")

    def test_supplementary_plane_cjk_true(self):
        # CJK Extension B (U+20000–U+2A6DF) lives above the BMP.
        # _CJK_RE must cover \U00020000-\U0002FFFF or the LIKE fallback
        # never fires for these rare but valid characters.
        assert _contains_cjk("\U00020000")


class TestMatchCenteredSnippet:
    def test_short_text_returned_unchanged(self):
        text = "short text"
        assert _match_centered_snippet(text, "short", max_chars=240) == text

    def test_exact_max_chars_returned_unchanged(self):
        text = "x" * 240
        assert _match_centered_snippet(text, "x", max_chars=240) == text

    def test_tier1_phrase_match_centered(self):
        # Put the query phrase deep inside the text so naive head-truncation
        # would miss it entirely.
        prefix = "a" * 300
        phrase = "the target phrase"
        suffix = "z" * 200
        text = prefix + phrase + suffix
        snippet = _match_centered_snippet(text, "the target phrase", max_chars=100)
        assert "the target phrase" in snippet
        assert snippet.startswith("…")

    def test_tier2_co_occurrence_centered(self):
        # Two terms that appear together 400 chars in; individual terms also
        # appear near position 0 — proximity co-occurrence must win.
        filler = "alpha " * 60          # ~360 chars, 'alpha' at start
        cluster = "alpha gamma gamma"   # both terms near each other
        text = filler + cluster + " " + "z" * 200
        snippet = _match_centered_snippet(text, "alpha gamma", max_chars=80)
        # The cluster ("alpha gamma gamma") should appear in the window.
        assert "gamma" in snippet

    def test_tier3_individual_term_fallback(self):
        # 'rare_term' appears once deep in the text; no phrase / co-occurrence.
        prefix = "b" * 300
        text = prefix + "rare_term" + "c" * 100
        snippet = _match_centered_snippet(text, "rare_term", max_chars=60)
        assert "rare_term" in snippet

    def test_no_match_returns_head(self):
        text = "a" * 300 + "b" * 100
        snippet = _match_centered_snippet(text, "zzznothere", max_chars=100)
        # Must still return max_chars chars and not crash.
        assert len(snippet) <= 100 + 1  # +1 for trailing ellipsis
        assert snippet.startswith("a")

    def test_ellipsis_added_when_truncated(self):
        text = "x" * 100 + "FOUND" + "y" * 400
        snippet = _match_centered_snippet(text, "FOUND", max_chars=50)
        assert "…" in snippet

    def test_no_ellipsis_on_full_match(self):
        text = "hello world"
        snippet = _match_centered_snippet(text, "hello", max_chars=240)
        assert "…" not in snippet

    def test_window_bias_25_before_75_after(self):
        # Match is at position 0; window should start at 0 (bias floor).
        text = "TARGET" + "z" * 400
        snippet = _match_centered_snippet(text, "TARGET", max_chars=100)
        assert snippet.startswith("TARGET")

    def test_multiple_occurrences_best_coverage(self):
        # Two closely-spaced occurrences near position 200; one isolated at 0.
        text = "hit " + "a" * 200 + "hit hit " + "b" * 200
        snippet = _match_centered_snippet(text, "hit", max_chars=60)
        # The window covering two occurrences should be chosen.
        assert snippet.count("hit") >= 2


class TestLikeEscape:
    def test_percent_escaped(self):
        assert _like_escape("50% done") == "50\\% done"

    def test_underscore_escaped(self):
        assert _like_escape("user_id") == "user\\_id"

    def test_backslash_escaped(self):
        assert _like_escape("a\\b") == "a\\\\b"

    def test_plain_text_unchanged(self):
        assert _like_escape("hello world") == "hello world"

    def test_cjk_text_unchanged(self):
        assert _like_escape("人工智能") == "人工智能"


class TestCjkFallbackIntegration:
    """Integration tests: CJK queries on a real DB should find content via LIKE."""

    @pytest.mark.asyncio
    async def test_cjk_query_finds_chunk_via_like(self, tmp_path):
        from conftest import _make_loop

        import nano_hermes
        from nano_hermes.session.search import _contains_cjk, hybrid_search
        from nano_hermes.config import RetrievalConfig
        import numpy as np

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Insert a session row first (FK constraint on chunks.session_id).
        import time
        with hook.db:
            hook.db.execute(
                "INSERT INTO sessions (id, session_key, started_at) VALUES (1, 'test', ?)",
                (time.time(),),
            )

        # Insert a chunk with Chinese content.  Deliberately do NOT insert a
        # chunks_vec row: without a vector the ANN path cannot return this
        # chunk, so the only way it surfaces is through the CJK LIKE fallback.
        # This isolates the fallback path — if we inserted a zero vec and used
        # a zero query vec, sqlite-vec might return it via ANN (distance 0)
        # and the assertion would pass regardless of the LIKE code.
        cjk_text = "这是关于人工智能的讨论"
        with hook.db:
            hook.db.execute(
                "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
                "VALUES (1, 0, 'user', ?, ?)",
                (cjk_text, time.time()),
            )
            chunk_id = hook.db.execute("SELECT last_insert_rowid()").fetchone()[0]
            # FTS trigger fires automatically on INSERT INTO chunks.
            # No chunks_vec row — vector path cannot return this chunk.

        cfg = RetrievalConfig()
        query = "人工智能"
        assert _contains_cjk(query)

        qv = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
        hits = hybrid_search(hook.db, query, qv, cfg)

        # The CJK LIKE fallback must have fired and surfaced the chunk.
        chunk_ids = {h.chunk_id for h in hits}
        assert chunk_id in chunk_ids, (
            f"CJK LIKE fallback failed — chunk {chunk_id} not in {chunk_ids}"
        )


class TestTerminalFallbackNeverRaises:
    """The last-resort _fts_only_fallback is returned straight from search()'s
    except handlers with no outer guard — it must never propagate an error."""

    def test_non_operationalerror_returns_graceful(self, tmp_path, monkeypatch):
        from conftest import _make_loop

        import nano_hermes
        from nano_hermes.config import RetrievalConfig

        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)
        tool = loop.tools.get("session_search")

        def _boom(*a, **k):
            raise sqlite3.DatabaseError("disk I/O error")  # NOT an OperationalError

        monkeypatch.setattr("nano_hermes.session.search._fts_rows", _boom)
        out = tool._fts_only_fallback("some query", RetrievalConfig(), reason="embedder down")
        assert out.startswith("no matches")
        assert "embedder down" in out
