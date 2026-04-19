"""Tests for session_search snippet centering and CJK LIKE fallback."""
from __future__ import annotations

import pytest

from nano_hermes.session.search import _contains_cjk, _match_centered_snippet


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

        # Insert a chunk with Chinese content.
        cjk_text = "这是关于人工智能的讨论"
        with hook.db:
            hook.db.execute(
                "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
                "VALUES (1, 0, 'user', ?, ?)",
                (cjk_text, time.time()),
            )
            chunk_id = hook.db.execute("SELECT last_insert_rowid()").fetchone()[0]
            # FTS trigger fires automatically on INSERT INTO chunks.
            # Add a dummy vector row (zero vector).
            dim = hook.config.embedding.target_dims
            zero_vec = np.zeros(dim, dtype=np.float32).tobytes()
            hook.db.execute(
                "INSERT INTO chunks_vec(chunk_id, embedding) VALUES (?, ?)",
                (chunk_id, zero_vec),
            )

        cfg = RetrievalConfig()
        query = "人工智能"
        assert _contains_cjk(query)

        # Use a zero query vector — vec results will likely be empty or low.
        qv = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
        hits = hybrid_search(hook.db, query, qv, cfg)

        # The CJK LIKE fallback must surface this chunk.
        chunk_ids = {h.chunk_id for h in hits}
        assert chunk_id in chunk_ids, (
            f"CJK LIKE fallback failed — chunk {chunk_id} not in {chunk_ids}"
        )
