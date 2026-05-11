"""Tests for MMR diversity reranking (session/mmr.py)."""
from __future__ import annotations

import pytest

from nano_hermes.session.mmr import MMRHit, mmr_rerank, _jaccard


def _hit(chunk_id: int, content: str, score: float) -> MMRHit:
    return MMRHit(chunk_id=chunk_id, session_id=1, content=content, score=score)


class TestJaccard:
    def test_identical_sets(self):
        a = frozenset(["a", "b", "c"])
        assert _jaccard(a, a) == pytest.approx(1.0)

    def test_disjoint_sets(self):
        a = frozenset(["a", "b"])
        b = frozenset(["c", "d"])
        assert _jaccard(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = frozenset(["a", "b", "c"])
        b = frozenset(["b", "c", "d"])
        # intersection={b,c}=2, union={a,b,c,d}=4
        assert _jaccard(a, b) == pytest.approx(0.5)

    def test_both_empty(self):
        assert _jaccard(frozenset(), frozenset()) == pytest.approx(1.0)


class TestMMRRerank:
    def test_empty_input(self):
        assert mmr_rerank([]) == []

    def test_single_hit(self):
        h = _hit(1, "only hit", 1.0)
        result = mmr_rerank([h])
        assert len(result) == 1
        assert result[0].chunk_id == 1

    def test_lambda_1_preserves_order(self):
        """λ=1.0 → pure relevance, order unchanged."""
        hits = [
            _hit(1, "alpha beta", 0.9),
            _hit(2, "gamma delta", 0.7),
            _hit(3, "epsilon zeta", 0.5),
        ]
        result = mmr_rerank(hits, lam=1.0)
        assert [h.chunk_id for h in result] == [1, 2, 3]

    def test_near_duplicate_suppressed(self):
        """Two near-identical hits → the second should be pushed down by MMR."""
        # Hit 1 and Hit 2 are nearly identical; Hit 3 is diverse.
        hits = [
            _hit(1, "the cat sat on the mat the cat sat", 0.9),
            _hit(2, "the cat sat on the mat the cat sat again", 0.85),  # near-dup
            _hit(3, "machine learning neural network deep learning", 0.7),
        ]
        result = mmr_rerank(hits, lam=0.5, k=2)
        ids = [h.chunk_id for h in result]
        assert 1 in ids, "most relevant hit should be first"
        # Hit 3 (diverse) should beat near-dup hit 2
        assert 3 in ids, f"diverse hit should be selected over near-dup; got {ids}"

    def test_k_limits_output(self):
        hits = [_hit(i, f"content {i}", 1.0 / (i + 1)) for i in range(10)]
        result = mmr_rerank(hits, lam=0.7, k=3)
        assert len(result) == 3

    def test_diversity_at_low_lambda(self):
        """At λ=0.0, second selected item should be maximally different from first."""
        hits = [
            _hit(1, "python programming language", 0.9),
            _hit(2, "python programming language code", 0.85),   # near-dup of 1
            _hit(3, "cooking recipe pasta sauce tomato", 0.6),   # very different
        ]
        result = mmr_rerank(hits, lam=0.0, k=2)
        ids = [h.chunk_id for h in result]
        assert 3 in ids, f"diverse hit should be selected at λ=0; got {ids}"

    def test_score_preserved_in_output(self):
        """RRF scores must be carried through unchanged."""
        h = _hit(42, "test content", 0.314)
        result = mmr_rerank([h])
        assert result[0].score == pytest.approx(0.314)


class TestMMRIntegration:
    """MMR wired into hybrid_search via RetrievalConfig.mmr_lambda."""

    def test_mmr_lambda_config_default(self):
        from nano_hermes.config import RetrievalConfig
        cfg = RetrievalConfig()
        assert cfg.mmr_lambda == pytest.approx(0.7)

    def test_mmr_lambda_1_equals_no_mmr(self):
        """mmr_lambda=1.0 should return same order as pure relevance."""
        from nano_hermes.session.mmr import mmr_rerank

        hits = [
            _hit(1, "relevant document", 0.9),
            _hit(2, "less relevant doc", 0.5),
        ]
        result_mmr = mmr_rerank(hits, lam=1.0, k=2)
        assert [h.chunk_id for h in result_mmr] == [1, 2]
