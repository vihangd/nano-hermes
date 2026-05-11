"""Maximal Marginal Relevance (MMR) reranking for session_search results.

MMR selects hits that are relevant to the query but diverse from each other:
  MMR(d) = λ · relevance(d) − (1−λ) · max_sim(d, already_selected)

Similarity is computed via Jaccard on token sets (no embedding round-trip,
O(n²) but cheap for the small candidate pools used here, typically ≤50 docs).

λ=1.0 → pure relevance (same order as RRF input).
λ=0.7 → recommended balance (plan default).
λ=0.0 → pure diversity (greedy cover).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence


@dataclass
class MMRHit:
    chunk_id: int
    session_id: int
    content: str
    score: float  # original RRF score (preserved for downstream display)


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(text.lower()))


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


def mmr_rerank(
    hits: Sequence[MMRHit],
    *,
    lam: float = 0.7,
    k: int | None = None,
) -> list[MMRHit]:
    """Return up to *k* hits reranked by MMR diversity.

    *hits* must already be sorted by descending relevance (RRF score).
    *lam* controls the relevance/diversity trade-off (0.7 is recommended).
    *k* defaults to ``len(hits)`` (rerank all, don't truncate).
    """
    if not hits:
        return []
    if lam == 1.0:
        return list(hits)[:k]

    k = k if k is not None else len(hits)
    token_sets = [_tokenize(h.content) for h in hits]
    max_rrf = max(h.score for h in hits) or 1.0

    selected: list[MMRHit] = []
    selected_tokens: list[frozenset[str]] = []
    remaining = list(range(len(hits)))

    while remaining and len(selected) < k:
        best_idx = -1
        best_score = float("-inf")

        for i in remaining:
            relevance = hits[i].score / max_rrf  # normalised to [0, 1]
            if not selected_tokens:
                diversity = 0.0
            else:
                diversity = max(
                    _jaccard(token_sets[i], sel_toks)
                    for sel_toks in selected_tokens
                )
            mmr_score = lam * relevance - (1 - lam) * diversity
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        selected.append(hits[best_idx])
        selected_tokens.append(token_sets[best_idx])
        remaining.remove(best_idx)

    return selected
