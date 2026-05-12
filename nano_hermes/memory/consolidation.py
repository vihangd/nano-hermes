"""Embedding-based memory consolidation (MemGPT/Letta pattern).

Greedy cosine-similarity clustering of memory entries: entries whose
similarity to an existing cluster centroid exceeds *threshold* are merged
into that cluster; the longest entry survives.  Agent-invoked, not
automatic — the agent calls memory_patch(action="consolidate") when it
judges memory is bloated.

Vectors from EmbeddingChain are L2-normalised, so np.dot == cosine sim.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from ..embedding.chain import EmbeddingChain

_DEFAULT_THRESHOLD = 0.92


def split_entries(text: str) -> list[str]:
    """Split a memory slot's text into individual entries.

    Tries paragraph splits (double-newline) first; falls back to
    line splits so single-line bullet entries are handled correctly.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if len(paragraphs) > 1:
        return paragraphs
    return [line.strip() for line in text.splitlines() if line.strip()]


def greedy_cluster(
    embeddings: list[np.ndarray],
    threshold: float = _DEFAULT_THRESHOLD,
) -> list[list[int]]:
    """Greedy cosine-similarity clustering.

    Walks embeddings in order; assigns each to the closest existing cluster
    if similarity >= threshold, otherwise starts a new one.  Centroids are
    updated as running means.  Returns a list of clusters, each a list of
    entry indices.
    """
    clusters: list[list[int]] = []
    centroids: list[np.ndarray] = []

    for i, vec in enumerate(embeddings):
        if not centroids:
            clusters.append([i])
            centroids.append(vec.copy())
            continue

        sims = [float(np.dot(vec, c)) for c in centroids]
        best = max(range(len(sims)), key=lambda x: sims[x])

        if sims[best] >= threshold:
            n = len(clusters[best])
            clusters[best].append(i)
            centroids[best] = centroids[best] + (vec - centroids[best]) / (n + 1)
        else:
            clusters.append([i])
            centroids.append(vec.copy())

    return clusters


async def consolidate_entries(
    entries: list[str],
    embedder_factory: "Callable[[], EmbeddingChain]",
    threshold: float = _DEFAULT_THRESHOLD,
) -> tuple[list[str], int]:
    """Cluster and deduplicate *entries* by semantic similarity.

    Returns ``(surviving_entries, n_removed)``.  For each cluster the
    longest entry is kept as the canonical form — longer entries tend to
    carry more context.
    """
    if len(entries) < 2:
        return list(entries), 0

    async with embedder_factory() as chain:
        vecs = await chain.embed(entries)

    clusters = greedy_cluster(vecs, threshold)

    surviving = [entries[max(cl, key=lambda i: len(entries[i]))] for cl in clusters]
    n_removed = len(entries) - len(surviving)
    return surviving, n_removed
