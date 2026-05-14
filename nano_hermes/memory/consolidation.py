"""Embedding-based memory consolidation (MemGPT/Letta pattern).

Greedy cosine-similarity clustering of memory entries: entries whose
similarity to an existing cluster centroid exceeds *threshold* are merged
into that cluster; the longest entry survives.  Agent-invoked, not
automatic — the agent calls memory_patch(action="consolidate") when it
judges memory is bloated.

Vectors from EmbeddingChain are L2-normalised, so np.dot == cosine sim.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any, Callable

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
            new_c = centroids[best] + (vec - centroids[best]) / (n + 1)
            norm = float(np.linalg.norm(new_c))
            centroids[best] = new_c / norm if norm > 0.0 else new_c
        else:
            clusters.append([i])
            centroids.append(vec.copy())

    return clusters


async def find_hub_clusters(
    db: sqlite3.Connection,
    *,
    min_sessions: int = 2,
    max_chunks: int = 150,
    cluster_threshold: float = 0.88,
) -> list[dict[str, Any]]:
    """Detect recurring content clusters across successful sessions.

    Queries chunks from sessions with at least one ``outcome='ok'`` trajectory,
    caps the pool at *max_chunks* (newest first) for Pi budget, then loads
    their pre-stored embeddings from ``chunks_vec`` and clusters by cosine
    similarity. Returns clusters that span ≥ *min_sessions* distinct sessions —
    these are the episodic "hubs" worth distilling into durable semantic facts.

    Uses stored embeddings (no extra API call); chunks without a corresponding
    row in ``chunks_vec`` are silently skipped.

    Each hub dict contains:
      ``sessions``: sorted list of session_id ints
      ``samples``:  up to 3 representative chunk texts (≤500 chars each)
    """
    rows = db.execute(
        """
        SELECT c.id, c.session_id, c.content
        FROM chunks c
        WHERE c.session_id IN (
            SELECT DISTINCT session_id FROM trajectories
            WHERE outcome = 'ok' AND session_id IS NOT NULL
        )
        ORDER BY c.created_at DESC
        LIMIT ?
        """,
        (max_chunks,),
    ).fetchall()

    if len(rows) < 2:
        return []

    chunk_ids = [r[0] for r in rows]
    session_ids_list = [r[1] for r in rows]
    contents = [r[2] for r in rows]

    placeholders = ",".join("?" * len(chunk_ids))
    emb_rows = db.execute(
        f"SELECT chunk_id, embedding FROM chunks_vec "
        f"WHERE chunk_id IN ({placeholders})",
        chunk_ids,
    ).fetchall()
    emb_by_id = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in emb_rows}

    # Align to rows that have embeddings in chunks_vec
    aligned = [
        (cid, sid, txt)
        for cid, sid, txt in zip(chunk_ids, session_ids_list, contents)
        if cid in emb_by_id
    ]
    if len(aligned) < 2:
        return []

    vecs = [emb_by_id[cid] for cid, _, _ in aligned]
    clusters = greedy_cluster(vecs, cluster_threshold)

    hubs: list[dict[str, Any]] = []
    for cl in clusters:
        if len(cl) < 2:
            continue
        cl_sessions = {aligned[i][1] for i in cl}
        if len(cl_sessions) < min_sessions:
            continue
        samples = [aligned[i][2][:500] for i in cl[:3]]
        hubs.append({"sessions": sorted(cl_sessions), "samples": samples})

    return hubs


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
