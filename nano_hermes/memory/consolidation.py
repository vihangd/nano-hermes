"""Embedding-based memory consolidation (MemGPT/Letta pattern).

Greedy cosine-similarity clustering of memory entries: entries whose
similarity to an existing cluster centroid exceeds *threshold* are merged
into that cluster; the longest entry survives.  Agent-invoked, not
automatic — the agent calls memory_patch(action="consolidate") when it
judges memory is bloated.

Vectors from EmbeddingChain are L2-normalised, so np.dot == cosine sim.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from ..embedding.chain import EmbeddingChain

log = logging.getLogger(__name__)

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
        cl_chunk_ids = sorted(aligned[i][0] for i in cl)
        hubs.append({
            "sessions": sorted(cl_sessions),
            "samples": samples,
            "chunk_ids": cl_chunk_ids,
        })

    return hubs


_DISTILL_PROMPT = """\
Below are {n} representative memory chunks from a recurring theme across {n_sess} sessions.
Distill them into a single durable semantic fact and annotate it for an A-MEM-style
note graph. Reply with ONLY a JSON object, no preamble, no code fence:

{{
  "fact": "<=100 words, the durable lesson learned",
  "keywords": ["3-7 lowercase single-word retrieval keys"],
  "tags": ["1-4 broader category labels"],
  "context": "one-line situational description (when this applies)",
  "importance": <integer 1-10, 1=trivial, 5=routine, 10=foundational>
}}

CHUNKS:
{chunks}
"""


_DEFAULT_IMPORTANCE = 5


async def distill_hub_to_fact(hook: Any, hub: dict) -> dict | None:
    """Distill a hub cluster into a fact + A-MEM annotations via LLM.

    Returns a dict ``{"fact", "keywords", "tags", "context", "importance"}``
    on success, or None on any failure. The LLM is asked for JSON; we
    salvage a plain-text fall-back if the response is non-JSON so a single
    misbehaving model doesn't break distillation entirely.
    """
    import json as _json  # noqa: PLC0415

    provider = getattr(hook._loop, "provider", None)
    if provider is None:
        return None
    model = getattr(hook._loop, "model", None)
    if model is None:
        log.warning("distill_hub_to_fact: no model configured on hook._loop — skipping hub")
        return None
    chunks_text = "\n---\n".join(hub["samples"])
    prompt = _DISTILL_PROMPT.format(
        n=len(hub["samples"]),
        n_sess=len(hub["sessions"]),
        chunks=chunks_text,
    )
    try:
        resp = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=400,
        )
        content = (resp.content or "").strip()
    except Exception:
        return None
    if not content:
        return None
    # Tolerate ```json fences or trailing prose.
    payload = content
    if "```" in payload:
        # Pull text between the first pair of backticks.
        parts = payload.split("```")
        if len(parts) >= 2:
            payload = parts[1]
            if payload.lstrip().lower().startswith("json"):
                payload = payload.split("\n", 1)[1] if "\n" in payload else ""
    try:
        data = _json.loads(payload)
        fact = str(data.get("fact") or "").strip()
        if not fact:
            return None
        return {
            "fact": fact,
            "keywords": [str(x) for x in data.get("keywords") or []][:10],
            "tags": [str(x) for x in data.get("tags") or []][:6],
            "context": str(data.get("context") or "").strip()[:240],
            "importance": _clamp_importance(data.get("importance")),
        }
    except (ValueError, TypeError):
        # Non-JSON response — fall back to using the raw text as the fact.
        return {
            "fact": content,
            "keywords": [],
            "tags": [],
            "context": "",
            "importance": _DEFAULT_IMPORTANCE,
        }


def _clamp_importance(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return _DEFAULT_IMPORTANCE
    return max(1, min(10, n))


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
