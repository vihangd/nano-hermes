"""A-MEM (Phase 6) — Zettelkasten linking on semantic_facts.

When a new fact is distilled we embed it, store the vector in
``semantic_facts_vec``, then find the top-k most similar prior facts and
record edges in ``semantic_fact_links`` (similarity ≥ threshold). The
graph is undirected and stored canonically with ``fact_a_id < fact_b_id``
so each pair has exactly one row.

No LLM call here — embeddings only. The fact's annotations
(keywords/tags/context/importance) come from the distillation prompt
upstream; this module just adds the geometric edge.
"""
from __future__ import annotations

import sqlite3
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_DEFAULT_TOP_K = 3
_DEFAULT_SIM_THRESHOLD = 0.78


async def link_new_fact(
    hook: "NanoHermesHook",
    fact_id: int,
    fact_text: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
    sim_threshold: float = _DEFAULT_SIM_THRESHOLD,
) -> int:
    """Embed ``fact_text``, store in ``semantic_facts_vec``, link to similar facts.

    Returns the number of edges written. Failures (embedding provider down,
    no prior embedded facts) yield 0 — never raise out of distillation.
    """
    async with hook.embedder() as chain:
        [vec] = await chain.embed([fact_text])
    if vec is None:
        return 0
    vec_bytes = np.asarray(vec, dtype=np.float32).tobytes()

    db = hook.db
    db.execute(
        "INSERT OR REPLACE INTO semantic_facts_vec (fact_id, embedding) VALUES (?, ?)",
        (fact_id, vec_bytes),
    )
    db.commit()

    # KNN over prior facts. vec0's MATCH returns rows ordered by distance
    # (lower = closer); cosine distance = 1 - cosine_similarity.
    rows = db.execute(
        "SELECT fact_id, distance FROM semantic_facts_vec "
        "WHERE embedding MATCH ? AND fact_id != ? AND k = ? "
        "ORDER BY distance",
        (vec_bytes, fact_id, top_k),
    ).fetchall()

    now = time.time()
    written = 0
    for other_id, distance in rows:
        similarity = 1.0 - float(distance)
        if similarity < sim_threshold:
            continue
        a, b = (fact_id, other_id) if fact_id < other_id else (other_id, fact_id)
        db.execute(
            "INSERT OR IGNORE INTO semantic_fact_links "
            "(fact_a_id, fact_b_id, similarity, created_at) "
            "VALUES (?, ?, ?, ?)",
            (a, b, similarity, now),
        )
        written += 1
    db.commit()
    return written


def neighbours_of(
    db: sqlite3.Connection, fact_id: int, *, limit: int = 5
) -> list[tuple[int, float]]:
    """Return up to *limit* linked fact IDs ordered by similarity desc.

    Useful for traversal during retrieval — not used by linking itself.
    """
    rows = db.execute(
        """
        SELECT CASE WHEN fact_a_id = ? THEN fact_b_id ELSE fact_a_id END AS other,
               similarity
        FROM semantic_fact_links
        WHERE fact_a_id = ? OR fact_b_id = ?
        ORDER BY similarity DESC
        LIMIT ?
        """,
        (fact_id, fact_id, fact_id, limit),
    ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]
