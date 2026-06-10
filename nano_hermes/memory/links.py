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

import json
import sqlite3
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_DEFAULT_TOP_K = 3
_DEFAULT_SIM_THRESHOLD = 0.78


def _union_capped(existing: list[str], incoming: list[str], cap: int) -> list[str]:
    """Append novel *incoming* items to *existing* (order-preserving, deduped),
    capped at *cap*. Existing items are never dropped to make room."""
    out = list(existing)
    seen = set(existing)
    for item in incoming:
        if len(out) >= cap:
            break
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _evolve_neighbour(
    db: sqlite3.Connection, new_fact_id: int, neighbour_id: int, max_tags: int
) -> bool:
    """A-MEM: fold the new fact's keywords/tags into a neighbour's, so an older
    fact gains discoverability from a strong new connection. Zero-LLM, capped.
    Returns True if the neighbour row changed."""
    rows = db.execute(
        "SELECT id, keywords, tags FROM semantic_facts WHERE id IN (?, ?)",
        (new_fact_id, neighbour_id),
    ).fetchall()
    by_id = {r[0]: r for r in rows}
    new, nbr = by_id.get(new_fact_id), by_id.get(neighbour_id)
    if new is None or nbr is None:
        return False

    def _load(raw: str | None) -> list[str]:
        try:
            v = json.loads(raw) if raw else []
            return [str(x) for x in v] if isinstance(v, list) else []
        except (json.JSONDecodeError, ValueError):
            return []

    new_kw, new_tags = _load(new[1]), _load(new[2])
    nbr_kw, nbr_tags = _load(nbr[1]), _load(nbr[2])
    merged_kw = _union_capped(nbr_kw, new_kw, max_tags)
    merged_tags = _union_capped(nbr_tags, new_tags, max_tags)
    if merged_kw == nbr_kw and merged_tags == nbr_tags:
        return False
    db.execute(
        "UPDATE semantic_facts SET keywords = ?, tags = ? WHERE id = ?",
        (json.dumps(merged_kw), json.dumps(merged_tags), neighbour_id),
    )
    return True


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

    # Don't link to facts already superseded (bi-temporal invalid_at) — a
    # stale fact shouldn't accrue new edges.
    valid_ids: set[int] = set()
    if rows:
        ph = ",".join("?" * len(rows))
        valid_ids = {
            r[0]
            for r in db.execute(
                f"SELECT id FROM semantic_facts "
                f"WHERE id IN ({ph}) AND invalid_at IS NULL",
                [r[0] for r in rows],
            ).fetchall()
        }

    now = time.time()
    written = 0
    best_neighbour: int | None = None  # strongest edge — rows are distance-sorted
    for other_id, distance in rows:
        if other_id not in valid_ids:
            continue
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
        if best_neighbour is None:
            best_neighbour = other_id
        written += 1

    # A-MEM neighbour evolution: enrich only the single closest neighbour, so
    # an older fact keeps surfacing as the graph grows. Zero-LLM, gated, capped.
    mem_cfg = hook.config.memory
    if best_neighbour is not None and mem_cfg.amem_evolve_neighbours:
        _evolve_neighbour(
            db, fact_id, best_neighbour, mem_cfg.amem_neighbour_max_tags
        )

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
        SELECT other, similarity FROM (
            SELECT CASE WHEN fact_a_id = ? THEN fact_b_id ELSE fact_a_id END AS other,
                   similarity
            FROM semantic_fact_links
            WHERE fact_a_id = ? OR fact_b_id = ?
        )
        JOIN semantic_facts ON semantic_facts.id = other
        WHERE semantic_facts.invalid_at IS NULL
        ORDER BY similarity DESC
        LIMIT ?
        """,
        (fact_id, fact_id, fact_id, limit),
    ).fetchall()
    return [(int(r[0]), float(r[1])) for r in rows]
