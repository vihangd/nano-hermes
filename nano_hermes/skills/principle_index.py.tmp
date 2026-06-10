"""Embedding index + dedup for the principles store (ACE delta-playbook).

The principles table is nano-hermes's playbook: condition/action rules injected
each turn by FTS match. This module adds the ACE pieces — an embedding per
principle (``principles_vec``) so near-duplicates can be detected and merged
instead of piling up. Used by both the manual ``record_principle`` tool and the
automatic principle curator, so the two write paths can't duplicate each other.

Embedding is best-effort: if every provider is down we still store the
principle (FTS-only, no dedup) rather than dropping it.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from ..embedding.chain import AllProvidersFailed

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


def principle_text(condition: str, action: str) -> str:
    """Canonical text embedded for a principle (what we dedup on)."""
    return f"{condition} {action}".strip()


def _store_vec(db, principle_id: int, vec: np.ndarray) -> None:
    # vec0 has no UPSERT — delete any existing row first, then insert.
    db.execute("DELETE FROM principles_vec WHERE principle_id = ?", (principle_id,))
    db.execute(
        "INSERT INTO principles_vec (principle_id, embedding) VALUES (?, ?)",
        (principle_id, vec.astype(np.float32).tobytes()),
    )


def find_duplicate(
    db, vec: np.ndarray, threshold: float, *, exclude_id: int | None = None
) -> int | None:
    """Return the id of the nearest principle whose cosine sim >= *threshold*,
    or None. Tables use distance_metric=cosine, so sim = 1 - distance."""
    rows = db.execute(
        "SELECT principle_id, distance FROM principles_vec "
        "WHERE embedding MATCH ? AND k = 3 ORDER BY distance",
        (vec.astype(np.float32).tobytes(),),
    ).fetchall()
    for pid, dist in rows:
        if exclude_id is not None and int(pid) == exclude_id:
            continue
        return int(pid) if (1.0 - dist) >= threshold else None
    return None


async def upsert_principle(
    hook: "NanoHermesHook",
    *,
    condition: str,
    action: str,
    expected_outcome: str | None,
    origin: str,
    dedup_threshold: float,
    protect_manual: bool = False,
) -> tuple[int, str]:
    """Insert a principle (+ FTS + vec) or merge it into a near-duplicate.

    Returns ``(principle_id, outcome)`` where outcome is one of
    ``"inserted"`` | ``"merged"`` | ``"inserted_no_vec"``. The deterministic
    by-id merge (never an LLM rewrite of the table) is the ACE anti-collapse
    invariant; embedding dedup keeps the playbook from accumulating duplicates.
    """
    db = hook.db
    now = time.time()
    vec: np.ndarray | None = None
    try:
        async with hook.embedder() as chain:
            [vec] = await chain.embed([principle_text(condition, action)])
    except AllProvidersFailed:
        vec = None  # FTS-only fallback — no dedup possible

    if vec is not None:
        dup = find_duplicate(db, vec, dedup_threshold)
        if dup is not None:
            # Don't let the auto-curator overwrite a manually-authored or
            # pinned principle it happened to land near — leave it untouched.
            if protect_manual:
                guard = db.execute(
                    "SELECT origin, pinned FROM principles WHERE id = ?", (dup,)
                ).fetchone()
                if guard and (guard[0] == "agent" or guard[1]):
                    return dup, "skipped"
            # Refresh the surviving row's guidance + embedding in place.
            db.execute(
                "UPDATE principles SET action = ?, expected_outcome = ?, updated_at = ? "
                "WHERE id = ?",
                (action, expected_outcome or None, now, dup),
            )
            _store_vec(db, dup, vec)
            db.commit()
            return dup, "merged"

    cur = db.execute(
        "INSERT INTO principles "
        "(condition, action, expected_outcome, confidence, created_at, updated_at, origin) "
        "VALUES (?, ?, ?, 0.5, ?, ?, ?)",
        (condition, action, expected_outcome or None, now, now, origin),
    )
    pid = int(cur.lastrowid)
    db.execute(
        "INSERT INTO principles_fts (rowid, condition, action, content_id) "
        "VALUES (?, ?, ?, ?)",
        (pid, condition, action, pid),
    )
    if vec is not None:
        _store_vec(db, pid, vec)
    db.commit()
    return pid, ("inserted" if vec is not None else "inserted_no_vec")
