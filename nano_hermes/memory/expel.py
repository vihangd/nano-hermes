"""ExpeL contrastive insight extraction (arXiv:2308.10144, AAAI 2024).

After each completed session, find the most similar past session with the
OPPOSITE outcome class (success ↔ fail/partial). When similarity exceeds
the configured threshold, one LLM call extracts a contrastive insight
explaining what drove the difference. The insight is stored in
``semantic_facts`` with ``fact_type='expel'`` and is retrieved alongside
Dynamic Cheatsheet lessons at inference.

Default off (``expel_enabled=False``). One conditional LLM call per session
boundary — skipped entirely when no contrasting session is found above
threshold.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING, Any

from ..utils.error_classifier import classify_llm_response

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_EXPEL_PROMPT = """\
Two similar tasks had DIFFERENT outcomes. Extract ONE concise insight \
(max 2 sentences) that explains the key difference and would help handle \
this type of task better in the future.

TASK A (outcome: {outcome_a}):
{task_a}
Key events: {events_a}

TASK B (outcome: {outcome_b}):
{task_b}
Key events: {events_b}

Output ONLY the insight text. \
If no meaningful insight can be extracted, output exactly: SKIP"""


def _outcome_class(outcome: str) -> str:
    """Map raw outcome to 'success' or 'failure' for contrast pairing."""
    return "success" if outcome == "success" else "failure"


def _opposite_outcomes(outcome_class: str) -> tuple[str, ...]:
    if outcome_class == "success":
        return ("fail", "partial")
    return ("success",)


def _chunk_text(db: sqlite3.Connection, session_id: int, limit: int = 400) -> str:
    """Return concatenated content of the first few chunks for a session."""
    rows = db.execute(
        "SELECT content FROM chunks WHERE session_id = ? "
        "ORDER BY turn_index ASC LIMIT 5",
        (session_id,),
    ).fetchall()
    return " ".join(r[0] for r in rows)[:limit]


def find_contrasting_session(
    db: sqlite3.Connection,
    session_id: int,
    outcome: str,
    *,
    threshold: float = 0.75,
    lookback_days: int = 90,
) -> tuple[int, str] | None:
    """Find the most similar session with the opposite outcome class.

    Returns ``(contrasting_session_id, contrasting_outcome)`` or ``None``.
    Uses embedding cosine similarity if available; falls back to None
    (no contrast) when embeddings are absent.
    """
    opp_outcomes = _opposite_outcomes(_outcome_class(outcome))
    placeholders = ",".join("?" * len(opp_outcomes))
    since = time.time() - lookback_days * 86400

    # Get current session's first-chunk embedding.
    # chunks_vec is a sqlite_vec virtual table; use scalar subquery for rowid lookup.
    cur_row = db.execute(
        "SELECT embedding FROM chunks_vec "
        "WHERE chunk_id = ("
        "  SELECT c.id FROM chunks c WHERE c.session_id = ? "
        "  ORDER BY c.turn_index ASC LIMIT 1"
        ")",
        (session_id,),
    ).fetchone()
    if cur_row is None or cur_row[0] is None:
        return None

    cur_emb = cur_row[0]

    # Find past sessions with opposite outcome that have embeddings.
    candidates = db.execute(
        f"SELECT t.session_id, t.outcome FROM trajectories t "
        f"WHERE t.outcome IN ({placeholders}) "
        f"  AND t.session_id != ? "
        f"  AND t.created_at >= ? "
        f"ORDER BY t.created_at DESC LIMIT 50",
        (*opp_outcomes, session_id, since),
    ).fetchall()

    if not candidates:
        return None

    best_sid: int | None = None
    best_outcome: str | None = None
    best_sim: float = -1.0

    try:
        import numpy as np  # noqa: PLC0415

        cur_vec = np.frombuffer(cur_emb, dtype=np.float32)
        norm_cur = float(np.linalg.norm(cur_vec))
        if norm_cur < 1e-9:
            return None

        for cand_sid, cand_outcome in candidates:
            emb_row = db.execute(
                "SELECT embedding FROM chunks_vec "
                "WHERE rowid = ("
                "  SELECT c.id FROM chunks c WHERE c.session_id = ? "
                "  ORDER BY c.turn_index ASC LIMIT 1"
                ")",
                (cand_sid,),
            ).fetchone()
            if emb_row is None or emb_row[0] is None:
                continue
            cand_vec = np.frombuffer(emb_row[0], dtype=np.float32)
            norm_c = float(np.linalg.norm(cand_vec))
            if norm_c < 1e-9:
                continue
            sim = float(np.dot(cur_vec, cand_vec) / (norm_cur * norm_c))
            if sim > best_sim:
                best_sim = sim
                best_sid = cand_sid
                best_outcome = cand_outcome
    except Exception:
        log.debug("expel: cosine similarity failed", exc_info=True)
        return None

    if best_sim < threshold or best_sid is None:
        return None
    return (best_sid, best_outcome)


def _store_insight(
    db: sqlite3.Connection, insight: str, task_category: str
) -> int:
    """Insert contrastive insight into semantic_facts. Returns row id."""
    cur = db.execute(
        "INSERT INTO semantic_facts "
        "(content, source_chunk_ids, keywords, tags, context, importance, "
        " fact_type, task_category, created_at) "
        "VALUES (?, '[]', '[]', '[]', ?, 6, 'expel', ?, ?)",
        (insight, task_category, task_category, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


async def extract_contrastive_insight(
    hook: "NanoHermesHook",
    session_id: int,
    outcome: str,
    messages: list[dict[str, Any]],
) -> str | None:
    """Find a contrasting session and extract an insight if above threshold.

    Returns the insight text on success, None otherwise.
    Stores the insight in semantic_facts with fact_type='expel'.
    """
    cfg = hook.config
    threshold: float = getattr(cfg, "expel_similarity_threshold", 0.75)

    contrast = find_contrasting_session(
        hook.db, session_id, outcome, threshold=threshold
    )
    if contrast is None:
        return None

    contrast_sid, contrast_outcome = contrast

    # Gather task text and key events for both sessions.
    task_a = _chunk_text(hook.db, session_id)
    events_a = task_a  # reuse first-chunk text as proxy for events
    task_b = _chunk_text(hook.db, contrast_sid)
    events_b = task_b

    if not task_a or not task_b:
        return None

    prompt = _EXPEL_PROMPT.format(
        outcome_a=outcome,
        task_a=task_a[:400],
        events_a=events_a[:400],
        outcome_b=contrast_outcome,
        task_b=task_b[:400],
        events_b=events_b[:400],
    )

    try:
        resp = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(hook._loop, "model", None),
            max_tokens=150,
        )
    except Exception:
        log.debug("expel: LLM call failed", exc_info=True)
        return None

    err = classify_llm_response(resp)
    if err is not None:
        log.debug("expel: LLM error %s — skipping", err.reason.value)
        return None

    insight = (resp.content or "").strip()
    if not insight or insight.upper().startswith("SKIP") or len(insight) < 20:
        return None

    # Use the current session's first chunk as task category.
    first_chunk = _chunk_text(hook.db, session_id, limit=120)
    fact_id = _store_insight(hook.db, insight, first_chunk)
    log.info(
        "expel: stored contrastive insight (session %d ↔ %d, sim≥%.2f, fact=%d)",
        session_id,
        contrast_sid,
        threshold,
        fact_id,
    )
    return insight
