"""Ratchet skill-cap + retirement (arXiv:2605.22148).

Two complementary guardrails that prevent skill-bank divergence:

1. **Contribution-score retirement** — skills whose contribution score
   ĉ(s) = (2·success_count − use_count) / use_count falls at or below
   −retire_threshold are set to ``deprecated``.  Requires a minimum
   evidence floor (``n_min`` uses) before a skill is eligible.

2. **Cap enforcement** — if the active agent-origin non-pinned skill count
   exceeds ``skill_cap``, the lowest-scoring skills are deprecated until
   the count is within the cap.

Both actions use ``status='deprecated'`` — soft delete only; rows and
version history are preserved for forensics and possible reinstatement.
Pinned skills are always exempt from both rules.

Called from ``_run_evolution_cycle`` after the umbrella merge step.
Default-off (``ratchet_enabled = False``); zero LLM calls.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


def _contribution_score(use_count: int, success_count: int) -> float:
    """ĉ(s) = (2·successes − uses) / uses  ∈ [−1, 1]."""
    if use_count <= 0:
        return 0.0
    return (2 * success_count - use_count) / use_count


def run_ratchet(hook: "NanoHermesHook") -> dict[str, list[str]]:
    """Run contribution-score retirement + cap enforcement.

    Returns ``{"retired": [...], "cap_evicted": [...]}`` listing skill
    names that were deprecated by each rule this cycle.
    """
    cfg = hook.config.skill_stats
    if not getattr(cfg, "ratchet_enabled", False):
        return {"retired": [], "cap_evicted": []}

    n_min: int = getattr(cfg, "ratchet_n_min", 100)
    tau: float = getattr(cfg, "ratchet_retire_threshold", 0.10)
    cap: int = getattr(cfg, "ratchet_skill_cap", 50)
    db: sqlite3.Connection = hook.db

    # ------------------------------------------------------------------ #
    # Step 1: contribution-score retirement                                #
    # ------------------------------------------------------------------ #
    retired: list[str] = []
    rows = db.execute(
        "SELECT name, use_count, success_count FROM skill_stats "
        "WHERE status = 'active' AND origin = 'agent' AND pinned = 0 "
        "  AND use_count >= ?",
        (n_min,),
    ).fetchall()
    for name, uses, successes in rows:
        if _contribution_score(uses, successes) <= -tau:
            db.execute(
                "UPDATE skill_stats SET status = 'deprecated' WHERE name = ?",
                (name,),
            )
            retired.append(name)
            log.info(
                "ratchet: retiring %s (ĉ=%.3f, uses=%d)",
                name,
                _contribution_score(uses, successes),
                uses,
            )
    if retired:
        db.commit()

    # ------------------------------------------------------------------ #
    # Step 2: cap enforcement — evict lowest-ĉ non-pinned agent skills    #
    # ------------------------------------------------------------------ #
    cap_evicted: list[str] = []
    total_active: int = db.execute(
        "SELECT COUNT(*) FROM skill_stats "
        "WHERE status = 'active' AND origin = 'agent' AND pinned = 0"
    ).fetchone()[0]
    to_evict = max(0, total_active - cap)
    if to_evict > 0:
        # Lowest ĉ first (NULLIF guards divide-by-zero for use_count=0 rows).
        evict_candidates = db.execute(
            "SELECT name, use_count, success_count FROM skill_stats "
            "WHERE status = 'active' AND origin = 'agent' AND pinned = 0 "
            "ORDER BY CAST(2 * success_count - use_count AS REAL)"
            "       / NULLIF(use_count, 0) ASC",
        ).fetchall()
        for name, uses, successes in evict_candidates:
            if len(cap_evicted) >= to_evict:
                break
            db.execute(
                "UPDATE skill_stats SET status = 'deprecated' WHERE name = ?",
                (name,),
            )
            cap_evicted.append(name)
            log.info(
                "ratchet: cap-evicting %s (ĉ=%.3f, cap=%d)",
                name,
                _contribution_score(uses, successes),
                cap,
            )
        if cap_evicted:
            db.commit()

    return {"retired": retired, "cap_evicted": cap_evicted}
