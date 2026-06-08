"""Recency decay — a single, dependency-free scoring primitive.

Used by the retrieval paths that rank stored rows (trajectory_search and
global-reflection injection) to gently prefer fresher items, and conceptually
mirrors the age component of fact eviction. Kept tiny and pure so it stays off
any hot DB path and is trivial to test.
"""
from __future__ import annotations


def recency_decay(age_days: float, half_life_days: float) -> float:
    """Exponential recency factor in ``(0, 1]``.

    Returns 1.0 for a brand-new item and halves every ``half_life_days``:
    ``0.5 ** (age_days / half_life_days)``. Negative ages (clock skew) clamp
    to 1.0; a non-positive half-life disables decay (returns 1.0).
    """
    if half_life_days <= 0 or age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)
