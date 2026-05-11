"""Skill co-occurrence tracking.

Records which skills are frequently used together within a session, and
exposes the top co-used skills for display in skill_search results.
"""
from __future__ import annotations

import itertools
import logging
import sqlite3
import time

log = logging.getLogger(__name__)


def record_composition(db: sqlite3.Connection, skills: set[str] | frozenset[str]) -> None:
    """Insert or update pairwise co-occurrence counts for *skills*.

    Pairs are normalised (alphabetical order) so (a, b) and (b, a) map
    to the same row.  No-op if fewer than 2 skills.
    """
    names = sorted(skills)
    if len(names) < 2:
        return
    now = time.time()
    try:
        with db:
            for a, b in itertools.combinations(names, 2):
                db.execute(
                    """
                    INSERT INTO skill_compositions (skill_a, skill_b, count, last_used)
                    VALUES (?, ?, 1, ?)
                    ON CONFLICT(skill_a, skill_b) DO UPDATE SET
                        count = count + 1,
                        last_used = excluded.last_used
                    """,
                    (a, b, now),
                )
    except Exception:
        log.exception("skill composition record failed")


def get_compositions(
    db: sqlite3.Connection,
    skill_name: str,
    *,
    limit: int = 3,
    min_count: int = 2,
) -> list[str]:
    """Return skill names that co-occur most with *skill_name*.

    Only returns pairs with co-occurrence >= *min_count* to filter noise
    from single accidental co-uses.
    """
    rows = db.execute(
        """
        SELECT CASE WHEN skill_a = ? THEN skill_b ELSE skill_a END AS partner, count
        FROM skill_compositions
        WHERE (skill_a = ? OR skill_b = ?) AND count >= ?
        ORDER BY count DESC
        LIMIT ?
        """,
        (skill_name, skill_name, skill_name, min_count, limit),
    ).fetchall()
    return [r[0] for r in rows]
