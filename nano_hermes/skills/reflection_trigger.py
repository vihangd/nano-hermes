"""Skill-quality reflection triggers (OPRO-inspired).

When a skill has a mixed success rate (0.3 ≤ rate ≤ 0.8) after enough
uses, we inject a one-time suggestion for the agent to reflect on it.
Very high (>0.8) or very low (<0.3) success rates carry clear signals:
the skill is working or it should be deprecated; reflection adds little.
The mid-range (30–80%) is where targeted reflection has the most value.

Each trigger fires only once per skill per threshold crossing — the
coordinator tracks which skills have already been suggested so the same
skill doesn't spam every session.
"""
from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..config import SkillStatsConfig

_LOW_BOUND = 0.3
_HIGH_BOUND = 0.8


def check_skill_reflection_triggers(
    db: sqlite3.Connection,
    skill_names: list[str],
    cfg: "SkillStatsConfig",
    already_triggered: set[str],
) -> list[str]:
    """Return suggestion strings for skills that warrant reflection.

    *already_triggered* is mutated in-place to track which skills have
    been suggested so the caller can persist the set across calls.

    Rules:
    - use_count >= cfg.min_uses_for_success_rate (reuses existing config)
    - _LOW_BOUND <= success_rate <= _HIGH_BOUND
    - skill not already in *already_triggered*
    """
    suggestions: list[str] = []
    for name in skill_names:
        if name in already_triggered:
            continue
        row = db.execute(
            "SELECT use_count, success_count, status FROM skill_stats WHERE name = ?",
            (name,),
        ).fetchone()
        if not row:
            continue
        use_count, success_count, status = int(row[0]), int(row[1]), row[2]
        if status == "deprecated":
            continue
        if use_count < cfg.min_uses_for_success_rate:
            continue
        rate = success_count / use_count
        if _LOW_BOUND <= rate <= _HIGH_BOUND:
            pct = int(rate * 100)
            suggestions.append(
                f"Consider reflecting on the '{name}' skill — used {use_count} time(s) "
                f"with {pct}% success. A brief reflect() call noting what works and "
                f"what doesn't could improve future outcomes."
            )
            already_triggered.add(name)
    return suggestions
