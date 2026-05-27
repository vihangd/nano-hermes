"""Curator — idle-triggered skill maintenance (Phase 8).

Borrowed from hermes-agent's curator: a low-LLM-cost pass that archives
stale-but-not-failing skills so the corpus stays small and focused.

Rules (deliberately conservative):
- Only touches ACTIVE skills (drafts and deprecated are out of scope).
- Skill must have been EXERCISED — use_count >= curator_min_uses. Untested
  skills are left alone so the rewriter / propose_skill flow can deal with
  them.
- Skill must be DORMANT — last_used_at older than curator_stale_after_days.
- Action is always ARCHIVE (status → 'deprecated'); never deletes the row,
  never touches SKILL.md on disk. A skill_versions row is written for the
  audit trail with reason='curator: stale'.
- Runs at session-start with a cooldown so it doesn't fire on every launch.

No external LLM call — pure SQL.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_META_LAST_RUN = "curator.last_run_at"


def meta_get(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row[0] if row else None


def meta_set(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    db.commit()


@dataclass
class StaleSkill:
    name: str
    last_used_at: float
    use_count: int


def find_stale_skills(
    db: sqlite3.Connection,
    *,
    stale_after_days: int,
    min_uses: int,
    now: float | None = None,
) -> list[StaleSkill]:
    """Return active skills that look dormant by the curator's rules."""
    if stale_after_days <= 0:
        return []
    now_ts = now if now is not None else time.time()
    cutoff = now_ts - stale_after_days * 86400
    rows = db.execute(
        """
        SELECT name, last_used_at, use_count
        FROM skill_stats
        WHERE status = 'active'
          AND use_count >= ?
          AND last_used_at IS NOT NULL
          AND last_used_at < ?
        ORDER BY last_used_at ASC
        """,
        (min_uses, cutoff),
    ).fetchall()
    return [StaleSkill(name=r[0], last_used_at=r[1], use_count=r[2]) for r in rows]


def archive_skill(
    db: sqlite3.Connection, name: str, *, current_body: str | None = None
) -> None:
    """Move a skill to status='deprecated' and record an audit row.

    *current_body* is the SKILL.md snapshot the caller already loaded;
    None when unavailable (e.g. file moved). The skill_versions row is
    written regardless so the action is always traceable.
    """
    db.execute(
        "UPDATE skill_stats SET status = 'deprecated' WHERE name = ?",
        (name,),
    )
    db.execute(
        "INSERT INTO skill_versions (skill_name, body, reason, created_at) "
        "VALUES (?, ?, ?, ?)",
        (name, current_body or "", "curator: stale", time.time()),
    )
    db.commit()


def should_run(
    db: sqlite3.Connection, cooldown_hours: int, *, now: float | None = None
) -> bool:
    """Return True if the cooldown has elapsed since the last curator pass."""
    if cooldown_hours <= 0:
        return True
    now_ts = now if now is not None else time.time()
    raw = meta_get(db, _META_LAST_RUN)
    if raw is None:
        return True
    try:
        last = float(raw)
    except ValueError:
        return True
    return (now_ts - last) >= cooldown_hours * 3600


def mark_run(db: sqlite3.Connection, *, now: float | None = None) -> None:
    meta_set(db, _META_LAST_RUN, str(now if now is not None else time.time()))


def run_curator(hook: "NanoHermesHook") -> list[str]:
    """Synchronous curator pass. Returns the names of skills it archived.

    Designed to be safe to call from a background thread (no async / LLM).
    Skips silently when the cooldown has not elapsed.
    """
    cfg = hook.config.skill_stats
    if not getattr(cfg, "curator_enabled", True):
        return []
    if not should_run(hook.db, cfg.curator_cooldown_hours):
        log.debug("curator: cooldown — skipping")
        return []
    stale = find_stale_skills(
        hook.db,
        stale_after_days=cfg.curator_stale_after_days,
        min_uses=cfg.curator_min_uses,
    )
    archived: list[str] = []
    for skill in stale:
        body = None
        skill_path = hook.workspace / "skills" / skill.name / "SKILL.md"
        if skill_path.exists():
            try:
                body = skill_path.read_text()
            except OSError:
                body = None
        archive_skill(hook.db, skill.name, current_body=body)
        archived.append(skill.name)
        log.info(
            "curator: archived %s (last used %.0f days ago, %d uses)",
            skill.name,
            (time.time() - skill.last_used_at) / 86400,
            skill.use_count,
        )
    mark_run(hook.db)
    return archived
