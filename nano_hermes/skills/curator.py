"""Curator — idle-triggered skill maintenance (Phase 8).

Borrowed from hermes-agent's curator: a low-LLM-cost pass that archives
stale-but-not-failing skills so the corpus stays small and focused.

Two-phase lifecycle (deliberately conservative):
- active -> stale: an EXERCISED (use_count >= curator_min_uses) but DORMANT
  (last_used_at older than curator_stale_after_days) skill is demoted to
  'stale'. Stale skills stay searchable; the agent using one again reactivates
  it to 'active' (see coordinator.skills.check_promotions).
- stale -> deprecated: a skill that has stayed dormant past
  curator_archive_after_days is archived (filtered out of search).
- Only touches agent-authored, unpinned skills (origin='agent', pinned=0).
  Never deletes the row, never touches SKILL.md on disk; each transition
  writes a skill_versions audit row.
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


def _find_dormant(
    db: sqlite3.Connection,
    *,
    from_status: str,
    dormant_after_days: int,
    min_uses: int,
    now: float,
) -> list[StaleSkill]:
    """Agent-authored, unpinned skills in *from_status* dormant past the cutoff."""
    if dormant_after_days <= 0:
        return []
    cutoff = now - dormant_after_days * 86400
    rows = db.execute(
        """
        SELECT name, last_used_at, use_count
        FROM skill_stats
        WHERE status = ?
          AND origin = 'agent'
          AND pinned = 0
          AND use_count >= ?
          AND last_used_at IS NOT NULL
          AND last_used_at < ?
        ORDER BY last_used_at ASC
        """,
        (from_status, min_uses, cutoff),
    ).fetchall()
    return [StaleSkill(name=r[0], last_used_at=r[1], use_count=r[2]) for r in rows]


def find_stale_skills(
    db: sqlite3.Connection,
    *,
    stale_after_days: int,
    min_uses: int,
    now: float | None = None,
) -> list[StaleSkill]:
    """Active skills dormant long enough to demote to 'stale'."""
    now_ts = now if now is not None else time.time()
    return _find_dormant(
        db,
        from_status="active",
        dormant_after_days=stale_after_days,
        min_uses=min_uses,
        now=now_ts,
    )


def find_deprecatable_skills(
    db: sqlite3.Connection,
    *,
    archive_after_days: int,
    now: float | None = None,
) -> list[StaleSkill]:
    """Already-stale skills dormant long enough to deprecate (archive)."""
    now_ts = now if now is not None else time.time()
    # min_uses=0: a stale skill was already exercised before it was staled.
    return _find_dormant(
        db,
        from_status="stale",
        dormant_after_days=archive_after_days,
        min_uses=0,
        now=now_ts,
    )


def transition_skill(
    db: sqlite3.Connection,
    name: str,
    *,
    new_status: str,
    reason: str,
    current_body: str | None = None,
) -> None:
    """Set a skill's status and record a skill_versions audit row.

    *current_body* is the SKILL.md snapshot the caller already loaded; None
    when unavailable (e.g. file moved). The audit row is written regardless.
    """
    db.execute(
        "UPDATE skill_stats SET status = ? WHERE name = ?",
        (new_status, name),
    )
    db.execute(
        "INSERT INTO skill_versions (skill_name, body, reason, created_at) "
        "VALUES (?, ?, ?, ?)",
        (name, current_body or "", reason, time.time()),
    )
    db.commit()


def mark_stale(
    db: sqlite3.Connection, name: str, *, current_body: str | None = None
) -> None:
    """active -> stale."""
    transition_skill(
        db, name, new_status="stale", reason="curator: stale", current_body=current_body
    )


def archive_skill(
    db: sqlite3.Connection, name: str, *, current_body: str | None = None
) -> None:
    """stale -> deprecated (archived)."""
    transition_skill(
        db,
        name,
        new_status="deprecated",
        reason="curator: archived",
        current_body=current_body,
    )


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


def _load_body(hook: "NanoHermesHook", name: str) -> str | None:
    skill_path = hook.workspace / "skills" / name / "SKILL.md"
    if skill_path.exists():
        try:
            return skill_path.read_text()
        except OSError:
            return None
    return None


def run_curator(hook: "NanoHermesHook") -> list[str]:
    """Synchronous two-phase curator pass. Returns names it transitioned
    (active->stale and stale->deprecated).

    Designed to be safe to call from a background thread (no async / LLM).
    Skips silently when the cooldown has not elapsed.
    """
    cfg = hook.config.skill_stats
    if not getattr(cfg, "curator_enabled", True):
        return []
    if not should_run(hook.db, cfg.curator_cooldown_hours):
        log.debug("curator: cooldown — skipping")
        return []

    touched: list[str] = []
    now = time.time()

    # Phase 2 first: deprecate skills already dormant past the archive window,
    # before this pass would re-stale freshly-active ones. (Order is moot —
    # the two passes select disjoint statuses — but reads cleaner this way.)
    for skill in find_deprecatable_skills(
        hook.db, archive_after_days=cfg.curator_archive_after_days, now=now
    ):
        archive_skill(hook.db, skill.name, current_body=_load_body(hook, skill.name))
        touched.append(skill.name)
        log.info(
            "curator: archived %s (stale -> deprecated, last used %.0f days ago)",
            skill.name,
            (now - skill.last_used_at) / 86400,
        )

    # Phase 1: demote dormant active skills to stale.
    for skill in find_stale_skills(
        hook.db, stale_after_days=cfg.curator_stale_after_days, min_uses=cfg.curator_min_uses, now=now
    ):
        mark_stale(hook.db, skill.name, current_body=_load_body(hook, skill.name))
        touched.append(skill.name)
        log.info(
            "curator: staled %s (active -> stale, last used %.0f days ago, %d uses)",
            skill.name,
            (now - skill.last_used_at) / 86400,
            skill.use_count,
        )

    mark_run(hook.db)
    return touched
