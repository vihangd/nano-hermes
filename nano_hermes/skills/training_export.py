"""Offline GEPA/MIPROv2 training data export.

Exports (skill_text, session_context, outcome) tuples for skills with
sufficient session history so that an off-device DSPy/GEPA optimiser can
improve the skill text.  The agent calls ``skill_export`` with a skill name
(or "all") to dump JSONL files that can be copied to a laptop/cloud box for
optimisation and then re-imported via ``propose_skill edit``.

Design notes:
- Corpus maturity is gated on *distinct sessions* (via ``trajectories``),
  NOT ``use_count`` (which is per-rating call, not per-session invocation).
- Each JSONL line is one trajectory: skill_name, skill_text, session_id,
  task, outcome, and up to ``chunk_limit`` chunks from that session.
- Output path: ``<workspace>/nano_hermes/exports/<skill>.<timestamp>.jsonl``
- Import path: ``propose_skill(action="edit", ...)`` — no new infra needed.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def count_skill_sessions(db: sqlite3.Connection, skill_name: str) -> int:
    """Count distinct sessions in which *skill_name* was used.

    Uses ``json_each`` for an exact equality check — avoids LIKE ``_``
    wildcard false-positives for skill names that contain underscores.
    """
    row = db.execute(
        "SELECT COUNT(DISTINCT t.session_id) "
        "FROM trajectories t, json_each(t.skills_used) j "
        "WHERE j.value = ?",
        (skill_name,),
    ).fetchone()
    return int(row[0]) if row else 0


def _get_skill_body_from_workspace(
    workspace: Path, skill_name: str
) -> str | None:
    """Read SKILL.md from workspace skills dir for *skill_name*."""
    skill_dir = workspace / "skills" / skill_name
    skill_file = skill_dir / "SKILL.md"
    if skill_file.exists():
        return skill_file.read_text(encoding="utf-8")
    # Try flat layout: skills/skill_name.md
    flat = workspace / "skills" / f"{skill_name}.md"
    if flat.exists():
        return flat.read_text(encoding="utf-8")
    return None


def export_skill_training_data(
    db: sqlite3.Connection,
    workspace: Path,
    skill_name: str,
    *,
    chunk_limit: int = 5,
) -> list[dict[str, Any]]:
    """Build training rows for *skill_name*.

    Returns a list of dicts (one per trajectory that used the skill).
    Each dict has: skill_name, skill_text, session_id, task, outcome,
    chunks (list of str, capped at *chunk_limit*).
    """
    skill_text = _get_skill_body_from_workspace(workspace, skill_name)
    if skill_text is None:
        skill_text = f"# {skill_name}\n(skill body not found in workspace)"

    rows = db.execute(
        """
        SELECT DISTINCT t.session_id, t.task, t.outcome
        FROM trajectories t, json_each(t.skills_used) j
        WHERE j.value = ?
        ORDER BY t.created_at DESC
        """,
        (skill_name,),
    ).fetchall()

    records: list[dict[str, Any]] = []
    for session_id, task, outcome in rows:
        chunk_rows = db.execute(
            "SELECT content FROM chunks WHERE session_id = ? "
            "ORDER BY turn_index ASC LIMIT ?",
            (session_id, chunk_limit),
        ).fetchall()
        chunks = [r[0] for r in chunk_rows]
        records.append({
            "skill_name": skill_name,
            "skill_text": skill_text,
            "session_id": session_id,
            "task": task or "",
            "outcome": outcome or "unknown",
            "chunks": chunks,
        })
    return records


def export_mature_skills(
    db: sqlite3.Connection,
    workspace: Path,
    exports_dir: Path,
    *,
    min_sessions: int = 50,
    chunk_limit: int = 5,
) -> list[str]:
    """Export all active skills that have ≥ *min_sessions* distinct sessions.

    Writes one JSONL file per skill under *exports_dir*.
    Returns list of file paths written.
    """
    exports_dir.mkdir(parents=True, exist_ok=True)

    active_skills = db.execute(
        "SELECT name FROM skill_stats WHERE status = 'active' ORDER BY name",
    ).fetchall()

    written: list[str] = []
    for (name,) in active_skills:
        n_sessions = count_skill_sessions(db, name)
        if n_sessions < min_sessions:
            log.debug("skill %r has %d sessions < %d — skipping", name, n_sessions, min_sessions)
            continue

        records = export_skill_training_data(
            db, workspace, name, chunk_limit=chunk_limit
        )
        if not records:
            log.debug("skill %r: no trajectory rows — skipping", name)
            continue

        ts = int(time.time())
        out_path = exports_dir / f"{name}.{ts}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        written.append(str(out_path))
        log.info("exported %d rows for skill %r → %s", len(records), name, out_path)

    return written
