"""Write-approval gate + pending store for autonomous evolution writes.

nano-hermes mutates skills and principles *autonomously* and *off the turn
path* (GEPA -> rewriter -> umbrella in ``hook._run_evolution_cycle``; the ACE
curator's ``apply_ops``). Those writes encode the model's own assumptions with
no human in the loop. ``snapshot_before_evolution`` gives a *post-commit* undo;
this module adds a *pre-commit hold*.

Two states, per subsystem (``skill_stats.write_approval`` /
``principles.write_approval``):

* ``"off"`` (default) -- writes commit immediately (today's behaviour).
* ``"approve"`` -- the autonomous caller *stages* its write to the
  ``pending_writes`` table instead of committing; a human approves/rejects via
  the ``pending_review`` tool or the ``nano-hermes pending`` CLI.

Only *autonomous* writes are gated. Foreground ``propose_skill`` / ``principle``
tool calls (the user is present, acting on intent) flow free. Deterministic,
snapshot-reversible maintenance (lifecycle curator archiving, decay/eviction,
A-MEM linking) is *not* gated -- it authors no new content.

The store is a SQLite table (survives restart; one DB per workspace), not
files. A skill payload is the new SKILL.md body; a principle payload is the
curator op list. For skills we also stash ``base_hash`` -- the sha256 of the
*live* SKILL.md at stage time -- so an approve that lands after the skill has
changed underneath is refused (marked ``stale``) rather than silently
clobbering the newer content.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from ..hook import NanoHermesHook

_SUBSYSTEMS = ("skills", "principles")


# --------------------------------------------------------------------------- #
# Gate resolution
# --------------------------------------------------------------------------- #
def gate_mode(hook: "NanoHermesHook", subsystem: str) -> str:
    """Return ``"approve"`` or ``"off"`` for *subsystem* from config.

    Defaults ``"off"`` for any unknown subsystem / missing config so existing
    installs keep committing immediately until the user opts in.
    """
    try:
        if subsystem == "skills":
            return hook.config.skill_stats.write_approval
        if subsystem == "principles":
            return hook.config.principles.write_approval
    except AttributeError:
        return "off"
    return "off"


def is_gated(hook: "NanoHermesHook", subsystem: str) -> bool:
    return gate_mode(hook, subsystem) == "approve"


# --------------------------------------------------------------------------- #
# Hashing + rendering (mirrors ProposeSkillTool._write_skill exactly)
# --------------------------------------------------------------------------- #
def _body_hash(text: str) -> str:
    """sha256 of a SKILL.md's full text. Computed live at stage AND approve."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _skill_path(workspace: Path, skill_name: str) -> Path:
    return workspace / "skills" / skill_name / "SKILL.md"


def _render_skill(skill_name: str, description: str, body: str) -> str:
    # Must match ProposeSkillTool._write_skill (propose_tool.py) byte-for-byte.
    return f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n"


def current_skill_hash(workspace: Path, skill_name: str) -> str | None:
    path = _skill_path(workspace, skill_name)
    if not path.exists():
        return None
    return _body_hash(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Staging
# --------------------------------------------------------------------------- #
def stage_skill_write(
    hook: "NanoHermesHook",
    *,
    skill_name: str,
    description: str,
    body: str,
    reason: str,
    origin: str,
) -> int:
    """Stage an autonomous SKILL.md rewrite. Returns the pending row id.

    ``base_hash`` is the live file hash now, so approve can detect a stale
    replay. The payload carries everything needed to render the new file.
    """
    payload = json.dumps({"description": description, "body": body})
    base_hash = current_skill_hash(hook.workspace, skill_name)
    cur = hook.db.execute(
        "INSERT INTO pending_writes "
        "(subsystem, skill_name, payload, base_hash, reason, origin, status, created_at) "
        "VALUES ('skills', ?, ?, ?, ?, ?, 'pending', ?)",
        (skill_name, payload, base_hash, reason, origin, time.time()),
    )
    hook.db.commit()
    return int(cur.lastrowid)


def stage_umbrella_write(
    hook: "NanoHermesHook",
    *,
    name: str,
    description: str,
    body: str,
    absorbed: list[str],
    reason: str,
) -> int:
    """Stage an umbrella merge: the new umbrella body plus the siblings it
    absorbs. Sibling hashes are captured so approve can refuse if any sibling
    changed since staging (the merge was computed against the old contents).
    """
    sib_hashes = {
        s: current_skill_hash(hook.workspace, s) for s in absorbed
    }
    payload = json.dumps({
        "description": description,
        "body": body,
        "absorbed": absorbed,
        "sibling_hashes": sib_hashes,
    })
    base_hash = current_skill_hash(hook.workspace, name)
    cur = hook.db.execute(
        "INSERT INTO pending_writes "
        "(subsystem, skill_name, payload, base_hash, reason, origin, status, created_at) "
        "VALUES ('skills', ?, ?, ?, ?, 'umbrella', 'pending', ?)",
        (name, payload, base_hash, reason, time.time()),
    )
    hook.db.commit()
    return int(cur.lastrowid)


def stage_principle_ops(
    hook: "NanoHermesHook",
    *,
    ops: list[dict],
    reason: str,
    origin: str = "curator",
) -> int:
    """Stage the curator's deterministic op list for later replay."""
    payload = json.dumps({"ops": ops})
    cur = hook.db.execute(
        "INSERT INTO pending_writes "
        "(subsystem, skill_name, payload, base_hash, reason, origin, status, created_at) "
        "VALUES ('principles', NULL, ?, NULL, ?, ?, 'pending', ?)",
        (payload, reason, origin, time.time()),
    )
    hook.db.commit()
    return int(cur.lastrowid)


# --------------------------------------------------------------------------- #
# Listing
# --------------------------------------------------------------------------- #
def list_pending(
    db: sqlite3.Connection, subsystem: str | None = None
) -> list[dict[str, Any]]:
    """Return open (status='pending') rows, newest first."""
    if subsystem:
        rows = db.execute(
            "SELECT id, subsystem, skill_name, reason, origin, created_at "
            "FROM pending_writes WHERE status='pending' AND subsystem=? "
            "ORDER BY id DESC",
            (subsystem,),
        ).fetchall()
    else:
        rows = db.execute(
            "SELECT id, subsystem, skill_name, reason, origin, created_at "
            "FROM pending_writes WHERE status='pending' ORDER BY id DESC"
        ).fetchall()
    return [
        {
            "id": r[0],
            "subsystem": r[1],
            "skill_name": r[2],
            "reason": r[3],
            "origin": r[4],
            "created_at": r[5],
        }
        for r in rows
    ]


def get_pending(db: sqlite3.Connection, pid: int) -> dict[str, Any] | None:
    r = db.execute(
        "SELECT id, subsystem, skill_name, payload, base_hash, reason, origin, status "
        "FROM pending_writes WHERE id=?",
        (pid,),
    ).fetchone()
    if not r:
        return None
    return {
        "id": r[0],
        "subsystem": r[1],
        "skill_name": r[2],
        "payload": r[3],
        "base_hash": r[4],
        "reason": r[5],
        "origin": r[6],
        "status": r[7],
    }


def diff_pending(db: sqlite3.Connection, workspace: Path, pid: int) -> str:
    """Human-readable view of a pending write: current vs proposed."""
    rec = get_pending(db, pid)
    if not rec:
        return f"no pending write #{pid}"
    if rec["subsystem"] == "principles":
        ops = json.loads(rec["payload"]).get("ops", [])
        lines = [f"#{pid} principles ({rec['origin']}) — {len(ops)} op(s):"]
        for op in ops:
            lines.append(f"  {op.get('op')}: {op.get('condition') or op.get('id') or ''}")
        return "\n".join(lines)
    data = json.loads(rec["payload"])
    cur_path = _skill_path(workspace, rec["skill_name"])
    cur = cur_path.read_text(encoding="utf-8") if cur_path.exists() else "(new skill)"
    proposed = _render_skill(rec["skill_name"], data["description"], data["body"])
    stale = current_skill_hash(workspace, rec["skill_name"]) != rec["base_hash"]
    flag = "  ⚠ STALE — skill changed since staging; approve will be refused\n" if stale else ""
    return (
        f"#{pid} skill '{rec['skill_name']}' ({rec['origin']}) — {rec['reason']}\n{flag}"
        f"--- current ---\n{cur}\n--- proposed ---\n{proposed}"
    )


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #
def reject(db: sqlite3.Connection, pid: int) -> str:
    rec = get_pending(db, pid)
    if not rec or rec["status"] != "pending":
        return f"no open pending write #{pid}"
    db.execute(
        "UPDATE pending_writes SET status='rejected', resolved_at=? WHERE id=?",
        (time.time(), pid),
    )
    db.commit()
    return f"rejected #{pid} ({rec['subsystem']} {rec['skill_name'] or ''})".rstrip()


def _replay_skill_to_disk(
    db: sqlite3.Connection,
    workspace: Path,
    skill_name: str,
    description: str,
    body: str,
    reason: str,
) -> str:
    """Commit a staged skill body to disk. Mirrors ProposeSkillTool._write_skill's
    core (security scan -> snapshot old version -> atomic write -> reset hash) so
    this single sync path is usable both from the async tool and the offline CLI.
    """
    from .._atomic import atomic_write_text  # noqa: PLC0415
    from ..skills.guard import scan_skill_content  # noqa: PLC0415
    from ..skills.rewriter import save_skill_version  # noqa: PLC0415

    err = scan_skill_content(body)
    if err:
        return f"Error: {err}"

    path = _skill_path(workspace, skill_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        save_skill_version(db, skill_name, path.read_text(encoding="utf-8"), reason)
    atomic_write_text(path, _render_skill(skill_name, description, body))
    # No commit here — approve_skill's final db.commit() flushes this hash-reset
    # together with _finish_umbrella's writes and the pending-row status update,
    # so the approve path doesn't leave a half-applied DB on a later failure.
    db.execute(
        "UPDATE skill_stats SET content_hash = NULL, indexed_at = NULL WHERE name = ?",
        (skill_name,),
    )
    return "ok"


def approve_skill(
    db: sqlite3.Connection, workspace: Path, pid: int, *, snapshot_retain: int = 3
) -> str:
    """Approve + replay a staged skill write (sync; offline-capable).

    Refuses (marks the row ``stale``) if the live SKILL.md changed since the
    write was staged — anti-clobber invariant. Takes an evolution snapshot
    before mutating so the approved write is itself one-undo reversible (the
    cycle-time snapshot is suppressed under the gate).
    """
    rec = get_pending(db, pid)
    if not rec or rec["status"] != "pending":
        return f"no open pending write #{pid}"
    if rec["subsystem"] != "skills":
        return f"#{pid} is a {rec['subsystem']} write — use the principle path"

    if current_skill_hash(workspace, rec["skill_name"]) != rec["base_hash"]:
        db.execute(
            "UPDATE pending_writes SET status='stale', resolved_at=? WHERE id=?",
            (time.time(), pid),
        )
        db.commit()
        return (
            f"refused #{pid}: skill '{rec['skill_name']}' changed since staging "
            "(marked stale, not applied). Re-run evolution to regenerate."
        )

    data = json.loads(rec["payload"])

    # Umbrella merges also absorb siblings; refuse if any sibling changed since
    # staging (the merge body was computed against the old sibling contents).
    if rec["origin"] == "umbrella":
        for sib, staged_hash in data.get("sibling_hashes", {}).items():
            if current_skill_hash(workspace, sib) != staged_hash:
                db.execute(
                    "UPDATE pending_writes SET status='stale', resolved_at=? WHERE id=?",
                    (time.time(), pid),
                )
                db.commit()
                return (
                    f"refused #{pid}: sibling '{sib}' changed since staging "
                    "(marked stale, not applied)."
                )

    # Snapshot before mutating — approve-time replay is the real write under the
    # gate, so it (not the suppressed cycle snapshot) carries the undo point.
    try:
        from ..skills.evolution_snapshot import snapshot_evolution  # noqa: PLC0415
        snapshot_evolution(workspace, retain=snapshot_retain)
    except Exception:  # noqa: BLE001 — snapshot is best-effort, never blocks approve
        pass

    result = _replay_skill_to_disk(
        db, workspace, rec["skill_name"], data["description"], data["body"],
        reason=f"approved pending #{pid}: {rec['reason']}",
    )
    if result != "ok":
        return result  # scan failure — leave row pending for inspection

    if rec["origin"] == "umbrella":
        _finish_umbrella(db, workspace, rec["skill_name"], data.get("absorbed", []))

    db.execute(
        "UPDATE pending_writes SET status='approved', resolved_at=? WHERE id=?",
        (time.time(), pid),
    )
    db.commit()
    return f"approved #{pid}: wrote skills/{rec['skill_name']}/SKILL.md"


def _finish_umbrella(
    db: sqlite3.Connection, workspace: Path, name: str, absorbed: list[str]
) -> None:
    """Post-write umbrella steps mirroring run_umbrella_merge: register the
    umbrella skill_stats row and deprecate the absorbed siblings."""
    from ..skills.curator import transition_skill  # noqa: PLC0415

    db.execute(
        "INSERT INTO skill_stats (name, status, origin) VALUES (?, 'active', 'agent') "
        "ON CONFLICT(name) DO UPDATE SET status = 'active', origin = 'agent'",
        (name,),
    )
    for sib in absorbed:
        if sib == name:
            continue
        sib_path = _skill_path(workspace, sib)
        cur_body = sib_path.read_text(encoding="utf-8") if sib_path.exists() else None
        transition_skill(
            db, sib, new_status="deprecated",
            reason=f"absorbed_into: {name}", current_body=cur_body,
        )


async def approve_principles(hook: "NanoHermesHook", pid: int) -> str:
    """Approve + replay staged curator ops (async; needs the hook for embeddings).

    Replays ``apply_ops`` against the *current* table — its deterministic
    dedup/prune re-merge correctly, so no stale-base check is needed (unlike
    skills, the ops are not a full-content overwrite).
    """
    from ..skills.principle_curator import apply_ops  # noqa: PLC0415

    rec = get_pending(hook.db, pid)
    if not rec or rec["status"] != "pending":
        return f"no open pending write #{pid}"
    if rec["subsystem"] != "principles":
        return approve_skill(hook.db, hook.workspace, pid)

    ops = json.loads(rec["payload"]).get("ops", [])
    counts = await apply_ops(hook, ops, hook.config.principles)
    hook.db.execute(
        "UPDATE pending_writes SET status='approved', resolved_at=? WHERE id=?",
        (time.time(), pid),
    )
    hook.db.commit()
    return f"approved #{pid}: applied curator ops {counts}"
