"""Pre-evolution snapshots — a coarse one-shot undo for a bad auto-evolution batch.

GEPA / the rewriter rewrite SKILL.md files, and the curator + ACE principle
loop mutate skill_stats / principles in the DB. Per-skill `skill_versions`
cover single-skill diffs, but a bad batch across many skills (or the DB rows)
has no single undo. Before each evolution cycle we snapshot BOTH the state DB
(sqlite-safe, via the online-backup API) and the `skills/` directory into
``<workspace>/nano_hermes/snapshots/evolution/<ns>/``, keeping the last N.

Rollback is an OFFLINE operation — run it with the agent stopped, since it
replaces the live state DB and the skills directory:

    python -m nano_hermes.skills.evolution_snapshot /path/to/workspace
"""
from __future__ import annotations

import logging
import shutil
import tarfile
import time
from pathlib import Path

from ..backup import snapshot_db
from ..paths import plugin_root, state_db

log = logging.getLogger(__name__)

_EVO_SUBDIR = "snapshots/evolution"
_DB_SNAP = "state.snapshot.db"
_SKILLS_TAR = "skills.tar"


def _evo_root(workspace: Path) -> Path:
    return plugin_root(Path(workspace)) / _EVO_SUBDIR


def _snapshot_dirs(root: Path) -> list[Path]:
    # Directory names are zero-padded-equivalent ns timestamps (same length),
    # so a lexical sort is chronological.
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.exists() else []


def snapshot_evolution(
    workspace: Path, *, retain: int = 5, now_ns: int | None = None
) -> Path:
    """Snapshot the DB + skills/ dir before an evolution batch. Returns the dir.

    Prunes oldest snapshots so at most *retain* are kept. Best-effort on the
    individual parts: a missing DB or skills dir is simply skipped.
    """
    ws = Path(workspace)
    root = _evo_root(ws)
    root.mkdir(parents=True, exist_ok=True)
    ts = now_ns if now_ns is not None else time.time_ns()
    dest = root / str(ts)

    # Build into a temp dir and rename into place only after BOTH artifacts
    # succeed, so latest_snapshot() never points at a half-written snapshot a
    # rollback would then trust.
    tmp = root / f"_tmp_{ts}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        db = state_db(ws)
        if db.exists():
            snapshot_db(db, tmp / _DB_SNAP, overwrite=True)
        skills = ws / "skills"
        if skills.exists():
            with tarfile.open(tmp / _SKILLS_TAR, "w") as tf:
                tf.add(skills, arcname="skills")
        shutil.rmtree(dest, ignore_errors=True)
        tmp.rename(dest)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    _prune(root, retain)
    return dest


def _prune(root: Path, retain: int) -> None:
    snaps = _snapshot_dirs(root)
    to_delete = snaps if retain <= 0 else snaps[:-retain]
    for old in to_delete:
        shutil.rmtree(old, ignore_errors=True)


def latest_snapshot(workspace: Path) -> Path | None:
    snaps = _snapshot_dirs(_evo_root(Path(workspace)))
    return snaps[-1] if snaps else None


def rollback_evolution(workspace: Path) -> Path | None:
    """Restore the most recent pre-evolution snapshot. OFFLINE only — replaces
    the live state DB and skills/ dir. Returns the restored snapshot dir, or
    None when there's nothing to restore."""
    ws = Path(workspace)
    snap = latest_snapshot(ws)
    if snap is None:
        return None

    db_snap = snap / _DB_SNAP
    if db_snap.exists():
        dst = state_db(ws)
        dst.parent.mkdir(parents=True, exist_ok=True)
        # Remove the live DB and its WAL/SHM sidecars, then rebuild via the
        # sqlite online-backup API so the restored file is self-consistent
        # (a raw copy can leave a WAL header referencing a now-gone sidecar).
        for ext in ("", "-wal", "-shm"):
            side = Path(str(dst) + ext)
            if side.exists():
                side.unlink()
        snapshot_db(db_snap, dst, overwrite=True)

    skills_tar = snap / _SKILLS_TAR
    if skills_tar.exists():
        # Validate the archive BEFORE destroying the live skills dir — a
        # truncated tar must not leave the workspace with no skills at all.
        with tarfile.open(skills_tar) as tf:
            members = tf.getmembers()
        if not members:
            raise ValueError(f"refusing rollback: empty skills tar {skills_tar}")
        skills = ws / "skills"
        if skills.exists():
            shutil.rmtree(skills)
        with tarfile.open(skills_tar) as tf:
            try:
                tf.extractall(ws, filter="data")  # 3.12+: safe extraction
            except TypeError:
                tf.extractall(ws)  # 3.11 has no filter kwarg  # noqa: S202

    log.info("rolled back evolution state to snapshot %s", snap.name)
    return snap


if __name__ == "__main__":  # pragma: no cover - manual offline tool
    import sys

    ws = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    restored = rollback_evolution(ws)
    print(f"rolled back to {restored}" if restored else "no evolution snapshot found")
