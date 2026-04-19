"""SQLite-safe database snapshot utilities.

Uses ``sqlite3.Connection.backup()`` (the official online-backup API) rather
than raw file copy, so snapshots are safe even while the agent is running in
WAL mode — the backup API coordinates with the WAL and produces a consistent
copy without stopping writes.

Usage (from a cron script)::

    from pathlib import Path
    from nano_hermes.backup import snapshot_quick

    snapshot_quick(Path("~/.nano-hermes").expanduser())
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_SNAPSHOT_SUFFIX = ".snapshot.db"
_DEFAULT_RETAIN = 20


def snapshot_db(src: Path, dst: Path, *, overwrite: bool = False) -> None:
    """Copy *src* to *dst* using the SQLite online-backup API.

    Safe to call while the source database is open and being written to.
    Raises ``FileExistsError`` if *dst* already exists and *overwrite* is
    False (the default).  Raises ``FileNotFoundError`` if *src* does not
    exist.
    """
    src = Path(src)
    dst = Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"source database not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(
            f"snapshot destination already exists: {dst} "
            "(pass overwrite=True to replace it)"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    # Remove stale dst so sqlite3.connect() creates a fresh, empty database.
    # (backup() into a non-SQLite file raises DatabaseError.)
    if dst.exists():
        dst.unlink()
    with sqlite3.connect(str(src)) as src_conn, sqlite3.connect(str(dst)) as dst_conn:
        src_conn.backup(dst_conn)


def snapshot_quick(
    state_dir: Path,
    db_name: str = "nano-hermes.db",
    retain: int = _DEFAULT_RETAIN,
) -> Path:
    """Write a timestamped snapshot of *db_name* inside *state_dir/snapshots/*.

    Prunes the oldest snapshots so at most *retain* files are kept.
    Returns the path of the newly created snapshot.
    """
    state_dir = Path(state_dir)
    src = state_dir / db_name
    snap_dir = state_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    ts = time.time_ns()
    dst = snap_dir / f"{db_name}.{ts}{_SNAPSHOT_SUFFIX}"
    snapshot_db(src, dst)
    _prune(snap_dir, db_name, retain)
    return dst


def _prune(snap_dir: Path, db_name: str, retain: int) -> None:
    pattern = f"{db_name}.*{_SNAPSHOT_SUFFIX}"
    snapshots = sorted(snap_dir.glob(pattern))
    for old in snapshots[:-retain]:
        try:
            old.unlink()
        except OSError:
            pass
