"""Tests for nano_hermes.backup — SQLite-safe snapshot utilities."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest

from nano_hermes.backup import _prune, snapshot_db, snapshot_quick


def _make_db(path: Path, *, rows: int = 3) -> Path:
    """Create a minimal SQLite DB with *rows* test rows. Returns *path*."""
    with sqlite3.connect(str(path)) as conn:
        conn.execute("CREATE TABLE data (val TEXT)")
        conn.executemany(
            "INSERT INTO data VALUES (?)", [(f"row{i}",) for i in range(rows)]
        )
    return path


class TestSnapshotDb:
    def test_snapshot_creates_destination(self, tmp_path):
        src = _make_db(tmp_path / "src.db")
        dst = tmp_path / "dst.db"
        snapshot_db(src, dst)
        assert dst.exists()

    def test_snapshot_round_trips_data(self, tmp_path):
        src = _make_db(tmp_path / "src.db", rows=5)
        dst = tmp_path / "dst.db"
        snapshot_db(src, dst)
        with sqlite3.connect(str(dst)) as conn:
            rows = conn.execute("SELECT val FROM data ORDER BY val").fetchall()
        assert len(rows) == 5
        assert rows[0][0] == "row0"

    def test_snapshot_passes_integrity_check(self, tmp_path):
        src = _make_db(tmp_path / "src.db")
        dst = tmp_path / "dst.db"
        snapshot_db(src, dst)
        with sqlite3.connect(str(dst)) as conn:
            result = conn.execute("PRAGMA integrity_check").fetchone()[0]
        assert result == "ok"

    def test_missing_src_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="source database not found"):
            snapshot_db(tmp_path / "ghost.db", tmp_path / "dst.db")

    def test_existing_dst_refused_without_overwrite(self, tmp_path):
        src = _make_db(tmp_path / "src.db")
        dst = tmp_path / "dst.db"
        dst.write_bytes(b"existing")
        with pytest.raises(FileExistsError, match="already exists"):
            snapshot_db(src, dst)
        # File must be untouched.
        assert dst.read_bytes() == b"existing"

    def test_overwrite_flag_replaces_destination(self, tmp_path):
        src = _make_db(tmp_path / "src.db")
        dst = tmp_path / "dst.db"
        dst.write_bytes(b"stale")
        snapshot_db(src, dst, overwrite=True)
        with sqlite3.connect(str(dst)) as conn:
            assert conn.execute("SELECT COUNT(*) FROM data").fetchone()[0] == 3

    def test_creates_parent_directories(self, tmp_path):
        src = _make_db(tmp_path / "src.db")
        dst = tmp_path / "deep" / "nested" / "dst.db"
        snapshot_db(src, dst)
        assert dst.exists()

    def test_open_write_transaction_excluded_from_snapshot(self, tmp_path):
        """Rows inserted inside an uncommitted transaction must not appear
        in a snapshot taken concurrently (the backup API sees committed state).
        """
        src = _make_db(tmp_path / "src.db", rows=2)
        dst = tmp_path / "dst.db"

        # Open a connection and begin an explicit transaction, then snapshot
        # *before* committing.
        writer = sqlite3.connect(str(src))
        writer.isolation_level = None  # autocommit off
        writer.execute("BEGIN")
        writer.execute("INSERT INTO data VALUES ('uncommitted')")

        snapshot_db(src, dst)

        writer.rollback()
        writer.close()

        with sqlite3.connect(str(dst)) as conn:
            vals = {r[0] for r in conn.execute("SELECT val FROM data").fetchall()}
        assert "uncommitted" not in vals


class TestSnapshotQuick:
    def test_creates_snapshot_in_snapshots_subdir(self, tmp_path):
        _make_db(tmp_path / "nano-hermes.db")
        snap = snapshot_quick(tmp_path)
        assert snap.parent == tmp_path / "snapshots"
        assert snap.exists()

    def test_returns_path_of_new_snapshot(self, tmp_path):
        _make_db(tmp_path / "nano-hermes.db")
        snap = snapshot_quick(tmp_path)
        assert snap.suffix == ".db"
        assert "nano-hermes.db" in snap.name

    def test_pruning_keeps_at_most_retain_files(self, tmp_path):
        _make_db(tmp_path / "nano-hermes.db")
        for _ in range(5):
            snapshot_quick(tmp_path, retain=3)
            time.sleep(0.01)  # ensure distinct timestamps
        snaps = list((tmp_path / "snapshots").glob("*.snapshot.db"))
        assert len(snaps) == 3

    def test_missing_db_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            snapshot_quick(tmp_path)


class TestPrune:
    def test_prune_removes_oldest_when_over_limit(self, tmp_path):
        # Use nanosecond-style timestamps so sort order matches creation order.
        base_ns = 1_700_000_000_000_000_000
        for i in range(5):
            (tmp_path / f"x.db.{base_ns + i}.snapshot.db").write_bytes(b"x")
        _prune(tmp_path, "x.db", retain=3)
        remaining = sorted(tmp_path.glob("*.snapshot.db"))
        assert len(remaining) == 3
        # Oldest (base+0, base+1) removed; newest (base+2, base+3, base+4) kept.
        ts_vals = [int(p.stem.split(".")[2]) for p in remaining]
        assert all(t >= base_ns + 2 for t in ts_vals)

    def test_prune_noop_when_under_limit(self, tmp_path):
        for i in range(3):
            (tmp_path / f"x.db.{i}.snapshot.db").write_bytes(b"x")
        _prune(tmp_path, "x.db", retain=5)
        assert len(list(tmp_path.glob("*.snapshot.db"))) == 3
