"""Pre-evolution snapshot + offline rollback."""
from __future__ import annotations

from pathlib import Path

from nano_hermes.session.db import open_db
from nano_hermes.skills.evolution_snapshot import (
    latest_snapshot,
    rollback_evolution,
    snapshot_evolution,
)


def _skill(ws: Path, name: str, body: str) -> Path:
    d = ws / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    p = d / "SKILL.md"
    p.write_text(body)
    return p


def _seed_db(ws: Path) -> None:
    conn = open_db(ws, 512)
    conn.close()  # just create the file/schema


class TestSnapshot:
    def test_creates_db_and_skills_tar(self, tmp_path):
        _seed_db(tmp_path)
        _skill(tmp_path, "alpha", "v1")
        snap = snapshot_evolution(tmp_path, now_ns=1)
        assert (snap / "state.snapshot.db").exists()
        assert (snap / "skills.tar").exists()
        assert latest_snapshot(tmp_path) == snap

    def test_retain_prunes_oldest(self, tmp_path):
        _seed_db(tmp_path)
        _skill(tmp_path, "alpha", "v1")
        for ns in (1, 2, 3, 4):
            snapshot_evolution(tmp_path, retain=2, now_ns=ns)
        root = tmp_path / "nano_hermes" / "snapshots" / "evolution"
        kept = sorted(p.name for p in root.iterdir())
        assert kept == ["3", "4"]  # only the 2 newest survive

    def test_missing_db_and_skills_is_graceful(self, tmp_path):
        snap = snapshot_evolution(tmp_path, now_ns=1)
        assert snap.exists()
        assert not (snap / "state.snapshot.db").exists()
        assert not (snap / "skills.tar").exists()


class TestRollback:
    def test_restores_skill_file(self, tmp_path):
        _seed_db(tmp_path)
        path = _skill(tmp_path, "alpha", "good")
        snapshot_evolution(tmp_path, now_ns=1)
        path.write_text("BROKEN by evolution")  # simulate a bad rewrite
        _skill(tmp_path, "beta", "added after snapshot")

        restored = rollback_evolution(tmp_path)
        assert restored is not None
        assert path.read_text() == "good"  # reverted
        # skills/ replaced wholesale by the snapshot — beta is gone
        assert not (tmp_path / "skills" / "beta").exists()

    def test_restores_db_rows(self, tmp_path):
        _seed_db(tmp_path)
        _skill(tmp_path, "alpha", "v1")
        snapshot_evolution(tmp_path, now_ns=1)

        # Mutate the DB after the snapshot, then close before rollback.
        conn = open_db(tmp_path, 512)
        conn.execute(
            "INSERT INTO principles (condition, action, confidence, created_at) "
            "VALUES ('c', 'a', 0.5, 1.0)"
        )
        conn.commit()
        conn.close()

        rollback_evolution(tmp_path)

        conn = open_db(tmp_path, 512)
        n = conn.execute("SELECT COUNT(*) FROM principles").fetchone()[0]
        conn.close()
        assert n == 0  # the post-snapshot insert is gone

    def test_no_snapshot_returns_none(self, tmp_path):
        assert rollback_evolution(tmp_path) is None


class TestPartialFailureSafety:
    def test_failed_snapshot_does_not_shadow_good_one(self, tmp_path, monkeypatch):
        _seed_db(tmp_path)
        _skill(tmp_path, "alpha", "good")
        good = snapshot_evolution(tmp_path, now_ns=1)

        # Second snapshot fails mid-tar — must NOT leave a partial dir that
        # becomes latest_snapshot and shadows the good one.
        import nano_hermes.skills.evolution_snapshot as es

        def boom(*a, **k):
            raise OSError("disk full")

        monkeypatch.setattr(es.tarfile, "open", boom)
        try:
            snapshot_evolution(tmp_path, now_ns=2)
        except OSError:
            pass
        assert latest_snapshot(tmp_path) == good  # still the complete one
        root = tmp_path / "nano_hermes" / "snapshots" / "evolution"
        assert [p.name for p in root.iterdir() if p.is_dir()] == ["1"]

    def test_rollback_refuses_empty_tar_keeps_skills(self, tmp_path):
        _seed_db(tmp_path)
        _skill(tmp_path, "alpha", "live")
        snap = snapshot_evolution(tmp_path, now_ns=1)
        # Corrupt the snapshot's tar to empty; rollback must refuse, not wipe.
        (snap / "skills.tar").write_bytes(b"")
        import pytest
        with pytest.raises(Exception):
            rollback_evolution(tmp_path)
        assert (tmp_path / "skills" / "alpha" / "SKILL.md").read_text() == "live"
