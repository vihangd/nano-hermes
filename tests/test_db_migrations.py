"""Tests for the PRAGMA user_version migration gate."""
from __future__ import annotations

import sqlite3

from nano_hermes.session.db import _MIGRATIONS, _apply_migrations, open_db


def test_fresh_db_stamps_version_and_columns(tmp_path):
    db = open_db(str(tmp_path / "state.db"), 512)
    assert db.execute("PRAGMA user_version").fetchone()[0] == len(_MIGRATIONS)
    cols = {r[1] for r in db.execute("PRAGMA table_info(semantic_facts)")}
    assert "trust_score" in cols
    scols = {r[1] for r in db.execute("PRAGMA table_info(skill_stats)")}
    assert {"last_viewed_at", "view_count"} <= scols
    db.close()


def test_gate_skips_when_current(tmp_path):
    # A bare conn already stamped at target must NOT run migrations: a missing
    # column stays missing, proving the loop was skipped.
    conn = sqlite3.connect(str(tmp_path / "bare.db"))
    conn.execute("CREATE TABLE semantic_facts (id INTEGER PRIMARY KEY)")
    conn.execute(f"PRAGMA user_version = {len(_MIGRATIONS)}")
    conn.commit()
    _apply_migrations(conn)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(semantic_facts)")}
    assert "trust_score" not in cols  # skipped — not re-applied
    conn.close()
