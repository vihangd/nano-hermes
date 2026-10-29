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


def test_chunks_vec_migrated_to_cosine_preserving_rows(tmp_path):
    from sqlite_vec import serialize_float32

    ws = str(tmp_path / "state.db")
    db = open_db(ws, 4)
    # Recreate chunks_vec the pre-cosine way (default L2 metric).
    db.execute("DROP TABLE chunks_vec")
    db.execute(
        "CREATE VIRTUAL TABLE chunks_vec USING vec0("
        "chunk_id INTEGER PRIMARY KEY, embedding FLOAT[4])"
    )
    db.executemany(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
        [(1, serialize_float32([2, 0, 0, 0])), (2, serialize_float32([0, 1, 0, 0]))],
    )
    db.commit()
    db.close()

    db = open_db(ws, 4)
    ddl = db.execute(
        "SELECT sql FROM sqlite_master WHERE name='chunks_vec'"
    ).fetchone()[0]
    assert "distance_metric=cosine" in ddl
    ids = [r[0] for r in db.execute("SELECT chunk_id FROM chunks_vec ORDER BY chunk_id")]
    assert ids == [1, 2]
    hits = db.execute(
        "SELECT chunk_id, distance FROM chunks_vec WHERE embedding MATCH ? "
        "AND k = 2 ORDER BY distance",
        (serialize_float32([1, 0, 0, 0]),),
    ).fetchall()
    # Cosine: [1,0,0,0] vs [2,0,0,0] is distance 0 (L2 would be 1.0).
    assert hits[0][0] == 1 and abs(hits[0][1]) < 1e-6
    db.close()
