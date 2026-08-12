"""FTS index health: corruption detection, staleness marker, self-heal rebuild.

``_fts_rows`` swallows OperationalError so a malformed query only drops the
lexical channel for that query. But the same exception type covers a *corrupt*
index, which silently degrades hybrid RRF to vector-only for good. These tests
pin the distinction, the marker, and the rebuild.
"""
from __future__ import annotations

import sqlite3

import pytest

from nano_hermes.session.db import (
    FTS_TABLES,
    is_fts_corruption,
    mark_fts_stale,
    open_db,
    rebuild_stale_fts,
    stale_fts_tables,
)
from nano_hermes.session.search import _fts_rows, _fts_table_from_sql


@pytest.fixture
def db(tmp_path):
    return open_db(str(tmp_path / "state.db"), 512)


class TestCorruptionClassification:
    @pytest.mark.parametrize(
        "msg",
        [
            "database disk image is malformed",
            "no such table: chunks_fts",
            "database corruption detected",
        ],
    )
    def test_corruption_messages_detected(self, msg):
        assert is_fts_corruption(sqlite3.OperationalError(msg))

    @pytest.mark.parametrize(
        "msg",
        [
            'fts5: syntax error near "("',
            "unable to use function MATCH in the requested context",
        ],
    )
    def test_query_errors_not_treated_as_corruption(self, msg):
        # A bad MATCH expression is per-query and self-correcting — flagging it
        # would trigger pointless rebuilds on ordinary user input.
        assert not is_fts_corruption(sqlite3.OperationalError(msg))


class TestTableNameExtraction:
    def test_extracts_matched_table(self):
        sql = (
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY rank LIMIT ?"
        )
        assert _fts_table_from_sql(sql) == "chunks_fts"

    def test_extracts_joined_form(self):
        sql = (
            "SELECT chunks.id FROM chunks_fts JOIN chunks "
            "ON chunks.id = chunks_fts.rowid WHERE chunks_fts MATCH ? LIMIT ?"
        )
        assert _fts_table_from_sql(sql) == "chunks_fts"

    def test_unknown_when_absent(self):
        assert _fts_table_from_sql("SELECT 1") == "unknown"


class TestStaleMarker:
    def test_no_markers_on_fresh_db(self, db):
        assert stale_fts_tables(db) == []

    def test_mark_and_read_back(self, db):
        mark_fts_stale(db, "chunks_fts")
        assert stale_fts_tables(db) == ["chunks_fts"]

    def test_marking_is_idempotent(self, db):
        mark_fts_stale(db, "chunks_fts")
        mark_fts_stale(db, "chunks_fts")
        assert stale_fts_tables(db) == ["chunks_fts"]

    def test_mark_never_raises_on_broken_connection(self, db):
        db.close()
        mark_fts_stale(db, "chunks_fts")  # must not raise

    def test_corrupt_query_sets_marker_via_fts_rows(self, db):
        # End-to-end on a real connection: drop the index out from under the
        # query so it raises "no such table" (a corruption signal). _fts_rows
        # must degrade to [] *and* record the marker on that same connection.
        db.execute("DROP TABLE chunks_fts")
        db.commit()

        rows = _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "hello world",
            5,
        )
        assert rows == []  # still degrades gracefully
        assert stale_fts_tables(db) == ["chunks_fts"]

    def test_syntax_error_does_not_set_marker(self, db, monkeypatch):
        marked: list[str] = []
        monkeypatch.setattr(
            "nano_hermes.session.search.mark_fts_stale",
            lambda conn, table: marked.append(table),
        )

        class _Executor:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError('fts5: syntax error near "("')

        rows = _fts_rows(
            _Executor(),
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "hello",
            5,
        )
        assert rows == []
        assert marked == [], "a per-query syntax error must not flag the index"
        assert stale_fts_tables(db) == []


class TestRebuild:
    def test_rebuild_clears_marker(self, db):
        mark_fts_stale(db, "chunks_fts")
        repaired = rebuild_stale_fts(db)
        assert repaired == ["chunks_fts"]
        assert stale_fts_tables(db) == []

    def test_rebuild_restores_lexical_hits(self, db):
        db.execute(
            "INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)"
        )
        db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (1, 0, 'user', 'rollback the deployment', 0)"
        )
        db.commit()
        # Wipe the index behind FTS5's back, then rebuild from content.
        db.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('delete-all')")
        db.commit()
        assert _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "rollback",
            5,
        ) == []

        mark_fts_stale(db, "chunks_fts")
        rebuild_stale_fts(db)

        rows = _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "rollback",
            5,
        )
        assert len(rows) == 1

    def test_unknown_marker_is_not_interpolated_into_sql(self, db):
        # Guard against a bogus meta key becoming a SQL injection vector.
        mark_fts_stale(db, "chunks_fts; DROP TABLE chunks --")
        assert rebuild_stale_fts(db) == []
        assert db.execute("SELECT count(*) FROM chunks").fetchone() is not None

    def test_unknown_marker_is_cleared_not_left_forever(self, db):
        # An unrecognised marker can never be rebuilt, so leaving it in place
        # would make nano_status report a permanent, unfixable degradation.
        mark_fts_stale(db, "bogus_fts")
        rebuild_stale_fts(db)
        assert stale_fts_tables(db) == []

    def test_all_known_tables_rebuildable(self, db):
        for t in FTS_TABLES:
            mark_fts_stale(db, t)
        assert sorted(rebuild_stale_fts(db)) == sorted(FTS_TABLES)
        assert stale_fts_tables(db) == []

    def test_open_db_self_heals(self, tmp_path):
        path = str(tmp_path / "heal.db")
        conn = open_db(path, 512)
        mark_fts_stale(conn, "chunks_fts")
        conn.close()

        reopened = open_db(path, 512)
        assert stale_fts_tables(reopened) == []


class TestMarkerOnlyForKnownTables:
    def test_unidentifiable_table_is_not_marked(self, db):
        # SQL with no "<name>_fts MATCH" to extract → "unknown", which no
        # rebuild could ever clear. Better to record nothing.
        from nano_hermes.session.search import _fts_rows as _r

        class _Executor:
            def execute(self, *a, **k):
                raise sqlite3.OperationalError("database disk image is malformed")

        _r(_Executor(), "SELECT 1 WHERE x MATCH ?", "hello", 1)
        assert stale_fts_tables(db) == []

    def test_known_table_still_marked(self, db):
        _fts_rows(
            db,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "hello",
            5,
        )
        # Healthy index — no marker.
        assert stale_fts_tables(db) == []


class TestPrinciplesFtsIsMonitored:
    """principles_fts is declared in FTS_TABLES, so its query path must run
    through _fts_rows or corruption there is undetectable."""

    def test_principle_query_routes_through_fts_rows(self, monkeypatch, tmp_path):
        from conftest import _make_loop
        import nano_hermes

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        seen: list[str] = []
        real = __import__(
            "nano_hermes.session.search", fromlist=["_fts_rows"]
        )._fts_rows

        def _spy(executor, sql, query_text, *params):
            seen.append(sql)
            return real(executor, sql, query_text, *params)

        monkeypatch.setattr("nano_hermes.session.search._fts_rows", _spy)

        hook._reflection_coord.get_principle_injections(
            [{"role": "user", "content": "deploy rollback procedure for staging"}]
        )

        assert any("principles_fts" in s for s in seen), (
            "principle injection bypassed _fts_rows — corruption of "
            "principles_fts would go undetected"
        )
