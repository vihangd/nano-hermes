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


class TestWritePathSurvivesCorruptIndex:
    """A derived index must never cost canonical data.

    chunks_ai fires inside `INSERT INTO chunks`, so a corrupt chunks_fts makes
    the transcript write itself raise — and hook.py wraps archiving in a broad
    except that only logs, so the session is silently never archived. Dropping
    the FTS table reproduces this deterministically ("no such table" is already
    classified as corruption).
    """

    def _corrupt(self, db):
        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        db.commit()
        db.execute("DROP TABLE chunks_fts")
        db.commit()

    def _insert(self, db, text, turn=0):
        cur = db.cursor()
        from nano_hermes.session.db import fts_guarded_write
        fts_guarded_write(db, "chunks_fts", lambda: cur.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (1, ?, 'user', ?, 0)", (turn, text),
        ))
        db.commit()
        return cur

    def test_unguarded_insert_really_does_fail(self, db):
        # Anchors the whole class: if a corrupt index stopped breaking the
        # canonical write, everything below would pass vacuously.
        self._corrupt(db)
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(
                "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
                "VALUES (1, 0, 'user', 'x', 0)"
            )

    def test_guarded_insert_preserves_the_row(self, db):
        self._corrupt(db)
        self._insert(db, "survived the corruption")
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1

    def test_corruption_detaches_triggers_and_marks_stale(self, db):
        self._corrupt(db)
        self._insert(db, "row")
        assert stale_fts_tables(db) == ["chunks_fts"]
        live = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'chunks_%'"
        )]
        assert live == [], f"sync triggers still attached: {live}"

    def test_non_corruption_error_still_propagates(self, db):
        # A genuine write bug must stay loud rather than being retried away.
        from nano_hermes.session.db import fts_guarded_write
        with pytest.raises(sqlite3.IntegrityError):
            fts_guarded_write(db, "chunks_fts", lambda: db.execute(
                "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
                "VALUES (99999, 0, 'user', 'orphan fk', 0)"
            ))
        assert stale_fts_tables(db) == [], "a non-corruption error flagged the index"

    def test_detach_refuses_unknown_table(self, db):
        from nano_hermes.session.db import detach_fts
        detach_fts(db, "bogus_fts; DROP TABLE chunks --")
        assert stale_fts_tables(db) == []
        assert db.execute("SELECT count(*) FROM chunks").fetchone() is not None

    def test_reopen_restores_triggers_and_clears_marker(self, tmp_path):
        from nano_hermes.session.db import open_db as _open
        path = str(tmp_path / "recover.db")
        db = _open(path, 512)
        self._corrupt(db)
        self._insert(db, "written while detached")
        db.close()

        db2 = _open(path, 512)
        assert stale_fts_tables(db2) == []
        live = sorted(r[0] for r in db2.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'chunks_%'"
        ))
        assert live == ["chunks_ad", "chunks_ai"]

    def test_rows_written_while_detached_become_searchable(self, tmp_path):
        # The recovery that matters: data accepted during the outage must not
        # be permanently invisible to search.
        from nano_hermes.session.db import open_db as _open
        path = str(tmp_path / "recover2.db")
        db = _open(path, 512)
        self._corrupt(db)
        self._insert(db, "rollback the staging deploy", turn=0)
        self._insert(db, "unrelated filler note", turn=1)
        db.close()

        db2 = _open(path, 512)
        hits = _fts_rows(
            db2,
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "rollback", 5,
        )
        assert len(hits) == 1

    def test_trajectory_write_is_guarded_too(self, db):
        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        db.commit()
        db.execute("DROP TABLE trajectories_fts")
        db.commit()
        from nano_hermes.session.db import fts_guarded_write
        fts_guarded_write(db, "trajectories_fts", lambda: db.execute(
            "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
            "VALUES (1, 'rebuild the index', '[]', 'ok', 0)"
        ))
        db.commit()
        assert db.execute("SELECT count(*) FROM trajectories").fetchone()[0] == 1
        assert stale_fts_tables(db) == ["trajectories_fts"]


class TestRealCorruptionExceptionClass:
    """Genuine SQLITE_CORRUPT surfaces as a bare ``sqlite3.DatabaseError``, not
    ``OperationalError``. The earlier handlers caught only the narrower class,
    so they fired for a dropped table but NOT for real on-disk corruption —
    the case they exist for. These pin the distinction."""

    def test_corrupt_file_raises_bare_databaseerror(self, tmp_path):
        path = tmp_path / "corrupt.db"
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE t(a)")
        c.executemany("INSERT INTO t VALUES(?)", [("x" * 200,)] * 500)
        c.commit()
        c.close()
        data = bytearray(path.read_bytes())
        for i in range(4096, min(len(data), 20000)):
            data[i] = 0xFF
        path.write_bytes(bytes(data))

        c2 = sqlite3.connect(path)
        with pytest.raises(sqlite3.DatabaseError) as ei:
            c2.execute("SELECT count(*) FROM t").fetchone()
        assert not isinstance(ei.value, sqlite3.OperationalError), (
            "corruption is now an OperationalError — the narrower handlers "
            "would be correct and these widenings could be reverted"
        )
        assert is_fts_corruption(ei.value)

    def test_read_path_degrades_on_bare_databaseerror(self):
        class _Corrupt:
            def execute(self, *a, **k):
                raise sqlite3.DatabaseError("database disk image is malformed")

            def commit(self):
                pass

        assert _fts_rows(
            _Corrupt(),
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
            "query", 5,
        ) == []

    def test_read_path_still_raises_real_defects(self):
        # Widening the catch must not swallow programming errors.
        class _Broken:
            def execute(self, *a, **k):
                raise sqlite3.ProgrammingError("closed cursor")

        with pytest.raises(sqlite3.ProgrammingError):
            _fts_rows(
                _Broken(),
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? LIMIT ?",
                "query", 5,
            )


class TestDetachDoesNotDiscardCallerRows:
    """detach_fts runs mid-write with the caller's earlier rows uncommitted on
    the same connection. It must never roll those back — a connection-level
    `with conn:` would, destroying the data the guard exists to protect."""

    def test_uncommitted_rows_survive_a_failing_detach(self, db):
        from nano_hermes.session import db as dbmod

        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        db.commit()
        # Uncommitted canonical row, mid-batch.
        db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (1, 0, 'user', 'earlier row', 0)"
        )

        class _DropFails:
            """sqlite3.Connection.execute is read-only, so wrap rather than patch."""

            def __init__(self, conn):
                self._conn = conn

            def execute(self, sql, *a, **k):
                if sql.startswith("DROP TRIGGER"):
                    raise sqlite3.OperationalError("database is locked")
                return self._conn.execute(sql, *a, **k)

            def __getattr__(self, name):
                return getattr(self._conn, name)

        dbmod.detach_fts(_DropFails(db), "chunks_fts")

        db.commit()
        assert db.execute("SELECT count(*) FROM chunks").fetchone()[0] == 1, (
            "a failed detach rolled back the caller's uncommitted rows"
        )


class TestPrinciplesFtsWriteIsGuarded:
    """principles_fts is filled by an explicit INSERT, not a trigger, so
    detach-and-retry cannot help — the mirror is skipped instead so the
    canonical principle still lands."""

    async def test_principle_survives_corrupt_index(self, tmp_path):
        import nano_hermes
        from conftest import _make_loop
        from nano_hermes.skills.principle_tool import PrincipleTool

        hook = nano_hermes.install(_make_loop(tmp_path))
        hook.db.execute("DROP TABLE principles_fts")
        hook.db.commit()

        result = await PrincipleTool(hook=hook).execute(
            condition="When deploying to prod",
            action="Confirm the rollback command first",
        )

        assert not result.startswith("Error"), result
        assert hook.db.execute("SELECT count(*) FROM principles").fetchone()[0] == 1
        assert stale_fts_tables(hook.db) == ["principles_fts"]


class TestStandaloneIndexIsRepopulated:
    """principles_fts is STANDALONE, not external-content. FTS5's 'rebuild'
    succeeds on it but only re-tokenises rows already present — it cannot
    recover a row that was never indexed. Rebuilding it that way cleared the
    stale marker while leaving the principle permanently unsearchable."""

    def _add(self, db, condition, action):
        db.execute(
            "INSERT INTO principles (condition, action, expected_outcome, "
            "confidence, created_at, updated_at, origin) "
            "VALUES (?, ?, 'x', 0.5, 0, 0, 'agent')", (condition, action),
        )
        db.commit()

    def test_unindexed_principle_becomes_searchable_after_rebuild(self, db):
        self._add(db, "rotating certs", "renew before expiry")
        assert db.execute("SELECT count(*) FROM principles_fts").fetchone()[0] == 0

        mark_fts_stale(db, "principles_fts")
        assert rebuild_stale_fts(db) == ["principles_fts"]

        hits = db.execute(
            "SELECT count(*) FROM principles_fts WHERE principles_fts MATCH ?",
            ('"certs"',),
        ).fetchone()[0]
        assert hits == 1, "'rebuild' cannot refill a standalone index"

    def test_rebuild_does_not_duplicate_existing_rows(self, db):
        self._add(db, "deploying prod", "check rollback")
        db.execute(
            "INSERT INTO principles_fts (rowid, condition, action, content_id) "
            "VALUES (1, 'deploying prod', 'check rollback', 1)"
        )
        db.commit()
        mark_fts_stale(db, "principles_fts")
        rebuild_stale_fts(db)
        assert db.execute("SELECT count(*) FROM principles_fts").fetchone()[0] == 1


class TestDetachIsDurableAndTargeted:
    def test_detach_commits_itself(self, db):
        # archiver commits only `if new_ids`, so a detach merely folded into the
        # caller's transaction would be discarded exactly when it is needed. The
        # observable property: after detach_fts there is no pending transaction
        # left holding the marker or the trigger drop.
        from nano_hermes.session import db as dbmod

        db.execute("INSERT INTO sessions (session_key, started_at) VALUES ('s', 0)")
        assert db.in_transaction, "fixture precondition: an open write txn"

        dbmod.detach_fts(db, "chunks_fts")

        assert not db.in_transaction, "detach left its state on the caller's txn"
        assert stale_fts_tables(db) == ["chunks_fts"]
        live = [r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'chunks_%'"
        )]
        assert live == []

    def test_unrelated_missing_table_does_not_strip_triggers(self, db):
        # "no such table" for something else must not silently drop this
        # index's sync triggers — detaching is a destructive schema change.
        from nano_hermes.session.db import fts_guarded_write

        with pytest.raises(sqlite3.DatabaseError):
            fts_guarded_write(db, "chunks_fts", lambda: db.execute(
                "INSERT INTO totally_unrelated_table (x) VALUES (1)"
            ))
        live = sorted(r[0] for r in db.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name LIKE 'chunks_%'"
        ))
        assert live == ["chunks_ad", "chunks_ai"]
        assert stale_fts_tables(db) == []
