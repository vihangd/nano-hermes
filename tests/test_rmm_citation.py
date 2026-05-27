"""Tests for RMM citation feedback — view/cite counters and ranking influence."""
from __future__ import annotations

import time

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.coordinator.citation import cite_score, is_cited


def _seed_session(db) -> int:
    cur = db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        ("rmm:1", time.time()),
    )
    return int(cur.lastrowid)


def _insert_reflection(db, session_id: int, content: str) -> int:
    cur = db.execute(
        "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
        (session_id, content, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


class TestCiteScore:
    def test_full_overlap(self):
        assert cite_score("alpha bravo charlie delta", "alpha bravo charlie delta result") == 1.0

    def test_no_overlap(self):
        assert cite_score("alpha bravo charlie", "xyzzy unrelated stuff") == 0.0

    def test_short_tokens_ignored(self):
        # 3-char tokens don't count; only "alpha" / "bravo" matter on each side.
        score = cite_score("hi do alpha bravo", "alpha bravo here")
        # Both significant tokens from injection appear in response.
        assert score == 1.0

    def test_partial_overlap(self):
        score = cite_score("alpha bravo charlie delta", "alpha bravo only")
        assert 0.45 < score < 0.55  # 2 of 4

    def test_empty_injection_returns_zero(self):
        assert cite_score("", "anything") == 0.0

    def test_empty_response_returns_zero(self):
        assert cite_score("alpha bravo", "") == 0.0


class TestIsCited:
    def test_above_threshold(self):
        assert is_cited("alpha bravo charlie delta", "alpha bravo here")  # 0.5 ≥ 0.3

    def test_below_threshold(self):
        assert not is_cited("alpha bravo charlie delta echo", "alpha mostly elsewhere")  # 0.2 < 0.3

    def test_custom_threshold(self):
        assert is_cited("alpha bravo charlie", "alpha here", threshold=0.3)
        assert not is_cited("alpha bravo charlie", "alpha here", threshold=0.5)


class TestSchemaMigration:
    def test_reflections_has_view_and_cite_columns(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        cols = {
            row[1] for row in hook.db.execute("PRAGMA table_info(reflections)").fetchall()
        }
        assert "view_count" in cols
        assert "cite_count" in cols

    def test_defaults_zero(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        sid = _seed_session(hook.db)
        rid = _insert_reflection(hook.db, sid, "test content")
        row = hook.db.execute(
            "SELECT view_count, cite_count FROM reflections WHERE id = ?", (rid,)
        ).fetchone()
        assert row == (0, 0)


class TestRecordIterationCitations:
    def test_bumps_view_count_for_all_injected(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        sid = _seed_session(hook.db)
        r1 = _insert_reflection(hook.db, sid, "alpha bravo charlie delta")
        r2 = _insert_reflection(hook.db, sid, "echo foxtrot golf hotel")
        coord = hook._reflection_coord
        coord._injected_this_iteration = [(r1, "alpha bravo charlie delta"), (r2, "echo foxtrot golf hotel")]
        coord.record_iteration_citations("totally unrelated response text")
        rows = {r[0]: (r[1], r[2]) for r in hook.db.execute("SELECT id, view_count, cite_count FROM reflections").fetchall()}
        assert rows[r1] == (1, 0)  # (view, cite) — no overlap
        assert rows[r2] == (1, 0)

    def test_bumps_cite_count_when_cited(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        sid = _seed_session(hook.db)
        r1 = _insert_reflection(hook.db, sid, "alpha bravo charlie delta")
        r2 = _insert_reflection(hook.db, sid, "echo foxtrot golf hotel")
        coord = hook._reflection_coord
        coord._injected_this_iteration = [(r1, "alpha bravo charlie delta"), (r2, "echo foxtrot golf hotel")]
        # Response uses tokens only from r1.
        coord.record_iteration_citations("answer involves alpha and bravo and charlie analysis")
        rows = {r[0]: (r[1], r[2]) for r in hook.db.execute("SELECT id, view_count, cite_count FROM reflections").fetchall()}
        assert rows[r1] == (1, 1)
        assert rows[r2] == (1, 0)

    def test_clears_injection_list(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        sid = _seed_session(hook.db)
        r1 = _insert_reflection(hook.db, sid, "alpha bravo charlie delta")
        coord = hook._reflection_coord
        coord._injected_this_iteration = [(r1, "alpha bravo charlie delta")]
        coord.record_iteration_citations("alpha bravo response")
        assert coord._injected_this_iteration == []
        # Running again with empty list is a no-op.
        coord.record_iteration_citations("alpha bravo again")
        row = hook.db.execute(
            "SELECT view_count, cite_count FROM reflections WHERE id = ?", (r1,)
        ).fetchone()
        assert row == (1, 1)  # didn't double-bump


class TestInjectionTracksPerIteration:
    def test_session_injections_populate_iteration_list(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        sid = _seed_session(hook.db)
        r1 = _insert_reflection(hook.db, sid, "first reflection")
        r2 = _insert_reflection(hook.db, sid, "second reflection")
        coord = hook._reflection_coord
        assert coord._injected_this_iteration == []
        msgs = coord.get_session_injections(sid)
        assert msgs  # something was injected
        injected_ids = {rid for rid, _ in coord._injected_this_iteration}
        assert injected_ids == {r1, r2}
