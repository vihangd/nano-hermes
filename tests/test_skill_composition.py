"""Tests for skill co-occurrence tracking (skills/composition.py)."""
from __future__ import annotations

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.composition import get_compositions, record_composition


class TestRecordComposition:
    def test_records_pair(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"skill-a", "skill-b"})

        row = hook.db.execute(
            "SELECT count FROM skill_compositions WHERE skill_a = ? AND skill_b = ?",
            ("skill-a", "skill-b"),
        ).fetchone()
        assert row is not None
        assert row[0] == 1

    def test_normalises_pair_order(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"z-skill", "a-skill"})

        row = hook.db.execute("SELECT skill_a, skill_b FROM skill_compositions").fetchone()
        assert row is not None
        # Alphabetical: a-skill < z-skill
        assert row[0] == "a-skill"
        assert row[1] == "z-skill"

    def test_increments_count_on_second_recording(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"skill-a", "skill-b"})
        record_composition(hook.db, {"skill-a", "skill-b"})

        row = hook.db.execute(
            "SELECT count FROM skill_compositions WHERE skill_a = ? AND skill_b = ?",
            ("skill-a", "skill-b"),
        ).fetchone()
        assert row[0] == 2

    def test_no_op_for_single_skill(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"only-skill"})

        count = hook.db.execute("SELECT COUNT(*) FROM skill_compositions").fetchone()[0]
        assert count == 0

    def test_records_all_pairs_in_triple(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"a", "b", "c"})

        count = hook.db.execute("SELECT COUNT(*) FROM skill_compositions").fetchone()[0]
        assert count == 3  # (a,b), (a,c), (b,c)


class TestGetCompositions:
    def test_returns_frequent_partners(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Record skill-a + skill-b together twice (above min_count=2)
        record_composition(hook.db, {"skill-a", "skill-b"})
        record_composition(hook.db, {"skill-a", "skill-b"})

        partners = get_compositions(hook.db, "skill-a", limit=3, min_count=2)
        assert "skill-b" in partners

    def test_excludes_below_min_count(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"skill-a", "skill-b"})  # count=1

        partners = get_compositions(hook.db, "skill-a", limit=3, min_count=2)
        assert partners == []

    def test_returns_both_directions(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        record_composition(hook.db, {"x-skill", "y-skill"})
        record_composition(hook.db, {"x-skill", "y-skill"})

        # Should work regardless of which is skill_a or skill_b
        from_x = get_compositions(hook.db, "x-skill", limit=3, min_count=2)
        from_y = get_compositions(hook.db, "y-skill", limit=3, min_count=2)
        assert "y-skill" in from_x
        assert "x-skill" in from_y

    def test_empty_when_no_compositions(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        partners = get_compositions(hook.db, "lone-skill", limit=3, min_count=2)
        assert partners == []
