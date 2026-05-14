"""Tests for search-time greedy diversity dedup in SkillIndexer._vec_query."""
from __future__ import annotations

import math

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.config import SkillStatsConfig
from nano_hermes.skills.indexer import SkillHit, SkillIndexer


def _insert_skill(db, name: str, vec: np.ndarray, status: str = "active") -> int:
    with db:
        db.execute(
            "INSERT OR REPLACE INTO skill_stats (name, status, use_count, success_count) "
            "VALUES (?, ?, 0, 0)",
            (name, status),
        )
        skill_id = db.execute(
            "SELECT id FROM skill_stats WHERE name = ?", (name,)
        ).fetchone()[0]
        db.execute("DELETE FROM skill_vec WHERE skill_id = ?", (skill_id,))
        db.execute(
            "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)",
            (skill_id, vec.astype(np.float32).tobytes()),
        )
    return skill_id


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _unit(dims: int, idx: int) -> np.ndarray:
    v = np.zeros(dims, dtype=np.float32)
    v[idx] = 1.0
    return v


DIMS = 512


class TestSkillHitSiblings:
    def test_siblings_default_empty(self):
        h = SkillHit(name="x", description="d", location="l", distance=0.1)
        assert h.siblings == []

    def test_siblings_independent_per_instance(self):
        a = SkillHit(name="a", description="", location="", distance=0.1)
        b = SkillHit(name="b", description="", location="", distance=0.2)
        a.siblings.append("x")
        assert b.siblings == []


class TestVecQueryDedup:
    """Unit-test _vec_query directly by seeding skill_vec and calling it."""

    def _make_indexer(self, db, threshold: float = 0.82) -> SkillIndexer:
        cfg = SkillStatsConfig(ranking_mode="off", skill_search_dedup_threshold=threshold)
        from unittest.mock import MagicMock
        loader = MagicMock()
        return SkillIndexer(
            db=db,
            skills_loader=loader,
            embedder_factory=MagicMock(),
            stats_config=cfg,
        )

    def test_distinct_skills_all_kept(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Three orthogonal skills — cosine similarity = 0, all distinct
        _insert_skill(db, "skill-a", _unit(DIMS, 0))
        _insert_skill(db, "skill-b", _unit(DIMS, 1))
        _insert_skill(db, "skill-c", _unit(DIMS, 2))

        indexer = self._make_indexer(db, threshold=0.82)
        desc = {"skill-a": "a", "skill-b": "b", "skill-c": "c"}
        loc = {"skill-a": "p/a", "skill-b": "p/b", "skill-c": "p/c"}

        hits = indexer._vec_query(_unit(DIMS, 0), k=5, description_by_name=desc, location_by_name=loc)
        names = [h.name for h in hits]
        assert "skill-a" in names
        assert "skill-b" in names
        assert "skill-c" in names
        assert all(h.siblings == [] for h in hits)

    def test_near_duplicate_becomes_sibling(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Two skills with cosine similarity ≈ 0.999 (well above 0.82 threshold)
        v_primary = _unit(DIMS, 0)
        theta = math.acos(0.999)
        v_dup = np.zeros(DIMS, dtype=np.float32)
        v_dup[0] = math.cos(theta)
        v_dup[1] = math.sin(theta)
        v_dup = _norm(v_dup)

        # Third skill is distinct
        v_other = _unit(DIMS, 2)

        _insert_skill(db, "primary", v_primary)
        _insert_skill(db, "duplicate", v_dup)
        _insert_skill(db, "other", v_other)

        indexer = self._make_indexer(db, threshold=0.82)
        desc = {"primary": "p", "duplicate": "d", "other": "o"}
        loc = {k: k for k in desc}

        # Query toward primary — duplicate should be suppressed
        hits = indexer._vec_query(v_primary, k=5, description_by_name=desc, location_by_name=loc)
        names = [h.name for h in hits]

        assert "primary" in names
        assert "duplicate" not in names
        assert "other" in names

        primary_hit = next(h for h in hits if h.name == "primary")
        assert "duplicate" in primary_hit.siblings

    def test_sibling_recorded_on_displacing_hit(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        v_a = _unit(DIMS, 0)

        # Two near-duplicates of v_a (cosine sims 0.9 and 0.85, both > 0.82)
        theta1 = math.acos(0.9)
        v_b = np.zeros(DIMS, dtype=np.float32)
        v_b[0] = math.cos(theta1)
        v_b[1] = math.sin(theta1)
        v_b = _norm(v_b)

        theta2 = math.acos(0.85)
        v_c = np.zeros(DIMS, dtype=np.float32)
        v_c[0] = math.cos(theta2)
        v_c[1] = math.sin(theta2)
        v_c = _norm(v_c)

        _insert_skill(db, "top", v_a)
        _insert_skill(db, "near1", v_b)
        _insert_skill(db, "near2", v_c)

        indexer = self._make_indexer(db, threshold=0.82)
        desc = {"top": "t", "near1": "n1", "near2": "n2"}
        loc = {k: k for k in desc}

        hits = indexer._vec_query(v_a, k=5, description_by_name=desc, location_by_name=loc)
        names = [h.name for h in hits]
        assert "top" in names
        assert "near1" not in names
        assert "near2" not in names
        top_hit = next(h for h in hits if h.name == "top")
        assert set(top_hit.siblings) == {"near1", "near2"}

    def test_threshold_1_0_disables_dedup(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        v_a = _unit(DIMS, 0)
        v_b = _norm(np.array([0.999] + [0.0] * (DIMS - 1), dtype=np.float32))
        _insert_skill(db, "skill-a", v_a)
        _insert_skill(db, "skill-b", v_b)

        # threshold=1.0 disables dedup entirely
        indexer = self._make_indexer(db, threshold=1.0)
        desc = {"skill-a": "a", "skill-b": "b"}
        loc = {k: k for k in desc}

        hits = indexer._vec_query(v_a, k=5, description_by_name=desc, location_by_name=loc)
        names = [h.name for h in hits]
        assert "skill-a" in names
        assert "skill-b" in names

    def test_deprecated_skills_excluded(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        v_a = _unit(DIMS, 0)
        _insert_skill(db, "active-skill", v_a)
        _insert_skill(db, "deprecated-skill", v_a, status="deprecated")

        indexer = self._make_indexer(db, threshold=0.82)
        desc = {"active-skill": "a", "deprecated-skill": "dep"}
        loc = {k: k for k in desc}

        hits = indexer._vec_query(v_a, k=5, description_by_name=desc, location_by_name=loc)
        names = [h.name for h in hits]
        assert "active-skill" in names
        assert "deprecated-skill" not in names

    def test_k_respected_after_dedup(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # 5 distinct skills
        for i in range(5):
            _insert_skill(db, f"skill-{i}", _unit(DIMS, i))

        indexer = self._make_indexer(db, threshold=0.82)
        desc = {f"skill-{i}": f"desc {i}" for i in range(5)}
        loc = {k: k for k in desc}

        query_vec = _unit(DIMS, 0)
        hits = indexer._vec_query(query_vec, k=3, description_by_name=desc, location_by_name=loc)
        assert len(hits) <= 3
