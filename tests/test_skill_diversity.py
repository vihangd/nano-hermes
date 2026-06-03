"""Tests for FactorMiner-inspired diversity gate at draft→active promotion."""
from __future__ import annotations

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.config import SkillStatsConfig
from nano_hermes.coordinator.skills import SkillUsageTracker


def _seed_skill_with_vec(
    db,
    name: str,
    vec: np.ndarray,
    status: str = "active",
    success_count: int = 5,
    use_count: int = 5,
) -> None:
    """Insert skill_stats + skill_vec rows directly, bypassing the indexer."""
    with db:
        # origin='agent' marks these as auto-evolvable (created via propose_skill);
        # only such skills are eligible for promotion/deprecation transitions.
        db.execute(
            "INSERT OR REPLACE INTO skill_stats "
            "(name, status, use_count, success_count, origin) VALUES (?, ?, ?, ?, 'agent')",
            (name, status, use_count, success_count),
        )
        skill_id = db.execute(
            "SELECT id FROM skill_stats WHERE name = ?", (name,)
        ).fetchone()[0]
        db.execute("DELETE FROM skill_vec WHERE skill_id = ?", (skill_id,))
        db.execute(
            "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)",
            (skill_id, vec.astype(np.float32).tobytes()),
        )


def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


DIMS = 512


class TestDiversityGateAtPromotion:
    def _tracker(self, db) -> SkillUsageTracker:
        cfg = SkillStatsConfig(promotion_threshold=3)
        return SkillUsageTracker(db=db, config=cfg)

    def _status(self, db, name: str) -> str:
        row = db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (name,)
        ).fetchone()
        return row[0] if row else "missing"

    def test_promotion_blocked_when_too_similar(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Active skill pointing in dim 0
        active_vec = np.zeros(DIMS, dtype=np.float32)
        active_vec[0] = 1.0
        _seed_skill_with_vec(db, "existing-skill", active_vec, status="active")

        # Draft skill very similar (cosine sim ≈ 0.9998)
        draft_vec = _norm(np.array([0.99] + [0.0] * (DIMS - 1), dtype=np.float32))
        _seed_skill_with_vec(
            db, "similar-draft", draft_vec, status="draft", success_count=5, use_count=5
        )

        tracker = self._tracker(db)
        tracker.check_promotions(["similar-draft"])

        assert self._status(db, "similar-draft") == "draft", (
            "promotion should be blocked: cosine sim is above 0.88 threshold"
        )

    def test_promotion_allowed_when_sufficiently_different(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Active skill pointing in dim 0
        active_vec = np.zeros(DIMS, dtype=np.float32)
        active_vec[0] = 1.0
        _seed_skill_with_vec(db, "existing-skill", active_vec, status="active")

        # Draft skill orthogonal (cosine sim = 0.0)
        distinct_vec = np.zeros(DIMS, dtype=np.float32)
        distinct_vec[1] = 1.0
        _seed_skill_with_vec(
            db, "distinct-draft", distinct_vec, status="draft", success_count=5, use_count=5
        )

        tracker = self._tracker(db)
        tracker.check_promotions(["distinct-draft"])

        assert self._status(db, "distinct-draft") == "active", (
            "orthogonal skill should be promoted normally"
        )

    def test_promotion_allowed_when_no_active_skills(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        draft_vec = np.zeros(DIMS, dtype=np.float32)
        draft_vec[0] = 1.0
        _seed_skill_with_vec(
            db, "first-skill", draft_vec, status="draft", success_count=5, use_count=5
        )

        tracker = self._tracker(db)
        tracker.check_promotions(["first-skill"])

        assert self._status(db, "first-skill") == "active", (
            "first skill with no active competitors should always promote"
        )

    def test_promotion_allowed_when_embedding_not_indexed(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Active skill in DB
        active_vec = np.zeros(DIMS, dtype=np.float32)
        active_vec[0] = 1.0
        _seed_skill_with_vec(db, "existing-skill", active_vec, status="active")

        # Draft skill with stats but NO skill_vec row (not yet indexed)
        with db:
            db.execute(
                "INSERT INTO skill_stats (name, status, use_count, success_count) "
                "VALUES ('unindexed-draft', 'draft', 5, 5)"
            )

        tracker = self._tracker(db)
        tracker.check_promotions(["unindexed-draft"])

        # No embedding → skip diversity check → allow promotion
        assert self._status(db, "unindexed-draft") == "active"

    def test_above_threshold_is_blocked(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        active_vec = np.zeros(DIMS, dtype=np.float32)
        active_vec[0] = 1.0
        _seed_skill_with_vec(db, "existing-skill", active_vec, status="active")

        # Construct vec with cosine sim 0.90 to active_vec (above 0.88 threshold).
        # Using 0.90 rather than exactly 0.88 to avoid float32 rounding artifacts
        # at the precise boundary value.
        import math
        theta = math.acos(0.90)
        above_vec = np.zeros(DIMS, dtype=np.float32)
        above_vec[0] = math.cos(theta)
        above_vec[1] = math.sin(theta)
        _seed_skill_with_vec(
            db, "above-draft", above_vec, status="draft", success_count=5, use_count=5
        )

        tracker = self._tracker(db)
        tracker.check_promotions(["above-draft"])

        assert self._status(db, "above-draft") == "draft", (
            "sim=0.90 > threshold=0.88 should block promotion"
        )

    def test_deprecation_still_fires_regardless_of_diversity(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        # Active skill that has chronic low success — should be deprecated
        vec = np.zeros(DIMS, dtype=np.float32)
        vec[0] = 1.0
        _seed_skill_with_vec(
            db, "failing-skill", vec, status="active", success_count=0, use_count=10
        )

        tracker = self._tracker(db)
        tracker.check_promotions(["failing-skill"])

        assert self._status(db, "failing-skill") == "deprecated"

    def test_custom_threshold_respected(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        db = hook.db

        active_vec = np.zeros(DIMS, dtype=np.float32)
        active_vec[0] = 1.0
        _seed_skill_with_vec(db, "existing-skill", active_vec, status="active")

        # Draft with cosine sim 0.92 (above default 0.88 but we'll set threshold to 0.95)
        import math
        theta = math.acos(0.92)
        draft_vec = np.zeros(DIMS, dtype=np.float32)
        draft_vec[0] = math.cos(theta)
        draft_vec[1] = math.sin(theta)
        _seed_skill_with_vec(
            db, "high-sim-draft", draft_vec, status="draft", success_count=5, use_count=5
        )

        # With threshold=0.95, sim=0.92 should NOT block
        cfg = SkillStatsConfig(
            promotion_threshold=3,
            diversity_similarity_threshold=0.95,
        )
        tracker = SkillUsageTracker(db=db, config=cfg)
        tracker.check_promotions(["high-sim-draft"])

        assert self._status(db, "high-sim-draft") == "active", (
            "0.92 sim with threshold=0.95 should allow promotion"
        )
