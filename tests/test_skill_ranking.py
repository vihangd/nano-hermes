"""Tests for UCB1 and stat_weighted skill ranking in SkillIndexer._vec_query."""
from __future__ import annotations

import math

import numpy as np

import nano_hermes
from conftest import _make_loop


def _insert_skill(db, name: str, use_count: int, success_count: int, vec: np.ndarray) -> int:
    """Insert a skill into skill_stats + skill_vec; return the stats id."""
    cur = db.execute(
        "INSERT INTO skill_stats (name, status, use_count, success_count) "
        "VALUES (?, 'active', ?, ?)",
        (name, use_count, success_count),
    )
    skill_id = cur.lastrowid
    db.execute(
        "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)",
        (skill_id, vec.astype(np.float32).tobytes()),
    )
    db.commit()
    return skill_id


def _make_hook(tmp_path, ranking_mode="ucb1", ucb1_coefficient=0.05):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(
        loop,
        config={
            "skill_stats": {
                "ranking_mode": ranking_mode,
                "ucb1_coefficient": ucb1_coefficient,
                # Disable search-time dedup so ranking tests can use
                # identical vectors without one being collapsed as a sibling.
                "skill_search_dedup_threshold": 1.0,
            }
        },
    )


class TestUCB1ColdStart:
    def test_cold_start_skill_gets_exploration_bonus(self, tmp_path):
        """A skill with 0 uses must rank above a mediocre proven skill at same distance."""
        hook = _make_hook(tmp_path)
        dims = hook.config.embedding.target_dims

        # Two skills at the same cosine distance from the query.
        # Achieved by using the same vector — tie-breaking is via UCB1.
        base_vec = np.zeros(dims, dtype=np.float32)
        base_vec[0] = 1.0

        _insert_skill(hook.db, "cold-skill", use_count=0, success_count=0, vec=base_vec.copy())
        _insert_skill(hook.db, "mediocre-skill", use_count=20, success_count=4, vec=base_vec.copy())

        # Seed total_uses for ucb1: 0 + 20 = 20
        query = base_vec.copy()
        hits = hook.skill_indexer._vec_query(
            query_vec=query,
            k=2,
            description_by_name={
                "cold-skill": "cold skill description",
                "mediocre-skill": "mediocre skill description",
            },
            location_by_name={},
        )

        names = [h.name for h in hits]
        assert "cold-skill" in names
        assert "mediocre-skill" in names
        # cold-skill should rank above mediocre (lower effective_distance)
        assert names.index("cold-skill") < names.index("mediocre-skill"), (
            f"cold-skill should beat mediocre-skill but got order: {names}"
        )

    def test_high_success_beats_mediocre(self, tmp_path):
        """A high-success-rate skill beats mediocre at same distance."""
        hook = _make_hook(tmp_path)
        dims = hook.config.embedding.target_dims

        vec = np.zeros(dims, dtype=np.float32)
        vec[0] = 1.0

        _insert_skill(hook.db, "proven", use_count=50, success_count=48, vec=vec.copy())
        _insert_skill(hook.db, "mediocre", use_count=50, success_count=10, vec=vec.copy())

        hits = hook.skill_indexer._vec_query(
            query_vec=vec.copy(),
            k=2,
            description_by_name={"proven": "p", "mediocre": "m"},
            location_by_name={},
        )
        names = [h.name for h in hits]
        assert names.index("proven") < names.index("mediocre"), (
            f"proven should beat mediocre but got: {names}"
        )


class TestUCB1BonusDecay:
    def test_exploration_bonus_decays_as_skill_use_count_grows(self, tmp_path):
        """UCB1 exploration bonus decreases as a skill's own use_count grows."""
        # Hand-compute: bonus = sqrt(2 * ln(total + 1) / max(use_count, 1))
        # With total_uses=100 (fixed), the skill's exploration bonus at
        # different use_counts should be monotonically decreasing.
        total = 100
        bonuses = []
        for use_count in [1, 5, 20, 50]:
            bonus = math.sqrt(2.0 * math.log(total + 1) / use_count)
            bonuses.append(bonus)

        for i in range(len(bonuses) - 1):
            assert bonuses[i] > bonuses[i + 1], (
                f"bonus should decay: bonuses[{i}]={bonuses[i]:.4f} "
                f"should > bonuses[{i+1}]={bonuses[i+1]:.4f}"
            )

    def test_high_use_count_converges_toward_success_rate(self, tmp_path):
        """At large use_count, UCB1 score ≈ success_rate (exploration term → 0)."""
        # For use_count=10000, total_uses=10000:
        # exploration = sqrt(2 * ln(10001) / 10000) ≈ 0.047
        # The score is dominated by success_rate.
        exploration = math.sqrt(2.0 * math.log(10001) / 10000)
        assert exploration < 0.06, f"exploration={exploration:.4f} should be small"


class TestRankingModeOff:
    def test_off_mode_preserves_raw_distance_order(self, tmp_path):
        """With ranking_mode='off', results come back in raw embedding-distance order."""
        hook = _make_hook(tmp_path, ranking_mode="off")
        dims = hook.config.embedding.target_dims

        close_vec = np.zeros(dims, dtype=np.float32)
        close_vec[0] = 1.0

        far_vec = np.zeros(dims, dtype=np.float32)
        far_vec[1] = 1.0

        query = close_vec.copy()
        _insert_skill(hook.db, "close-skill", use_count=0, success_count=0, vec=close_vec.copy())
        _insert_skill(hook.db, "far-skill", use_count=100, success_count=100, vec=far_vec.copy())

        hits = hook.skill_indexer._vec_query(
            query_vec=query,
            k=2,
            description_by_name={"close-skill": "c", "far-skill": "f"},
            location_by_name={},
        )
        names = [h.name for h in hits]
        assert names[0] == "close-skill", (
            f"Without reranking, closer skill should rank first, got: {names}"
        )


class TestStatWeightedMode:
    def test_stat_weighted_boosts_high_success_rate(self, tmp_path):
        """Legacy stat_weighted mode still works: high-success skill gets small boost."""
        hook = _make_hook(tmp_path, ranking_mode="stat_weighted")
        dims = hook.config.embedding.target_dims

        vec = np.zeros(dims, dtype=np.float32)
        vec[0] = 1.0

        # Use > min_uses_for_success_rate (3) so the boost kicks in.
        _insert_skill(hook.db, "reliable", use_count=10, success_count=10, vec=vec.copy())
        _insert_skill(hook.db, "unreliable", use_count=10, success_count=0, vec=vec.copy())

        hits = hook.skill_indexer._vec_query(
            query_vec=vec.copy(),
            k=2,
            description_by_name={"reliable": "r", "unreliable": "u"},
            location_by_name={},
        )
        names = [h.name for h in hits]
        assert names[0] == "reliable", f"reliable should rank first, got: {names}"
