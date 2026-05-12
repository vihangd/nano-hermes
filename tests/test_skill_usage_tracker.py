"""Tests for SkillUsageTracker coordinator."""
from __future__ import annotations

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.config import SkillStatsConfig
from nano_hermes.coordinator.skills import SkillUsageTracker


@pytest.fixture
def hook(tmp_path):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop)


@pytest.fixture
def cfg() -> SkillStatsConfig:
    return SkillStatsConfig()


@pytest.fixture
def tracker(hook, cfg) -> SkillUsageTracker:
    return SkillUsageTracker(db=hook.db, config=cfg)


class TestResetIteration:
    def test_clears_candidates_and_loaded(self, tracker: SkillUsageTracker) -> None:
        tracker._candidate_skills = ["foo", "bar"]
        tracker._loaded_skills = {"baz": 0}
        tracker.reset_iteration()
        assert tracker._candidate_skills == []
        assert tracker._loaded_skills == {}


class TestRecordCandidates:
    def test_extends_candidate_list(self, tracker: SkillUsageTracker) -> None:
        tracker.record_candidates(["a", "b"])
        tracker.record_candidates(["c"])
        assert tracker._candidate_skills == ["a", "b", "c"]


class TestUpdateAccumulators:
    def test_merges_candidates_and_reads_into_session(
        self, tracker: SkillUsageTracker
    ) -> None:
        tracker._candidate_skills = ["skill-a"]
        tracker._loaded_skills = {"skill-b": 0}
        tracker.update_accumulators()
        assert "skill-a" in tracker.session_skills_used
        assert "skill-b" in tracker.session_skills_used

    def test_accumulates_across_iterations(self, tracker: SkillUsageTracker) -> None:
        tracker.record_candidates(["first"])
        tracker.update_accumulators()
        tracker.reset_iteration()
        tracker.record_candidates(["second"])
        tracker.update_accumulators()
        assert "first" in tracker.session_skills_used
        assert "second" in tracker.session_skills_used


class TestResetSession:
    def test_returns_accumulated_data_and_clears(
        self, tracker: SkillUsageTracker
    ) -> None:
        tracker._session_skills_used = {"x", "y"}
        tracker._session_had_errors = True
        skills, loaded, errors = tracker.reset_session()
        assert skills == {"x", "y"}
        assert isinstance(loaded, set)
        assert errors is True
        assert tracker._session_skills_used == set()
        assert tracker._session_skills_loaded == set()
        assert tracker._session_had_errors is False


class TestExtractSkillNameFromPath:
    def test_absolute_path(self) -> None:
        name = SkillUsageTracker.extract_skill_name_from_path(
            "/workspace/skills/my-skill/SKILL.md"
        )
        assert name == "my-skill"

    def test_relative_path(self) -> None:
        name = SkillUsageTracker.extract_skill_name_from_path(
            "skills/my-skill/SKILL.md"
        )
        assert name == "my-skill"

    def test_dotslash_prefix(self) -> None:
        name = SkillUsageTracker.extract_skill_name_from_path(
            "./skills/my-skill/SKILL.md"
        )
        assert name == "my-skill"

    def test_non_skill_path_returns_none(self) -> None:
        assert SkillUsageTracker.extract_skill_name_from_path("/foo/bar.txt") is None

    def test_empty_string_returns_none(self) -> None:
        assert SkillUsageTracker.extract_skill_name_from_path("") is None

    def test_none_input_returns_none(self) -> None:
        assert SkillUsageTracker.extract_skill_name_from_path(None) is None  # type: ignore[arg-type]


class TestCheckPromotions:
    def test_promotes_draft_to_active_at_threshold(
        self, hook, cfg: SkillStatsConfig
    ) -> None:
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, use_count, success_count) "
            "VALUES (?, 'draft', 5, ?)",
            ("promo-skill", cfg.promotion_threshold),
        )
        hook.db.commit()

        tracker = SkillUsageTracker(db=hook.db, config=cfg)
        tracker.check_promotions(["promo-skill"])

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("promo-skill",)
        ).fetchone()
        assert row[0] == "active"

    def test_no_promotion_below_threshold(
        self, hook, cfg: SkillStatsConfig
    ) -> None:
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, use_count, success_count) "
            "VALUES (?, 'draft', 2, 1)",
            ("draft-skill",),
        )
        hook.db.commit()

        tracker = SkillUsageTracker(db=hook.db, config=cfg)
        tracker.check_promotions(["draft-skill"])

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("draft-skill",)
        ).fetchone()
        assert row[0] == "draft"

    def test_unknown_skill_is_a_noop(self, hook, cfg: SkillStatsConfig) -> None:
        tracker = SkillUsageTracker(db=hook.db, config=cfg)
        tracker.check_promotions(["does-not-exist"])  # must not raise
