"""Tests for skill reflection triggers (skills/reflection_trigger.py)."""
from __future__ import annotations

import nano_hermes
from conftest import _make_loop
from nano_hermes.config import SkillStatsConfig
from nano_hermes.skills.reflection_trigger import check_skill_reflection_triggers


def _seed_skill(hook, name: str, use_count: int, success_count: int) -> None:
    hook.db.execute(
        "INSERT OR REPLACE INTO skill_stats "
        "(name, status, use_count, success_count) VALUES (?, 'active', ?, ?)",
        (name, use_count, success_count),
    )
    hook.db.commit()


class TestCheckSkillReflectionTriggers:
    def _cfg(self) -> SkillStatsConfig:
        return SkillStatsConfig(min_uses_for_success_rate=3)

    def test_triggers_on_mixed_success_rate(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "mixed-skill", use_count=10, success_count=5)  # 50%
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["mixed-skill"], self._cfg(), triggered
        )
        assert suggestions, "should suggest reflection for 50% success rate"
        assert "mixed-skill" in triggered

    def test_no_trigger_on_high_success_rate(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "good-skill", use_count=10, success_count=9)  # 90%
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["good-skill"], self._cfg(), triggered
        )
        assert not suggestions, "high success rate should not trigger"

    def test_no_trigger_on_low_success_rate(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "bad-skill", use_count=10, success_count=2)  # 20%
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["bad-skill"], self._cfg(), triggered
        )
        assert not suggestions, "very low success rate handled by deprecation, not trigger"

    def test_no_trigger_below_min_uses(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "new-skill", use_count=2, success_count=1)  # 50% but < min_uses
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["new-skill"], self._cfg(), triggered
        )
        assert not suggestions, "below min_uses threshold"

    def test_fires_only_once(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "once-skill", use_count=10, success_count=5)
        triggered: set[str] = set()

        suggestions1 = check_skill_reflection_triggers(
            hook.db, ["once-skill"], self._cfg(), triggered
        )
        suggestions2 = check_skill_reflection_triggers(
            hook.db, ["once-skill"], self._cfg(), triggered
        )

        assert suggestions1, "should trigger on first check"
        assert not suggestions2, "should NOT trigger again (already in triggered set)"

    def test_boundary_at_30_pct(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "boundary-skill", use_count=10, success_count=3)  # exactly 30%
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["boundary-skill"], self._cfg(), triggered
        )
        assert suggestions, "exactly 30% should trigger (inclusive lower bound)"

    def test_boundary_at_80_pct(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        _seed_skill(hook, "upper-skill", use_count=10, success_count=8)  # exactly 80%
        triggered: set[str] = set()

        suggestions = check_skill_reflection_triggers(
            hook.db, ["upper-skill"], self._cfg(), triggered
        )
        assert suggestions, "exactly 80% should trigger (inclusive upper bound)"


class TestReflectionCoordinatorQueueing:
    def test_queue_and_take(self, tmp_path):
        from nano_hermes.coordinator.reflection import ReflectionCoordinator
        from nano_hermes.config import NanoHermesConfig
        from unittest.mock import MagicMock

        coord = ReflectionCoordinator(
            db=MagicMock(),
            config=NanoHermesConfig(),
            embedder_factory=MagicMock(),
        )
        coord.queue_skill_suggestions(["tip 1", "tip 2"])
        msg = coord.take_skill_suggestions()

        assert msg is not None
        assert msg["role"] == "system"
        assert "tip 1" in msg["content"]
        assert "tip 2" in msg["content"]

    def test_take_returns_none_when_empty(self, tmp_path):
        from nano_hermes.coordinator.reflection import ReflectionCoordinator
        from nano_hermes.config import NanoHermesConfig
        from unittest.mock import MagicMock

        coord = ReflectionCoordinator(
            db=MagicMock(),
            config=NanoHermesConfig(),
            embedder_factory=MagicMock(),
        )
        assert coord.take_skill_suggestions() is None

    def test_clears_after_take(self, tmp_path):
        from nano_hermes.coordinator.reflection import ReflectionCoordinator
        from nano_hermes.config import NanoHermesConfig
        from unittest.mock import MagicMock

        coord = ReflectionCoordinator(
            db=MagicMock(),
            config=NanoHermesConfig(),
            embedder_factory=MagicMock(),
        )
        coord.queue_skill_suggestions(["tip"])
        coord.take_skill_suggestions()
        assert coord.take_skill_suggestions() is None
