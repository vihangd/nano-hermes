"""Tests for Ratchet skill-cap + retirement (skill_retirement.py)."""
from __future__ import annotations

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.skill_retirement import _contribution_score, run_ratchet


def _hook(tmp_path, extra_config=None):
    cfg = {"skill_stats": {"ratchet_enabled": True, **(extra_config or {})}}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _add_skill(hook, name, *, use_count=0, success_count=0, status="active",
               origin="agent", pinned=0):
    hook.db.execute(
        "INSERT OR IGNORE INTO skill_stats "
        "(name, status, origin, use_count, success_count, pinned) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, status, origin, use_count, success_count, pinned),
    )
    hook.db.commit()


class TestContributionScore:
    def test_perfect_success(self):
        assert _contribution_score(10, 10) == 1.0

    def test_perfect_failure(self):
        assert _contribution_score(10, 0) == -1.0

    def test_mixed(self):
        # 6 success, 4 fail → (12 - 10)/10 = 0.2
        assert abs(_contribution_score(10, 6) - 0.2) < 1e-9

    def test_zero_uses(self):
        assert _contribution_score(0, 0) == 0.0

    def test_retirement_boundary(self):
        # ĉ = (2*40 - 100)/100 = -0.20 → below -0.10 threshold
        assert _contribution_score(100, 40) == pytest.approx(-0.20)


class TestRetirementDisabled:
    def test_disabled_by_default(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        _add_skill(hook, "bad_skill", use_count=200, success_count=40)
        result = run_ratchet(hook)
        assert result == {"retired": [], "cap_evicted": []}
        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'bad_skill'"
        ).fetchone()
        assert row[0] == "active"


class TestContributionScoreRetirement:
    def test_low_ĉ_skill_retired(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 10, "ratchet_retire_threshold": 0.10})
        # ĉ = (2*3 - 10)/10 = -0.40 → below -0.10
        _add_skill(hook, "bad", use_count=10, success_count=3)
        result = run_ratchet(hook)
        assert "bad" in result["retired"]
        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'bad'"
        ).fetchone()[0]
        assert status == "deprecated"

    def test_good_skill_untouched(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 10, "ratchet_retire_threshold": 0.10})
        # ĉ = (2*8 - 10)/10 = 0.6 → well above threshold
        _add_skill(hook, "good", use_count=10, success_count=8)
        result = run_ratchet(hook)
        assert "good" not in result["retired"]

    def test_below_n_min_not_retired(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 100, "ratchet_retire_threshold": 0.10})
        # ĉ = -0.40 but only 10 uses < 100 n_min
        _add_skill(hook, "new_bad", use_count=10, success_count=3)
        result = run_ratchet(hook)
        assert "new_bad" not in result["retired"]

    def test_pinned_skill_exempt(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 10, "ratchet_retire_threshold": 0.10})
        _add_skill(hook, "pinned_bad", use_count=10, success_count=0, pinned=1)
        result = run_ratchet(hook)
        assert "pinned_bad" not in result["retired"]

    def test_non_agent_origin_exempt(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 10, "ratchet_retire_threshold": 0.10})
        _add_skill(hook, "builtin_bad", use_count=10, success_count=0, origin="builtin")
        result = run_ratchet(hook)
        assert "builtin_bad" not in result["retired"]

    def test_already_deprecated_not_double_retired(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_n_min": 10, "ratchet_retire_threshold": 0.10})
        _add_skill(hook, "already_gone", use_count=10, success_count=0, status="deprecated")
        result = run_ratchet(hook)
        assert "already_gone" not in result["retired"]


class TestCapEnforcement:
    def test_cap_evicts_lowest_scoring(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_skill_cap": 2, "ratchet_n_min": 1000})
        # 3 active skills, cap=2 → lowest should be evicted
        _add_skill(hook, "best", use_count=10, success_count=9)
        _add_skill(hook, "mid", use_count=10, success_count=5)
        _add_skill(hook, "worst", use_count=10, success_count=1)
        result = run_ratchet(hook)
        assert "worst" in result["cap_evicted"]
        assert "best" not in result["cap_evicted"]
        assert len(result["cap_evicted"]) == 1

    def test_cap_not_triggered_when_within_limit(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_skill_cap": 10, "ratchet_n_min": 1000})
        _add_skill(hook, "s1", use_count=5, success_count=1)
        _add_skill(hook, "s2", use_count=5, success_count=1)
        result = run_ratchet(hook)
        assert result["cap_evicted"] == []

    def test_cap_evicts_only_agent_origin(self, tmp_path):
        hook = _hook(tmp_path, {"ratchet_skill_cap": 1, "ratchet_n_min": 1000})
        _add_skill(hook, "builtin_extra", use_count=1, success_count=0, origin="builtin")
        _add_skill(hook, "agent_extra", use_count=1, success_count=0, origin="agent")
        result = run_ratchet(hook)
        # builtin never evicted; agent_extra may be
        assert "builtin_extra" not in result["cap_evicted"]

    def test_retired_not_double_counted_in_cap(self, tmp_path):
        hook = _hook(tmp_path, {
            "ratchet_skill_cap": 1,
            "ratchet_n_min": 5,
            "ratchet_retire_threshold": 0.10,
        })
        # bad: ĉ = -0.40, eligible for retirement AND cap eviction
        _add_skill(hook, "bad", use_count=10, success_count=3)
        _add_skill(hook, "good", use_count=10, success_count=9)
        result = run_ratchet(hook)
        # bad should be in retired, NOT also in cap_evicted
        assert "bad" in result["retired"]
        assert "bad" not in result["cap_evicted"]


import pytest  # noqa: E402 (needed for pytest.approx above)
