"""Tests for skill activity telemetry (view tracking → curator anti-stale)."""
from __future__ import annotations

import time

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.curator import find_stale_skills


def _hook(tmp_path):
    return nano_hermes.install(_make_loop(tmp_path), config={})


def _seed(hook, name, *, use_count=5, last_used_age_days=0.0, last_viewed_age_days=None):
    now = time.time()
    lv = None if last_viewed_age_days is None else now - last_viewed_age_days * 86400
    hook.db.execute(
        "INSERT INTO skill_stats "
        "(name, status, use_count, last_used_at, last_viewed_at, origin, pinned) "
        "VALUES (?, 'active', ?, ?, ?, 'agent', 0)",
        (name, use_count, now - last_used_age_days * 86400, lv),
    )
    hook.db.commit()


class TestMarkViewed:
    def test_bumps_view_count_and_timestamp(self, tmp_path):
        hook = _hook(tmp_path)
        _seed(hook, "alpha")
        hook._mark_skills_viewed(["alpha"])
        row = hook.db.execute(
            "SELECT view_count, last_viewed_at FROM skill_stats WHERE name='alpha'"
        ).fetchone()
        assert row[0] == 1
        assert row[1] is not None

    def test_accepts_dict_of_loaded(self, tmp_path):
        hook = _hook(tmp_path)
        _seed(hook, "alpha")
        hook._mark_skills_viewed({"alpha": object()})  # dict → keys
        assert hook.db.execute(
            "SELECT view_count FROM skill_stats WHERE name='alpha'"
        ).fetchone()[0] == 1

    def test_empty_noop(self, tmp_path):
        hook = _hook(tmp_path)
        hook._mark_skills_viewed([])  # must not raise


class TestCuratorRespectsViews:
    def test_recently_viewed_not_stale(self, tmp_path):
        # Used 40d ago (dormant) but viewed 1d ago → NOT stale.
        hook = _hook(tmp_path)
        _seed(hook, "viewed", last_used_age_days=40, last_viewed_age_days=1)
        stale = find_stale_skills(hook.db, stale_after_days=30, min_uses=3)
        assert "viewed" not in [s.name for s in stale]

    def test_truly_dormant_still_stale(self, tmp_path):
        # Used 40d ago, viewed 40d ago → genuinely dormant → stale.
        hook = _hook(tmp_path)
        _seed(hook, "dormant", last_used_age_days=40, last_viewed_age_days=40)
        stale = find_stale_skills(hook.db, stale_after_days=30, min_uses=3)
        assert "dormant" in [s.name for s in stale]

    def test_never_viewed_uses_last_used(self, tmp_path):
        # last_viewed_at NULL → COALESCE to 0 → falls back to last_used_at.
        hook = _hook(tmp_path)
        _seed(hook, "old", last_used_age_days=40, last_viewed_age_days=None)
        stale = find_stale_skills(hook.db, stale_after_days=30, min_uses=3)
        assert "old" in [s.name for s in stale]
