"""Tests for the curator — stale-skill archival with cooldown and audit trail."""
from __future__ import annotations

import time

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.curator import (
    _META_LAST_RUN,
    archive_skill,
    find_stale_skills,
    meta_get,
    meta_set,
    run_curator,
    should_run,
)


def _seed_skill(
    hook,
    name: str,
    *,
    status: str = "active",
    use_count: int = 5,
    last_used_age_days: float = 0,
    origin: str = "agent",
    pinned: int = 0,
):
    now = time.time()
    # Default origin='agent' (auto-evolvable, created via propose_skill); only
    # such skills are eligible for curator archival.
    hook.db.execute(
        "INSERT INTO skill_stats (name, status, use_count, last_used_at, origin, pinned) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, status, use_count, now - last_used_age_days * 86400, origin, pinned),
    )
    hook.db.commit()
    skill_dir = hook.workspace / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(f"# {name}\n## Steps\n1. body")


def _make_hook(tmp_path, **overrides):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop, config={"skill_stats": overrides} if overrides else None)


class TestMetaKV:
    def test_set_get_roundtrip(self, tmp_path):
        hook = _make_hook(tmp_path)
        assert meta_get(hook.db, "missing") is None
        meta_set(hook.db, "k", "v1")
        assert meta_get(hook.db, "k") == "v1"
        meta_set(hook.db, "k", "v2")  # upsert
        assert meta_get(hook.db, "k") == "v2"


class TestFindStaleSkills:
    def test_finds_old_active_skill(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "old-skill", use_count=10, last_used_age_days=40)
        stale = find_stale_skills(hook.db, stale_after_days=30, min_uses=3)
        assert [s.name for s in stale] == ["old-skill"]

    def test_skips_recently_used(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "fresh", use_count=10, last_used_age_days=5)
        assert find_stale_skills(hook.db, stale_after_days=30, min_uses=3) == []

    def test_skips_low_use_count(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "untested", use_count=1, last_used_age_days=100)
        assert find_stale_skills(hook.db, stale_after_days=30, min_uses=3) == []

    def test_skips_non_active(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "draft-skill", status="draft", use_count=10, last_used_age_days=40)
        _seed_skill(hook, "deprecated-skill", status="deprecated", use_count=10, last_used_age_days=40)
        assert find_stale_skills(hook.db, stale_after_days=30, min_uses=3) == []

    def test_zero_days_disabled(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "old", use_count=10, last_used_age_days=100)
        assert find_stale_skills(hook.db, stale_after_days=0, min_uses=3) == []


class TestArchiveSkill:
    def test_sets_deprecated_and_writes_version(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook, "to-archive", use_count=5)
        archive_skill(hook.db, "to-archive", current_body="# to-archive\n...")
        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("to-archive",)
        ).fetchone()[0]
        assert status == "deprecated"
        version = hook.db.execute(
            "SELECT body, reason FROM skill_versions WHERE skill_name = ?",
            ("to-archive",),
        ).fetchone()
        assert version is not None
        assert version[1] == "curator: archived"
        assert "to-archive" in version[0]


class TestLifecycle:
    def test_run_curator_marks_active_stale_not_deprecated(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_stale_after_days=30,
            curator_archive_after_days=90,
            curator_min_uses=3,
            curator_cooldown_hours=24,
        )
        _seed_skill(hook, "dormant", use_count=10, last_used_age_days=45)
        touched = run_curator(hook)
        assert touched == ["dormant"]
        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'dormant'"
        ).fetchone()[0]
        assert status == "stale"  # not deprecated yet

    def test_stale_skill_deprecated_after_archive_window(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_stale_after_days=30,
            curator_archive_after_days=90,
            curator_min_uses=3,
            curator_cooldown_hours=0,  # allow repeated runs
        )
        # Dormant 120d: first run stales it; second run (still dormant) archives.
        _seed_skill(hook, "ancient", use_count=10, last_used_age_days=120)
        run_curator(hook)
        assert hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'ancient'"
        ).fetchone()[0] == "stale"
        run_curator(hook)
        assert hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = 'ancient'"
        ).fetchone()[0] == "deprecated"

    def test_stale_skill_still_searchable(self, tmp_path):
        """Stale skills are demoted, not hidden — search filters only 'deprecated'."""
        hook = _make_hook(tmp_path, curator_stale_after_days=30, curator_min_uses=3)
        _seed_skill(hook, "dormant", use_count=10, last_used_age_days=45)
        run_curator(hook)
        # The skill-search candidate query excludes only deprecated rows.
        rows = hook.db.execute(
            "SELECT name FROM skill_stats WHERE status != 'deprecated' AND name = 'dormant'"
        ).fetchall()
        assert rows == [("dormant",)]


class TestShouldRun:
    def test_first_run_returns_true(self, tmp_path):
        hook = _make_hook(tmp_path)
        assert should_run(hook.db, 24) is True

    def test_within_cooldown_returns_false(self, tmp_path):
        hook = _make_hook(tmp_path)
        meta_set(hook.db, _META_LAST_RUN, str(time.time()))
        assert should_run(hook.db, 24) is False

    def test_after_cooldown_returns_true(self, tmp_path):
        hook = _make_hook(tmp_path)
        meta_set(hook.db, _META_LAST_RUN, str(time.time() - 25 * 3600))
        assert should_run(hook.db, 24) is True

    def test_zero_cooldown_always_runs(self, tmp_path):
        hook = _make_hook(tmp_path)
        meta_set(hook.db, _META_LAST_RUN, str(time.time()))
        assert should_run(hook.db, 0) is True


class TestRunCurator:
    def test_archives_stale_skills(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_stale_after_days=30,
            curator_min_uses=3,
            curator_cooldown_hours=24,
        )
        _seed_skill(hook, "stale-a", use_count=10, last_used_age_days=45)
        _seed_skill(hook, "stale-b", use_count=10, last_used_age_days=60)
        _seed_skill(hook, "fresh", use_count=10, last_used_age_days=1)
        archived = run_curator(hook)
        assert set(archived) == {"stale-a", "stale-b"}
        # Marks cooldown.
        assert meta_get(hook.db, _META_LAST_RUN) is not None

    def test_skips_when_cooldown_active(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_stale_after_days=30,
            curator_min_uses=3,
            curator_cooldown_hours=24,
        )
        _seed_skill(hook, "stale", use_count=10, last_used_age_days=45)
        meta_set(hook.db, _META_LAST_RUN, str(time.time()))
        archived = run_curator(hook)
        assert archived == []
        # Skill remains active.
        status = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("stale",)
        ).fetchone()[0]
        assert status == "active"

    def test_disabled_returns_empty(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_enabled=False,
            curator_stale_after_days=30,
            curator_min_uses=3,
        )
        _seed_skill(hook, "stale", use_count=10, last_used_age_days=45)
        assert run_curator(hook) == []
        # No cooldown mark either.
        assert meta_get(hook.db, _META_LAST_RUN) is None

    def test_archived_skill_has_audit_trail(self, tmp_path):
        hook = _make_hook(
            tmp_path,
            curator_stale_after_days=30,
            curator_min_uses=3,
            curator_cooldown_hours=24,
        )
        _seed_skill(hook, "lapsed", use_count=10, last_used_age_days=90)
        run_curator(hook)
        version = hook.db.execute(
            "SELECT reason, body FROM skill_versions WHERE skill_name = ?", ("lapsed",)
        ).fetchone()
        assert version is not None
        assert version[0] == "curator: stale"
        assert "lapsed" in version[1]
