"""Tests for propose_skill rollback behaviour on write/DB failures.

In create mode, if any part of the write sequence raises an exception and
the directory did not exist before the call, the whole skill directory is
rmtree'd so the agent can retry cleanly. In edit mode we preserve existing
content — a bad write must never destroy user data.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import _make_loop

import nano_hermes
from nano_hermes.skills.propose_tool import ProposeSkillTool


class _FailingDB:
    """Thin proxy around a real sqlite3.Connection that raises on matching SQL.

    sqlite3.Connection has C-level slots so its `execute` attribute is
    read-only — we can't monkeypatch it directly. This wrapper intercepts
    `execute` while delegating everything else (including context-manager
    protocol) to the real connection.
    """

    def __init__(self, real, fail_predicate):
        self._real = real
        self._fail_predicate = fail_predicate

    def execute(self, sql, *args, **kwargs):
        if self._fail_predicate(sql):
            raise RuntimeError(f"simulated db failure on: {sql[:40]}...")
        return self._real.execute(sql, *args, **kwargs)

    def __enter__(self):
        return self._real.__enter__()

    def __exit__(self, *a):
        return self._real.__exit__(*a)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _make_tool(tmp_path: Path) -> tuple[ProposeSkillTool, object]:
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    tool = ProposeSkillTool(hook=hook)
    return tool, hook


class TestCreateModeRollback:
    @pytest.mark.asyncio
    async def test_create_rollback_on_db_failure(self, tmp_path):
        """If the DB upsert fails, the partially-written skill dir is removed."""
        tool, hook = _make_tool(tmp_path)

        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: "INSERT INTO skill_stats" in sql,
        )

        try:
            result = await tool.execute(
                name="db-fail-skill",
                description="Skill whose DB upsert will fail.",
                body="Body.",
                files=[{"path": "scripts/run.py", "content": "x = 1"}],
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        assert "Rolled back" in result
        # Directory must be gone — create mode rollback.
        assert not (tmp_path / "skills" / "db-fail-skill").exists()

    @pytest.mark.asyncio
    async def test_create_rollback_on_write_failure(self, tmp_path):
        """If a companion file write raises, rollback removes the directory.

        Patches the imported ``atomic_write_text`` reference inside the
        propose_tool module (NOT the source in nano_hermes._atomic) — calls
        in propose_tool resolve through propose_tool's own namespace.
        """
        tool, hook = _make_tool(tmp_path)

        import nano_hermes.skills.propose_tool as pt_mod

        original = pt_mod.atomic_write_text
        call_count = {"n": 0}

        def failing_atomic_write(file_path, content):
            call_count["n"] += 1
            if call_count["n"] >= 2:
                raise OSError("simulated disk error")
            return original(file_path, content)

        with patch.object(pt_mod, "atomic_write_text", failing_atomic_write):
            result = await tool.execute(
                name="write-fail-skill",
                description="Skill whose companion write will fail.",
                body="Body.",
                files=[{"path": "scripts/bad.py", "content": "x = 1"}],
            )

        assert "Error" in result
        assert "Rolled back" in result
        assert not (tmp_path / "skills" / "write-fail-skill").exists()


class TestEditModePreservesData:
    @pytest.mark.asyncio
    async def test_edit_restores_skill_md_on_db_failure(self, tmp_path):
        """Edit mode must restore SKILL.md to its original content when the
        DB upsert fails after the file has already been overwritten.
        """
        tool, hook = _make_tool(tmp_path)

        # First, create a skill successfully.
        await tool.execute(
            name="preserve-me",
            description="A skill that must survive a bad edit.",
            body="v1 body",
        )
        skill_dir = tmp_path / "skills" / "preserve-me"
        assert (skill_dir / "SKILL.md").exists()

        # Promote to active so edit is valid.
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?", ("preserve-me",)
            )

        # Now force the edit's UPDATE to fail via our wrapping proxy.
        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: sql.strip().startswith(
                "UPDATE skill_stats SET content_hash"
            ),
        )

        try:
            result = await tool.execute(
                action="edit",
                name="preserve-me",
                description="A skill that must survive a bad edit.",
                body="v2 body",
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        assert "Rolled back" in result
        assert "original content restored" in result
        # Directory must still exist — edit mode never rmtrees.
        assert skill_dir.exists()
        assert (skill_dir / "SKILL.md").exists()
        # And the body content must be the original v1, not v2 — this is the
        # core promise of edit-mode snapshot/restore.
        text = (skill_dir / "SKILL.md").read_text()
        assert "v1 body" in text
        assert "v2 body" not in text

    @pytest.mark.asyncio
    async def test_create_rollback_preserves_preexisting_directory(self, tmp_path):
        """If the directory existed BEFORE the call (e.g. orphaned), create
        mode should not rmtree it on error — but this scenario is blocked by
        the orphaned-directory guard, so the test confirms the guard fires
        rather than the rollback deleting user data.
        """
        tool, hook = _make_tool(tmp_path)

        # Simulate an orphaned directory (on disk, not in DB).
        pre_existing = tmp_path / "skills" / "pre-existing"
        pre_existing.mkdir(parents=True)
        (pre_existing / "SKILL.md").write_text(
            "---\nname: pre-existing\ndescription: existing\n---\n\nOld body.\n"
        )

        result = await tool.execute(
            name="pre-existing",
            description="Would overwrite.",
            body="New body.",
        )

        # The orphaned-directory guard refuses before any rollback logic runs.
        assert "Error" in result
        assert "already exists on disk" in result
        # Original file untouched.
        assert (pre_existing / "SKILL.md").read_text().startswith("---")
        assert "Old body" in (pre_existing / "SKILL.md").read_text()


class TestEditModeSnapshotRestore:
    """Edit mode must restore each touched file to its prior content/state on
    failure. Snapshots cover SKILL.md, overwritten companion files, deleted
    files, and newly-added files.
    """

    @pytest.mark.asyncio
    async def test_edit_restores_companion_file_on_db_failure(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="comp-restore",
            description="Has a companion file.",
            body="v1 body",
            files=[{"path": "scripts/run.py", "content": "print('v1')\n"}],
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?", ("comp-restore",)
            )

        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: sql.strip().startswith(
                "UPDATE skill_stats SET content_hash"
            ),
        )

        try:
            result = await tool.execute(
                action="edit",
                name="comp-restore",
                description="Has a companion file.",
                body="v2 body",
                files=[{"path": "scripts/run.py", "content": "print('v2')\n"}],
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        assert "original content restored" in result
        # Both files must be at v1.
        skill_dir = tmp_path / "skills" / "comp-restore"
        assert "v1 body" in (skill_dir / "SKILL.md").read_text()
        assert (skill_dir / "scripts" / "run.py").read_text() == "print('v1')\n"

    @pytest.mark.asyncio
    async def test_edit_undeletes_file_on_db_failure(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="undelete",
            description="Has a file we'll try to delete.",
            body="v1 body",
            files=[{"path": "scripts/foo.py", "content": "keep me\n"}],
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?", ("undelete",)
            )

        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: sql.strip().startswith(
                "UPDATE skill_stats SET content_hash"
            ),
        )

        try:
            result = await tool.execute(
                action="edit",
                name="undelete",
                description="Has a file we'll try to delete.",
                body="v2 body",
                delete_files=["scripts/foo.py"],
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        # The deleted file must be back with original content.
        target = tmp_path / "skills" / "undelete" / "scripts" / "foo.py"
        assert target.exists()
        assert target.read_text() == "keep me\n"

    @pytest.mark.asyncio
    async def test_edit_removes_newly_added_file_on_db_failure(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="add-then-fail",
            description="No companion files yet.",
            body="v1 body",
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?",
                ("add-then-fail",),
            )

        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: sql.strip().startswith(
                "UPDATE skill_stats SET content_hash"
            ),
        )

        try:
            result = await tool.execute(
                action="edit",
                name="add-then-fail",
                description="Now adds a file.",
                body="v2 body",
                files=[{"path": "scripts/new.py", "content": "x = 1\n"}],
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        # The newly-added file must NOT exist — its snapshot was None.
        skill_dir = tmp_path / "skills" / "add-then-fail"
        assert not (skill_dir / "scripts" / "new.py").exists()
        # SKILL.md must be back at v1.
        assert "v1 body" in (skill_dir / "SKILL.md").read_text()


class TestAtomicWriteHygiene:
    @pytest.mark.asyncio
    async def test_atomic_write_leaves_no_tmp_file_on_success(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        await tool.execute(
            name="hygiene",
            description="Atomic-write smoke test.",
            body="Body.",
            files=[{"path": "scripts/run.py", "content": "x = 1\n"}],
        )
        skill_dir = tmp_path / "skills" / "hygiene"
        # No leftover .SKILL.md.tmp.* or .run.py.tmp.* files.
        leftover = list(skill_dir.rglob(".*.tmp.*"))
        assert leftover == [], f"unexpected tempfiles: {leftover}"


class TestDocstringPointer:
    def test_docstring_mentions_skill_creator(self):
        """Drift guard: ProposeSkillTool docstring must point at skill-creator."""
        doc = ProposeSkillTool.__doc__ or ""
        assert "skill-creator" in doc, (
            "ProposeSkillTool docstring must mention the skill-creator skill "
            "so first-time agents know to read it before authoring."
        )
