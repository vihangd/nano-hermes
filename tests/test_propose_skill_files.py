"""Tests for propose_skill companion-file support (files + delete_files)."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import _make_loop

import nano_hermes
from nano_hermes.skills.propose_tool import ProposeSkillTool


def _make_tool(tmp_path: Path) -> tuple[ProposeSkillTool, object]:
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    tool = ProposeSkillTool(hook=hook)
    return tool, hook


class TestPathValidation:
    @pytest.mark.asyncio
    async def test_valid_scripts_path_accepted(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="my-skill",
            description="Does something useful.",
            body="## Usage\nRun the helper script.",
            files=[{"path": "scripts/run.py", "content": "print('hello')"}],
        )
        assert result.startswith("ok:")

    @pytest.mark.asyncio
    async def test_valid_references_path_accepted(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="ref-skill",
            description="Has a reference doc.",
            body="See references/api.md for details.",
            files=[{"path": "references/api.md", "content": "# API\n\nGET /foo"}],
        )
        assert result.startswith("ok:")

    @pytest.mark.asyncio
    async def test_valid_assets_path_accepted(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="asset-skill",
            description="Has a data file.",
            body="Load data from assets/data.csv.",
            files=[{"path": "assets/data.csv", "content": "a,b\n1,2\n"}],
        )
        assert result.startswith("ok:")

    @pytest.mark.asyncio
    async def test_path_traversal_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="bad-skill",
            description="Attempts path traversal.",
            body="Body.",
            files=[{"path": "scripts/../../../etc/passwd", "content": "root:x:0:0"}],
        )
        assert "Error" in result
        assert "invalid file path" in result

    @pytest.mark.asyncio
    async def test_root_level_file_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="bad-skill",
            description="Tries to write at root.",
            body="Body.",
            files=[{"path": "README.md", "content": "bad"}],
        )
        assert "Error" in result
        assert "invalid file path" in result

    @pytest.mark.asyncio
    async def test_nested_subdir_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="bad-skill",
            description="Nested subdir.",
            body="Body.",
            files=[{"path": "scripts/sub/deep.py", "content": "x = 1"}],
        )
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_duplicate_file_path_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="dup-skill",
            description="Has duplicate file.",
            body="Body.",
            files=[
                {"path": "scripts/run.py", "content": "a = 1"},
                {"path": "scripts/run.py", "content": "b = 2"},
            ],
        )
        assert "Error" in result
        assert "duplicate" in result


class TestFilesOnDisk:
    @pytest.mark.asyncio
    async def test_companion_files_written(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="with-files",
            description="Skill with companion files.",
            body="## Usage\nRun scripts/go.sh.",
            files=[
                {"path": "scripts/go.sh", "content": "#!/bin/bash\necho hi\n"},
                {"path": "references/notes.md", "content": "# Notes\n"},
            ],
        )
        skill_dir = tmp_path / "skills" / "with-files"
        assert (skill_dir / "SKILL.md").exists()
        assert (skill_dir / "scripts" / "go.sh").read_text() == "#!/bin/bash\necho hi\n"
        assert (skill_dir / "references" / "notes.md").read_text() == "# Notes\n"

    @pytest.mark.asyncio
    async def test_db_status_is_draft(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="draft-check",
            description="Check draft status.",
            body="Body.",
            files=[{"path": "scripts/run.py", "content": "x = 1"}],
        )
        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", ("draft-check",)
        ).fetchone()
        assert row is not None
        assert row[0] == "draft"

    @pytest.mark.asyncio
    async def test_security_block_in_file_stops_write(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        result = await tool.execute(
            name="evil-skill",
            description="Contains malicious script.",
            body="Body.",
            files=[
                {"path": "scripts/attack.sh", "content": "rm -rf /important"}
            ],
        )
        assert "Error" in result
        # Nothing should be written to disk.
        assert not (tmp_path / "skills" / "evil-skill").exists()


class TestSizeCap:
    @pytest.mark.asyncio
    async def test_oversized_payload_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        huge_content = "x" * (256 * 1024 + 1)
        result = await tool.execute(
            name="big-skill",
            description="Too large.",
            body=huge_content,
        )
        assert "Error" in result
        assert "too large" in result

    @pytest.mark.asyncio
    async def test_combined_size_counted(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        body = "a" * 200_000
        file_content = "b" * 100_000  # total > 256 KiB
        result = await tool.execute(
            name="combined-big",
            description="Combined size exceeds limit.",
            body=body,
            files=[{"path": "scripts/big.py", "content": file_content}],
        )
        assert "Error" in result
        assert "too large" in result


class TestEditWithFiles:
    @pytest.mark.asyncio
    async def test_edit_adds_new_file(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        # Create first.
        await tool.execute(
            name="edit-me",
            description="Initial.",
            body="v1",
        )
        # Promote to active so edit is valid.
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status = 'active' WHERE name = ?", ("edit-me",)
            )
        # Edit with a new companion file.
        result = await tool.execute(
            action="edit",
            name="edit-me",
            description="Updated.",
            body="v2",
            files=[{"path": "scripts/helper.py", "content": "def run(): pass\n"}],
        )
        assert result.startswith("ok:")
        assert (tmp_path / "skills" / "edit-me" / "scripts" / "helper.py").exists()

    @pytest.mark.asyncio
    async def test_edit_deletes_file(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        # Create with companion file.
        await tool.execute(
            name="delete-me",
            description="Has file to delete.",
            body="v1",
            files=[{"path": "scripts/old.py", "content": "old code\n"}],
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status = 'active' WHERE name = ?", ("delete-me",)
            )
        result = await tool.execute(
            action="edit",
            name="delete-me",
            description="Cleaned up.",
            body="v2",
            delete_files=["scripts/old.py"],
        )
        assert result.startswith("ok:")
        assert not (tmp_path / "skills" / "delete-me" / "scripts" / "old.py").exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_file_silently_ok(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="ghost-file",
            description="No companion files.",
            body="v1",
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status = 'active' WHERE name = ?", ("ghost-file",)
            )
        result = await tool.execute(
            action="edit",
            name="ghost-file",
            description="Edit.",
            body="v2",
            delete_files=["scripts/nonexistent.py"],
        )
        assert result.startswith("ok:")


class TestExistingDirWithNoDBRow:
    @pytest.mark.asyncio
    async def test_create_refuses_orphaned_directory(self, tmp_path):
        """Skill dir exists on disk (e.g. created via shell) but no DB row."""
        skill_dir = tmp_path / "skills" / "orphan"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: orphan\ndescription: x\n---\n\nBody.\n"
        )
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="orphan",
            description="New version.",
            body="Body.",
        )
        assert "Error" in result
        assert "already exists on disk" in result
