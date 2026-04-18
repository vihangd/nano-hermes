"""Tests for the propose_skill action='patch' surgical find-and-replace path."""
from __future__ import annotations

from pathlib import Path

import pytest

from conftest import _make_loop

import nano_hermes
from nano_hermes.skills.propose_tool import ProposeSkillTool

# Reused from the rollback module — proxy that lets specific SQL statements raise.
from test_propose_skill_rollback import _FailingDB


def _make_tool(tmp_path: Path) -> tuple[ProposeSkillTool, object]:
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    tool = ProposeSkillTool(hook=hook)
    return tool, hook


async def _make_skill(tool: ProposeSkillTool, hook, name: str, body: str, files=None):
    """Helper: create + promote-to-active so patch will accept it."""
    await tool.execute(
        name=name,
        description=f"Test skill {name}.",
        body=body,
        files=files or [],
    )
    with hook.db:
        hook.db.execute(
            "UPDATE skill_stats SET status='active' WHERE name=?", (name,)
        )


class TestPatchSkillMd:
    @pytest.mark.asyncio
    async def test_patch_skill_md_updates_body(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "patch-body", "First version of the body.")

        result = await tool.execute(
            action="patch",
            name="patch-body",
            old_string="First version",
            new_string="Second version",
        )

        assert result.startswith("ok:"), result
        text = (tmp_path / "skills" / "patch-body" / "SKILL.md").read_text()
        assert "Second version of the body." in text
        assert "First version" not in text
        # Frontmatter intact.
        assert text.startswith("---\nname: patch-body\n")

    @pytest.mark.asyncio
    async def test_patch_skill_md_with_replace_all(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "ra-skill", "foo and foo and more foo.")

        result = await tool.execute(
            action="patch",
            name="ra-skill",
            old_string="foo",
            new_string="bar",
            replace_all=True,
        )

        assert result.startswith("ok:")
        assert "(3 replacements)" in result
        text = (tmp_path / "skills" / "ra-skill" / "SKILL.md").read_text()
        assert "bar and bar and more bar." in text
        assert "foo" not in text

    @pytest.mark.asyncio
    async def test_patch_unique_match_required_when_not_replace_all(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "multi-match", "foo and foo again.")

        result = await tool.execute(
            action="patch",
            name="multi-match",
            old_string="foo",
            new_string="bar",
        )
        assert "Error" in result
        assert "matches 2 times" in result
        # Original content untouched.
        assert "foo and foo again" in (
            tmp_path / "skills" / "multi-match" / "SKILL.md"
        ).read_text()

    @pytest.mark.asyncio
    async def test_patch_old_string_not_found(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "no-match", "Hello world.")

        result = await tool.execute(
            action="patch",
            name="no-match",
            old_string="nonexistent",
            new_string="anything",
        )
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_patch_invalidates_content_hash(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "hash-bust", "Patch will bust the hash.")

        # Manually set a non-NULL content_hash so we can detect the patch
        # nullifies it.
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET content_hash='abc123' WHERE name=?",
                ("hash-bust",),
            )

        result = await tool.execute(
            action="patch",
            name="hash-bust",
            old_string="bust the hash",
            new_string="bust it harder",
        )
        assert result.startswith("ok:")

        row = hook.db.execute(
            "SELECT content_hash FROM skill_stats WHERE name=?", ("hash-bust",)
        ).fetchone()
        assert row[0] is None, f"content_hash should be NULL after patch, got {row[0]!r}"

    @pytest.mark.asyncio
    async def test_patch_preserves_use_count(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "preserve-counts", "Body for counter test.")

        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET use_count=7, success_count=4 WHERE name=?",
                ("preserve-counts",),
            )

        result = await tool.execute(
            action="patch",
            name="preserve-counts",
            old_string="Body for",
            new_string="Updated body for",
        )
        assert result.startswith("ok:")

        row = hook.db.execute(
            "SELECT use_count, success_count FROM skill_stats WHERE name=?",
            ("preserve-counts",),
        ).fetchone()
        assert row == (7, 4)

    @pytest.mark.asyncio
    async def test_patch_no_op_rejected(self, tmp_path):
        """If old_string == new_string, the patch is a no-op — reject it."""
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "noop", "Body with x in it.")

        result = await tool.execute(
            action="patch",
            name="noop",
            old_string="x",
            new_string="x",
        )
        assert "Error" in result
        assert "no-op" in result


class TestPatchCompanionFile:
    @pytest.mark.asyncio
    async def test_patch_companion_file(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(
            tool,
            hook,
            "comp-patch",
            "Body.",
            files=[{"path": "scripts/run.py", "content": 'print("v1")\n'}],
        )

        result = await tool.execute(
            action="patch",
            name="comp-patch",
            file_path="scripts/run.py",
            old_string='"v1"',
            new_string='"v2"',
        )
        assert result.startswith("ok:")
        assert "scripts/run.py" in result

        target = tmp_path / "skills" / "comp-patch" / "scripts" / "run.py"
        assert target.read_text() == 'print("v2")\n'
        # SKILL.md untouched.
        assert "Body." in (tmp_path / "skills" / "comp-patch" / "SKILL.md").read_text()

    @pytest.mark.asyncio
    async def test_patch_invalid_file_path_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "bad-path", "Body.")

        result = await tool.execute(
            action="patch",
            name="bad-path",
            file_path="../../etc/passwd",
            old_string="x",
            new_string="y",
        )
        assert "Error" in result
        assert "invalid file_path" in result

    @pytest.mark.asyncio
    async def test_patch_nonexistent_file_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "ghost", "Body.")

        result = await tool.execute(
            action="patch",
            name="ghost",
            file_path="scripts/missing.py",
            old_string="x",
            new_string="y",
        )
        assert "Error" in result
        assert "does not exist" in result

    @pytest.mark.asyncio
    async def test_patch_root_level_file_rejected(self, tmp_path):
        """file_path='SKILL.md' is rejected — to patch SKILL.md, omit file_path."""
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "root-target", "Body.")

        result = await tool.execute(
            action="patch",
            name="root-target",
            file_path="SKILL.md",
            old_string="Body",
            new_string="NewBody",
        )
        assert "Error" in result
        assert "invalid file_path" in result

    @pytest.mark.asyncio
    async def test_patch_companion_symlink_escape_rejected(self, tmp_path):
        """A companion file that is a symlink pointing outside the skill dir
        must be refused. ``_resolve_companion`` calls ``.resolve()`` which
        follows symlinks, then ``relative_to(skill_dir)`` rejects escapes —
        this test pins that contract so a future refactor can't silently
        weaken it.
        """
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "sym-skill", "Body.")

        # Set up the symlink outside the skill dir, pointing to a real file
        # the agent should not be able to overwrite via patch.
        outside = tmp_path / "outside_target.txt"
        outside.write_text("sensitive content")
        skill_scripts = tmp_path / "skills" / "sym-skill" / "scripts"
        skill_scripts.mkdir()
        (skill_scripts / "escape").symlink_to(outside)

        result = await tool.execute(
            action="patch",
            name="sym-skill",
            file_path="scripts/escape",
            old_string="sensitive content",
            new_string="pwned",
        )
        assert "Error" in result
        assert "escapes skill directory" in result
        # The outside file must be untouched.
        assert outside.read_text() == "sensitive content"


class TestPatchValidation:
    @pytest.mark.asyncio
    async def test_patch_breaks_frontmatter_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "fm-break", "Body content.")

        # Patch out the frontmatter closing line.
        result = await tool.execute(
            action="patch",
            name="fm-break",
            old_string="---\nname: fm-break",
            new_string="name: fm-break",
        )
        assert "Error" in result
        assert "would break SKILL.md" in result
        # Original SKILL.md must be intact.
        text = (tmp_path / "skills" / "fm-break" / "SKILL.md").read_text()
        assert text.startswith("---\nname: fm-break\n")

    @pytest.mark.asyncio
    async def test_patch_security_scan_blocks(self, tmp_path):
        """Patching destructive content into SKILL.md must be blocked.

        SKILL.md goes through the full body scan (not the relaxed scripts
        scan), so 'rm -rf /' is rejected as a destructive shell command.
        """
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "sec-block", "Run the safe command.")

        result = await tool.execute(
            action="patch",
            name="sec-block",
            old_string="safe command",
            new_string="rm -rf /important",
        )
        assert "Error" in result
        assert "security scan" in result
        # SKILL.md must still contain the original "safe command".
        text = (tmp_path / "skills" / "sec-block" / "SKILL.md").read_text()
        assert "safe command" in text
        assert "rm -rf" not in text

    @pytest.mark.asyncio
    async def test_patch_size_cap_after_patch(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        # Create a skill near the cap. Use a body close to (but under) 256 KiB
        # so a small replacement that grows it pushes over.
        body = "filler " + ("x" * (256 * 1024 - 50))  # comfortably under cap
        await _make_skill(tool, hook, "near-cap", body)

        # Replace the small "filler " marker with something much larger to
        # push us over the limit.
        huge = "y" * 200
        result = await tool.execute(
            action="patch",
            name="near-cap",
            old_string="filler ",
            new_string=huge,
        )
        assert "Error" in result
        assert "too large" in result
        # Original SKILL.md untouched.
        text = (tmp_path / "skills" / "near-cap" / "SKILL.md").read_text()
        assert "filler " in text


class TestPatchLifecycle:
    @pytest.mark.asyncio
    async def test_patch_deprecated_skill_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "depr", "Body.")
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='deprecated' WHERE name=?", ("depr",)
            )

        result = await tool.execute(
            action="patch",
            name="depr",
            old_string="Body",
            new_string="NewBody",
        )
        assert "Error" in result
        assert "deprecated" in result

    @pytest.mark.asyncio
    async def test_patch_missing_skill_rejected(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            action="patch",
            name="ghost-skill",
            old_string="x",
            new_string="y",
        )
        assert "Error" in result
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_patch_draft_skill_allowed(self, tmp_path):
        """Patch should work on draft skills, not just active ones."""
        tool, hook = _make_tool(tmp_path)
        # Create but do NOT promote to active — leave as draft.
        await tool.execute(
            name="still-draft",
            description="Draft skill.",
            body="Draft body.",
        )
        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name=?", ("still-draft",)
        ).fetchone()
        assert row[0] == "draft"

        result = await tool.execute(
            action="patch",
            name="still-draft",
            old_string="Draft body.",
            new_string="Patched draft body.",
        )
        assert result.startswith("ok:")


class TestPatchRollback:
    @pytest.mark.asyncio
    async def test_patch_db_failure_restores_original(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "patch-rb", "Original body content.")

        real_db = hook.db
        hook.db = _FailingDB(
            real_db,
            fail_predicate=lambda sql: sql.strip().startswith(
                "UPDATE skill_stats SET content_hash"
            ),
        )

        try:
            result = await tool.execute(
                action="patch",
                name="patch-rb",
                old_string="Original body",
                new_string="Patched body",
            )
        finally:
            hook.db = real_db

        assert "Error" in result
        assert "Rolled back" in result
        # File must contain the original, not the patched, content.
        text = (tmp_path / "skills" / "patch-rb" / "SKILL.md").read_text()
        assert "Original body content." in text
        assert "Patched body" not in text


class TestPatchArgValidation:
    @pytest.mark.asyncio
    async def test_patch_missing_old_string_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "missing-old", "Body.")

        result = await tool.execute(
            action="patch",
            name="missing-old",
            new_string="anything",
        )
        assert "Error" in result
        assert "old_string is required" in result

    @pytest.mark.asyncio
    async def test_patch_missing_new_string_rejected(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "missing-new", "Body.")

        result = await tool.execute(
            action="patch",
            name="missing-new",
            old_string="Body",
        )
        assert "Error" in result
        assert "new_string is required" in result

    @pytest.mark.asyncio
    async def test_patch_empty_new_string_deletes_match(self, tmp_path):
        """new_string='' is a deletion — explicitly supported."""
        tool, hook = _make_tool(tmp_path)
        await _make_skill(tool, hook, "delete-text", "Body with REMOVEME inside.")

        result = await tool.execute(
            action="patch",
            name="delete-text",
            old_string="REMOVEME ",
            new_string="",
        )
        assert result.startswith("ok:")
        text = (tmp_path / "skills" / "delete-text" / "SKILL.md").read_text()
        assert "Body with inside." in text
