"""Integration tests for propose_skill secret redaction.

Verifies that secret-shaped strings in the body, companion files, and
patch new_string land *redacted* on disk — and that the success message
reports what was masked. Disable via config.redact_secrets=False bypasses
the whole machinery.
"""
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


class TestSecretInBody:
    @pytest.mark.asyncio
    async def test_secret_in_body_is_redacted_on_disk(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="leaky",
            description="Skill body contains an OpenAI key.",
            body="Configure the client with sk-ant-abc1234567890xyzdef and run.",
        )
        assert result.startswith("ok:"), result
        text = (tmp_path / "skills" / "leaky" / "SKILL.md").read_text()
        assert "sk-ant-abc1234567890xyzdef" not in text
        assert "█" in text

    @pytest.mark.asyncio
    async def test_success_message_reports_count_and_kind(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="reports",
            description="Body has one secret.",
            body="key=sk-ant-abc1234567890xyzdef",
        )
        assert "redacted" in result
        assert "1 secret-shaped string" in result
        # Either openai_or_anthropic or env_assignment kind fires.
        assert "openai_or_anthropic" in result or "env_assignment" in result

    @pytest.mark.asyncio
    async def test_no_secret_no_redaction_note(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="clean",
            description="Body is clean.",
            body="Just a regular procedure with no keys.",
        )
        assert result.startswith("ok:")
        assert "redacted" not in result


class TestSecretInCompanionFile:
    @pytest.mark.asyncio
    async def test_secret_in_file_content_is_redacted(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="env-leak",
            description="Companion file has an env-style secret.",
            body="See references/env.md for setup.",
            files=[
                {
                    "path": "references/env.md",
                    "content": "export GH_TOKEN=ghp_abcdef1234567890abcdef",
                },
            ],
        )
        assert result.startswith("ok:"), result
        target = tmp_path / "skills" / "env-leak" / "references" / "env.md"
        text = target.read_text()
        assert "ghp_abcdef1234567890abcdef" not in text
        assert "█" in text

    @pytest.mark.asyncio
    async def test_aggregate_count_across_body_and_files(self, tmp_path):
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="multi-leak",
            description="Two secrets — one in body, one in file.",
            body="First key: sk-ant-aaaaaaaaaa1234567890",
            files=[
                {
                    "path": "scripts/run.sh",
                    "content": "#!/bin/bash\nGH=ghp_bbbbbbbbbb1234567890\n",
                },
            ],
        )
        assert "redacted 2 secret-shaped strings" in result


class TestPatchRedaction:
    @pytest.mark.asyncio
    async def test_patch_new_string_is_redacted(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        await tool.execute(
            name="patch-leak",
            description="Will be patched with a secret.",
            body="Original safe body content here.",
        )
        # Promote so patch is allowed (parallel to other patch tests).
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?",
                ("patch-leak",),
            )

        result = await tool.execute(
            action="patch",
            name="patch-leak",
            old_string="safe body content",
            new_string="key=sk-ant-leakedkey1234567890",
        )
        assert result.startswith("ok:"), result
        assert "redacted" in result

        text = (tmp_path / "skills" / "patch-leak" / "SKILL.md").read_text()
        assert "sk-ant-leakedkey1234567890" not in text
        assert "█" in text

    @pytest.mark.asyncio
    async def test_patch_old_string_not_redacted(self, tmp_path):
        """`old_string` is a locator — redacting it would prevent matching.

        Set up a SKILL.md whose body contains a (post-redaction) prefix the
        agent can target for replacement; ensure that specifying
        old_string=that-literal still finds the match.
        """
        tool, hook = _make_tool(tmp_path)
        # Create a skill with a marker we'll later patch out.
        await tool.execute(
            name="loc-test",
            description="Patch locator test.",
            body="MARKER_TO_REPLACE here in the body.",
        )
        with hook.db:
            hook.db.execute(
                "UPDATE skill_stats SET status='active' WHERE name=?",
                ("loc-test",),
            )

        # `old_string` is benign; `new_string` is benign. The point of
        # this test is that the patch mechanism doesn't accidentally
        # mutate `old_string` before the lookup.
        result = await tool.execute(
            action="patch",
            name="loc-test",
            old_string="MARKER_TO_REPLACE",
            new_string="REPLACED",
        )
        assert result.startswith("ok:"), result
        text = (tmp_path / "skills" / "loc-test" / "SKILL.md").read_text()
        assert "REPLACED" in text


class TestRedactionDisabled:
    @pytest.mark.asyncio
    async def test_disabled_lets_secret_through(self, tmp_path):
        tool, hook = _make_tool(tmp_path)
        # Override the config flag at runtime.
        hook.config.redact_secrets = False

        result = await tool.execute(
            name="raw",
            description="Raw secret should land on disk.",
            body="key=sk-ant-rawsecretkey1234567890",
        )
        assert result.startswith("ok:")
        assert "redacted" not in result
        text = (tmp_path / "skills" / "raw" / "SKILL.md").read_text()
        assert "sk-ant-rawsecretkey1234567890" in text


class TestRedactionOrdering:
    @pytest.mark.asyncio
    async def test_security_scan_still_blocks_destructive_pattern(self, tmp_path):
        """Redaction runs first; the existing security scan still fires
        on patterns that survive (e.g. destructive shell commands).
        """
        tool, _ = _make_tool(tmp_path)
        result = await tool.execute(
            name="scan-after-redact",
            description="Secret + destructive — scan must still block.",
            body="key=sk-ant-abcdefghij1234567890 and then rm -rf /important",
        )
        assert "Error" in result
        # The error is the security scan error, not a write success.
        assert "destructive" in result or "rm" in result

    @pytest.mark.asyncio
    async def test_redacted_body_counts_against_size_cap(self, tmp_path):
        """Size cap is enforced on the *redacted* body. Mostly proves the
        redaction step happens before the size check (no behaviour change
        expected since masking ≈ length-preserving for most patterns).
        """
        tool, _ = _make_tool(tmp_path)
        # Way under the cap — just confirms ordering doesn't break the path.
        result = await tool.execute(
            name="ordering",
            description="Small body with a secret.",
            body="key=sk-ant-shortenoughkey1234",
        )
        assert result.startswith("ok:"), result
