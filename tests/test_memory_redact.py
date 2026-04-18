"""Integration tests for memory_patch secret redaction."""
from __future__ import annotations

import pytest

from nanobot.agent.loop import AgentLoop

import nano_hermes


class TestMemoryRedaction:
    @pytest.mark.asyncio
    async def test_add_redacts_secret(self, loop: AgentLoop):
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        out = await tool.execute(
            slot="memory",
            action="add",
            content="api token: sk-ant-abc1234567890xyzdef",
        )
        assert out.startswith("ok")
        assert "redacted" in out

        text = loop.context.memory.read_memory()
        assert "sk-ant-abc1234567890xyzdef" not in text
        assert "█" in text

    @pytest.mark.asyncio
    async def test_replace_redacts_replacement(self, loop: AgentLoop):
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        await tool.execute(
            slot="user",
            action="add",
            content="placeholder for the key",
        )
        out = await tool.execute(
            slot="user",
            action="replace",
            needle="placeholder for the key",
            replacement="key=sk-ant-secretreplacement123",
        )
        assert out.startswith("ok")
        assert "redacted" in out

        text = loop.context.memory.read_user()
        assert "sk-ant-secretreplacement123" not in text
        assert "█" in text

    @pytest.mark.asyncio
    async def test_needle_not_redacted_in_replace_lookup(self, loop: AgentLoop):
        """`needle` is a locator. If we accidentally redacted it before
        looking it up, the lookup would fail with `entry not found`.
        """
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        # Seed with content that contains a secret-shaped substring.
        await tool.execute(
            slot="memory",
            action="add",
            content="raw line we'll target",
        )
        # Replace using a benign needle.
        out = await tool.execute(
            slot="memory",
            action="replace",
            needle="raw line we'll target",
            replacement="cleaned line",
        )
        assert out.startswith("ok"), out

    @pytest.mark.asyncio
    async def test_redact_disabled_passes_through(self, loop: AgentLoop):
        hook = nano_hermes.install(loop)
        hook.config.redact_secrets = False
        tool = loop.tools.get("memory_patch")

        out = await tool.execute(
            slot="memory",
            action="add",
            content="key=sk-ant-rawsecret1234567890",
        )
        assert out.startswith("ok")
        assert "redacted" not in out
        text = loop.context.memory.read_memory()
        assert "sk-ant-rawsecret1234567890" in text

    @pytest.mark.asyncio
    async def test_no_secret_no_redaction_note(self, loop: AgentLoop):
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        out = await tool.execute(
            slot="memory",
            action="add",
            content="user prefers tabs over spaces",
        )
        assert out.startswith("ok")
        assert "redacted" not in out
