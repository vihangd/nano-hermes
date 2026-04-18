"""Integration tests for atomic memory writes.

Confirms BudgetedMemory.write goes through nano_hermes._atomic.atomic_write_text
for all three slots (memory/user/soul) and that a mid-write failure preserves
the prior MEMORY.md content rather than corrupting it.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from nanobot.agent.loop import AgentLoop

import nano_hermes
import nano_hermes.memory.budgets as budgets_mod


class TestMemoryAtomicWrites:
    @pytest.mark.asyncio
    async def test_add_no_tmp_file_leftover(self, loop: AgentLoop):
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        await tool.execute(slot="memory", action="add", content="user prefers tabs")
        memory_dir = loop.context.memory.memory_file.parent
        leftover = list(memory_dir.glob(".MEMORY.md.tmp.*"))
        assert leftover == []

    @pytest.mark.asyncio
    async def test_add_failure_preserves_original_memory(self, loop: AgentLoop):
        """A simulated atomic-write failure during a second add must leave
        the prior MEMORY.md content intact.
        """
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        await tool.execute(slot="memory", action="add", content="kept entry")

        def boom(*a, **kw):
            raise OSError("simulated")

        with patch.object(budgets_mod, "atomic_write_text", boom):
            out = await tool.execute(
                slot="memory", action="add", content="should not survive"
            )

        assert "Error" in out
        text = loop.context.memory.read_memory()
        assert "kept entry" in text
        assert "should not survive" not in text

    @pytest.mark.asyncio
    async def test_user_slot_uses_atomic(self, loop: AgentLoop):
        """Confirms the slot→file dict routes USER.md correctly."""
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        await tool.execute(slot="user", action="add", content="prefers spaces")
        assert "prefers spaces" in loop.context.memory.read_user()
        user_dir = loop.context.memory.user_file.parent
        leftover = list(user_dir.glob(".USER.md.tmp.*"))
        assert leftover == []

    @pytest.mark.asyncio
    async def test_soul_slot_uses_atomic(self, loop: AgentLoop):
        """Confirms the slot→file dict routes SOUL.md correctly."""
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        await tool.execute(slot="soul", action="add", content="be kind")
        assert "be kind" in loop.context.memory.read_soul()
        soul_dir = loop.context.memory.soul_file.parent
        leftover = list(soul_dir.glob(".SOUL.md.tmp.*"))
        assert leftover == []

    @pytest.mark.asyncio
    async def test_overwrites_existing_memory_atomically(self, loop: AgentLoop):
        """Successive adds rewrite MEMORY.md cleanly via atomic_write_text;
        no `.MEMORY.md.tmp.*` accumulates between calls.
        """
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        await tool.execute(slot="memory", action="add", content="first")
        await tool.execute(slot="memory", action="add", content="second")
        await tool.execute(slot="memory", action="add", content="third")
        text = loop.context.memory.read_memory()
        for entry in ("first", "second", "third"):
            assert entry in text
        memory_dir = loop.context.memory.memory_file.parent
        assert list(memory_dir.glob(".MEMORY.md.tmp.*")) == []
