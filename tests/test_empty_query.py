"""Tests for empty query guards on search tools."""
from __future__ import annotations

import pytest

from nanobot.agent.loop import AgentLoop

import nano_hermes

from conftest import _unset_embedding_keys


# ---------------------------------------------------------------------------
# Phase 6: empty query guards on search tools
# ---------------------------------------------------------------------------

class TestEmptyQueryGuards:
    async def test_session_search_empty_query(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)
        tool = loop.tools.get("session_search")
        out = await tool.execute(query="")
        assert "Error" in out
        assert "empty" in out.lower()

    async def test_trajectory_search_empty_query(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)
        tool = loop.tools.get("trajectory_search")
        out = await tool.execute(query="")
        assert "Error" in out
        assert "empty" in out.lower()

    async def test_skill_search_empty_query(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)
        tool = loop.tools.get("skill_search")
        out = await tool.execute(query="")
        assert "Error" in out
        assert "empty" in out.lower()
