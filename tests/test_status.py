"""Tests for the nano_status tool."""
from __future__ import annotations

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes

from conftest import _unset_embedding_keys


# ---------------------------------------------------------------------------
# Phase 6: nano_status tool
# ---------------------------------------------------------------------------

class TestNanoStatus:
    async def test_nano_status_without_active_session(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)
        tool = loop.tools.get("nano_status")
        assert tool is not None
        out = await tool.execute()
        assert "session: none" in out

    async def test_nano_status_returns_structured_output(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        msgs: list[dict] = [{"role": "user", "content": "hello"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs))

        tool = loop.tools.get("nano_status")
        out = await tool.execute()
        assert "session:" in out
        assert "turns:" in out
        assert "salience:" in out
        assert "reflections:" in out
        assert "skills:" in out
        assert "db size:" in out
        # Session should be set now
        assert "session: none" not in out

    async def test_nano_status_skill_counts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        # Seed skills in each status
        for name, status in [("s1", "draft"), ("s2", "active"), ("s3", "deprecated")]:
            hook.db.execute(
                "INSERT INTO skill_stats (name, status, use_count, success_count) "
                "VALUES (?, ?, 0, 0)",
                (name, status),
            )
        hook.db.commit()

        tool = loop.tools.get("nano_status")
        out = await tool.execute()
        assert "1 draft" in out
        assert "1 active" in out
        assert "1 deprecated" in out
