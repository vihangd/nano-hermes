"""Integration tests for session-archive secret redaction.

The previously-shipped redaction (PR 1) covered content the agent writes
through tools (skills, memory, reflections). PR 3 closes the second
durable surface: messages that flow through the loop are archived to the
``chunks`` table AND embedded for semantic search. Both paths now use
the redacted text — never the raw — so tool output / user pastes /
assistant echoes containing secrets don't land on disk or cross the
embedding provider's wire.
"""
from __future__ import annotations

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes

from conftest import _unset_embedding_keys


class TestArchiveRedaction:
    @pytest.mark.asyncio
    async def test_tool_message_redacted_in_chunks(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages = [
            {"role": "user", "content": "show env"},
            {"role": "tool", "content": "API_KEY=sk-ant-leaked1234567890abc"},
        ]
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=messages)
        )

        rows = hook.db.execute(
            "SELECT role, content FROM chunks ORDER BY id"
        ).fetchall()
        roles = [r[0] for r in rows]
        assert "tool" in roles, "tool message should be archived"
        tool_content = next(c for r, c in rows if r == "tool")
        assert "sk-ant-leaked1234567890abc" not in tool_content
        assert "█" in tool_content

    @pytest.mark.asyncio
    async def test_user_message_redacted(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages = [
            {"role": "user", "content": "use ghp_userpasted1234567890 please"},
        ]
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=messages)
        )

        stored = hook.db.execute(
            "SELECT content FROM chunks WHERE role='user'"
        ).fetchone()[0]
        assert "ghp_userpasted1234567890" not in stored
        assert "█" in stored

    @pytest.mark.asyncio
    async def test_clean_message_unchanged(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages = [{"role": "user", "content": "what's the weather"}]
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=messages)
        )

        stored = hook.db.execute(
            "SELECT content FROM chunks WHERE role='user'"
        ).fetchone()[0]
        assert stored == "what's the weather"

    @pytest.mark.asyncio
    async def test_disabled_via_config(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """When ``redact_secrets=False``, raw archive content is preserved."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop, config={"redact_secrets": False})

        messages = [
            {"role": "tool", "content": "key=sk-ant-rawarchive1234567890"},
        ]
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=messages)
        )

        stored = hook.db.execute(
            "SELECT content FROM chunks WHERE role='tool'"
        ).fetchone()[0]
        assert "sk-ant-rawarchive1234567890" in stored

    @pytest.mark.asyncio
    async def test_embedded_text_is_redacted(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """The embedding chain receives the redacted text, not the raw —
        no plaintext secret crosses the embedding provider's wire.
        """
        captured: list[str] = []

        async def fake_embed(self, texts):
            captured.extend(texts)
            import numpy as np
            return [np.zeros(512, dtype=np.float32) for _ in texts]

        _unset_embedding_keys(monkeypatch)
        monkeypatch.setattr(
            "nano_hermes.embedding.chain.EmbeddingChain.embed",
            fake_embed,
        )

        hook = nano_hermes.install(loop)
        messages = [
            {"role": "tool", "content": "GH_TOKEN=ghp_embedleak1234567890abc"},
        ]
        await hook.after_iteration(
            AgentHookContext(iteration=0, messages=messages)
        )
        await hook.archiver.drain(timeout=2.0)

        assert captured, "embed never called"
        assert all("ghp_embedleak1234567890abc" not in t for t in captured)
        assert any("█" in t for t in captured)
