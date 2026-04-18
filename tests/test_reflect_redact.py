"""Integration tests for reflect tool secret redaction."""
from __future__ import annotations

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes

from conftest import _unset_embedding_keys


class TestReflectRedaction:
    @pytest.mark.asyncio
    async def test_reflect_content_redacted_in_db(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "go"}])
        )
        assert hook.current_session_id is not None

        tool = loop.tools.get("reflect")
        out = await tool.execute(
            content="Tried sk-ant-abc1234567890xyzdef but it failed; rotate the key.",
        )
        assert out.startswith("ok"), out
        assert "redacted" in out

        rows = hook.db.execute("SELECT content FROM reflections").fetchall()
        assert len(rows) == 1
        stored = rows[0][0]
        assert "sk-ant-abc1234567890xyzdef" not in stored
        assert "█" in stored

    @pytest.mark.asyncio
    async def test_reflect_no_secret_no_note(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "go"}])
        )

        tool = loop.tools.get("reflect")
        out = await tool.execute(content="Lesson: read the docs first.")
        assert out.startswith("ok")
        assert "redacted" not in out

    @pytest.mark.asyncio
    async def test_reflect_redact_disabled(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        hook.config.redact_secrets = False
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "go"}])
        )

        tool = loop.tools.get("reflect")
        out = await tool.execute(
            content="Use sk-ant-rawsecret1234567890 here.",
        )
        assert out.startswith("ok")
        assert "redacted" not in out

        stored = hook.db.execute("SELECT content FROM reflections").fetchone()[0]
        assert "sk-ant-rawsecret1234567890" in stored

    @pytest.mark.asyncio
    async def test_global_scope_embed_sees_redacted_text(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ):
        """Cross-session retrieval embeds reflections so future sessions can
        retrieve them by similarity. The text passed to embed() must be the
        redacted version — otherwise an embedded secret leaks via vector
        neighbours even if the on-disk DB row is masked.
        """
        import asyncio

        from nano_hermes.embedding.chain import EmbeddingChain

        captured_texts: list[str] = []

        async def capture_embed(self_inner, texts):
            captured_texts.extend(texts)
            import numpy as np
            dim = self_inner._config.target_dims
            return [np.zeros(dim, dtype=np.float32) for _ in texts]

        monkeypatch.setattr(EmbeddingChain, "embed", capture_embed)

        hook = nano_hermes.install(loop, config={"reflection_scope": "global"})
        await hook.before_iteration(
            AgentHookContext(iteration=0, messages=[{"role": "user", "content": "go"}])
        )

        tool = loop.tools.get("reflect")
        secret = "sk-ant-rawsecret1234567890abcdef"
        out = await tool.execute(content=f"Tried {secret}; rotate it.")
        assert out.startswith("ok"), out

        # Drain the background embed task scheduled by _schedule_embed.
        for _ in range(5):
            await asyncio.sleep(0)

        # capture_embed received the embedding text. None of the captured
        # strings may contain the raw secret.
        assert captured_texts, "embed was never invoked — global scope wiring broken"
        for t in captured_texts:
            assert secret not in t, (
                f"raw secret leaked into reflection embedding: {t!r}"
            )
