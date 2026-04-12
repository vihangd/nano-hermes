"""Tests for session schema, search fallback, archiver, and session ended_at."""
from __future__ import annotations

import pytest
import numpy as np

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes
from nano_hermes.session.search import hybrid_search

from conftest import _existing_hook, _seed_chunk, _unset_embedding_keys


# ---------------------------------------------------------------------------
# Session schema — FTS5 trigger + sqlite-vec vec0 MATCH
# ---------------------------------------------------------------------------

class TestSessionSchema:
    def test_fts_trigger_mirrors_inserts(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        _seed_chunk(loop, "remember the spice melange")
        hook = _existing_hook(loop)

        rows = hook.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'melange'"
        ).fetchall()
        assert rows, "chunks_ai trigger did not mirror the insert into chunks_fts"

    def test_vec0_match_roundtrip(self, loop: AgentLoop) -> None:
        """Validates sqlite-vec vec0 MATCH/k syntax with a hand-crafted vector."""
        nano_hermes.install(loop)
        chunk_id = _seed_chunk(loop, "the agent can cook pasta")
        hook = _existing_hook(loop)

        dims = hook.config.embedding.target_dims
        vec = np.ones(dims, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        hook.db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vec.tobytes()),
        )
        hook.db.commit()

        hits = hybrid_search(hook.db, "pasta", vec, hook.config.retrieval)
        assert hits, "hybrid_search returned nothing — check vec0 MATCH syntax"
        assert hits[0].chunk_id == chunk_id


# ---------------------------------------------------------------------------
# session_search tool — FTS5 fallback path
# ---------------------------------------------------------------------------

class TestSessionSearchFallback:
    async def test_falls_back_to_fts_when_all_providers_unreachable(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Strip every provider key; the embedding chain will raise
        # AllProvidersFailed and the tool should degrade to FTS5-only.
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        nano_hermes.install(loop)
        _seed_chunk(loop, "spice melange caravan")

        tool = loop.tools.get("session_search")
        assert tool is not None
        out = await tool.execute(query="melange")
        assert "melange" in out, f"FTS fallback returned nothing: {out!r}"


# ---------------------------------------------------------------------------
# after_iteration archival path
# ---------------------------------------------------------------------------

class TestArchiver:
    """``after_iteration`` should persist new messages and keep FTS current."""

    async def test_archive_inserts_chunks_and_populates_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "what's the capital of Nauru"},
                {"role": "assistant", "content": "Yaren District is the de facto seat."},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        # drain the background embed task (will no-op because keys were unset)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT role, content FROM chunks ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("user", "what's the capital of Nauru")
        assert rows[1][0] == "assistant"
        assert "Yaren" in rows[1][1]

        matches = hook.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'Nauru'"
        ).fetchall()
        assert matches, "FTS trigger did not pick up archived chunks"

    async def test_tool_only_assistant_messages_are_skipped(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                # assistant fires a tool call with no text — nothing to archive
                {"role": "assistant", "content": None, "tool_calls": [{"name": "x"}]},
                {"role": "user", "content": "carry on"},
                # assistant with empty string — also skip
                {"role": "assistant", "content": "   "},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT role, content FROM chunks ORDER BY id"
        ).fetchall()
        assert rows == [("user", "carry on")]

    async def test_watermark_prevents_reinsert_on_second_iteration(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [
            {"role": "user", "content": "first question"},
        ]
        ctx1 = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx1)
        await hook.after_iteration(ctx1)

        # second iteration — same list, one new message appended
        messages.append({"role": "assistant", "content": "first answer"})
        ctx2 = AgentHookContext(iteration=1, messages=messages)
        await hook.after_iteration(ctx2)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT content FROM chunks ORDER BY id"
        ).fetchall()
        assert [r[0] for r in rows] == ["first question", "first answer"], (
            f"watermark leak — archived {len(rows)} rows: {rows}"
        )

        # sanity: exactly one session row (both iterations share the list id)
        session_count = hook.db.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        assert session_count == 1

    async def test_archived_content_is_findable_via_session_search(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: archive a turn, then retrieve it with session_search
        over the FTS fallback path (no embedding network needed)."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "planning a trip to Reykjavik"},
                {"role": "assistant", "content": "Reykjavik is great in winter."},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        await hook.archiver.drain()

        tool = loop.tools.get("session_search")
        assert tool is not None
        out = await tool.execute(query="Reykjavik")
        assert "Reykjavik" in out, f"search returned: {out!r}"


# ---------------------------------------------------------------------------
# Phase 5: sessions.ended_at bug fix
# ---------------------------------------------------------------------------

class TestSessionEndedAt:
    async def test_session_boundary_sets_ended_at(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        msgs1: list[dict] = [{"role": "user", "content": "first session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs1))
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs1))
        session1_id = hook.current_session_id
        assert session1_id is not None

        # New messages list → triggers session boundary
        msgs2: list[dict] = [{"role": "user", "content": "second session"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs2))

        ended_at = hook.db.execute(
            "SELECT ended_at FROM sessions WHERE id = ?", (session1_id,)
        ).fetchone()[0]
        assert ended_at is not None, "ended_at should be set when session boundary detected"

    async def test_purge_deletes_ended_sessions(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import time as _time
        from nano_hermes.session.db import purge_older_than

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Seed an old session with ended_at in the past
        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:ended", old_ts, old_ts),
        )
        old_id = cur.lastrowid
        hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, 0, 'user', 'old stuff', ?)",
            (old_id, old_ts),
        )
        hook.db.commit()

        purge_older_than(hook.db, days=30)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM sessions WHERE id = ?", (old_id,)
        ).fetchone()[0] == 0
        assert hook.db.execute(
            "SELECT COUNT(*) FROM chunks WHERE session_id = ?", (old_id,)
        ).fetchone()[0] == 0
