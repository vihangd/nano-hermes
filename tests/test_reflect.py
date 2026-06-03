"""Tests for salience heuristics, reflect tool, reflection injection, and global reflection."""
from __future__ import annotations

from pathlib import Path

import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop

import nano_hermes
from nano_hermes.reflect.salience import (
    correction_score,
    last_user_text,
    tool_burst_score,
)

from conftest import _patch_embedding, _unset_embedding_keys


# ---------------------------------------------------------------------------
# Reflexion: reflect tool + salience nudges + per-session injection
# ---------------------------------------------------------------------------

class TestSalienceHeuristics:
    """Pure-function unit tests for reflect/salience.py."""

    def test_tool_burst_fires_at_threshold(self) -> None:
        assert tool_burst_score(0) == 0.0
        assert tool_burst_score(4) == 0.0
        assert tool_burst_score(5) > 0.0
        assert tool_burst_score(10) == tool_burst_score(5)  # flat above threshold

    def test_correction_score_detects_pushback_phrases(self) -> None:
        assert correction_score(None) == 0.0
        assert correction_score("thanks, that worked") == 0.0
        assert correction_score("no, that's wrong") > 0.0
        assert correction_score("Actually, I wanted Celsius") > 0.0
        assert correction_score("Try again — doesn't work") > 0.0

    def test_last_user_text_flattens_content_blocks(self) -> None:
        msgs: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plain string"},
            {"role": "assistant", "content": "reply"},
        ]
        assert last_user_text(msgs) == "plain string"

        block_msgs: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
            },
        ]
        assert last_user_text(block_msgs) == "hello  world"
        assert last_user_text([]) is None


class TestReflectTool:
    async def test_reflect_stores_reflection_for_current_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Bootstrap a session via before_iteration
        ctx = AgentHookContext(
            iteration=0,
            messages=[{"role": "user", "content": "kick things off"}],
        )
        await hook.before_iteration(ctx)
        assert hook.current_session_id is not None
        session_id = hook.current_session_id

        tool = loop.tools.get("reflect")
        assert tool is not None
        out = await tool.execute(
            content="Next time, check the config file before editing code."
        )
        assert out.startswith("ok")

        rows = hook.db.execute(
            "SELECT session_id, content FROM reflections"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == session_id
        assert "config file" in rows[0][1]

    async def test_reflect_fails_without_active_session(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        # Do NOT call before_iteration. current_session_id stays None.
        assert hook.current_session_id is None

        tool = loop.tools.get("reflect")
        out = await tool.execute(content="this should not land anywhere")
        assert out.startswith("Error")
        assert "no active session" in out


class TestReflectionInjection:
    async def test_new_reflection_is_injected_on_next_iteration(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [{"role": "user", "content": "start"}]

        # Iteration 0 — bootstrap + write a reflection
        ctx0 = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx0)
        assert hook.current_session_id is not None

        reflect = loop.tools.get("reflect")
        await reflect.execute(content="Lesson: prefer simpler tools first.")

        # Simulate the LLM appending an assistant response, then end-of-iter
        messages.append({"role": "assistant", "content": "understood"})
        await hook.after_iteration(ctx0)

        # Iteration 1 — before_iteration should inject the new reflection
        ctx1 = AgentHookContext(iteration=1, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx1)
        assert len(messages) == before_len + 1
        injected = messages[-1]
        assert injected["role"] == "system"
        assert "Reflections from earlier" in injected["content"]
        assert "simpler tools" in injected["content"]

        # Iteration 2 — already-injected reflection should NOT be re-added
        messages.append({"role": "assistant", "content": "ok"})
        await hook.after_iteration(ctx1)
        ctx2 = AgentHookContext(iteration=2, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx2)
        assert len(messages) == before_len  # no new injection

    async def test_reflections_scoped_to_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reflections from session A must NOT leak into session B."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Session A
        msgs_a: list[dict] = [{"role": "user", "content": "session A start"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_a))
        session_a = hook.current_session_id
        assert session_a is not None

        reflect = loop.tools.get("reflect")
        await reflect.execute(content="Session A — secret lesson")
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        # Session B — a totally new messages list
        msgs_b: list[dict] = [{"role": "user", "content": "session B start"}]
        ctx_b = AgentHookContext(iteration=0, messages=msgs_b)
        before_len = len(msgs_b)
        await hook.before_iteration(ctx_b)
        session_b = hook.current_session_id
        assert session_b is not None and session_b != session_a

        # No reflections should have been injected — session B is empty
        # of reflections and session A's are off-limits.
        assert len(msgs_b) == before_len
        for msg in msgs_b:
            assert "secret lesson" not in str(msg.get("content", ""))


class TestSalienceNudge:
    async def test_errors_trigger_nudge_next_iteration(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        # Lower the threshold so a single error fires the nudge cleanly.
        hook = nano_hermes.install(loop, config={"reflection": {"threshold": 3.0}})

        messages: list[dict] = [{"role": "user", "content": "do the thing"}]
        ctx0 = AgentHookContext(iteration=0, messages=messages, error="boom")
        await hook.before_iteration(ctx0)
        await hook.after_iteration(ctx0)
        assert hook._nudge_pending is True

        # Iteration 1 — nudge should be injected and cleared
        ctx1 = AgentHookContext(iteration=1, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx1)
        assert hook._nudge_pending is False
        assert len(messages) == before_len + 1
        nudge = messages[-1]
        assert nudge["role"] == "system"
        assert "reflect(content" in nudge["content"]

    async def test_quiet_iterations_do_not_trigger_nudge(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [{"role": "user", "content": "easy question"}]
        ctx = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        assert hook._nudge_pending is False


# ---------------------------------------------------------------------------
# Phase 5: global reflection mode
# ---------------------------------------------------------------------------

class TestGlobalReflection:
    async def test_global_reflections_injected_cross_session(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reflections from a past session appear in a new session when scope=global."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(
            loop, config={"reflection_scope": "global"}
        )

        # Session 1: write a reflection and embed it
        msgs1: list[dict] = [{"role": "user", "content": "task about web scraping"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs1))
        session1_id = hook.current_session_id
        assert session1_id is not None

        # Insert a reflection and manually write its embedding
        import numpy as np
        cur = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) "
            "VALUES (?, ?, ?)",
            (session1_id, "Always check robots.txt before scraping", 1.0),
        )
        ref_id = cur.lastrowid
        hook.db.commit()

        # Write a fake embedding that matches what _fake_embed returns for
        # "scraping job with BeautifulSoup" (no keyword match → _FAKE_VEC_UNRELATED
        # = unit vector on dim 2). Using the same vector ensures distance ≈ 0
        # which passes the global_inject_min_similarity threshold.
        fake_vec = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
        fake_vec[2] = 1.0
        hook.db.execute(
            "INSERT INTO reflections_vec (reflection_id, embedding) VALUES (?, ?)",
            (ref_id, fake_vec.tobytes()),
        )
        hook.db.commit()

        # Session 2: new messages list triggers boundary detection
        msgs2: list[dict] = [{"role": "user", "content": "scraping job with BeautifulSoup"}]
        # Force session boundary by archiving the new list
        hook.archiver.archive_and_embed(msgs2)
        # before_iteration will see the new session and inject global reflections
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs2))

        # Verify the reflection was injected as a system message
        system_msgs = [m["content"] for m in msgs2 if m.get("role") == "system"]
        combined = " ".join(system_msgs)
        assert "robots.txt" in combined, (
            f"Expected cross-session reflection in system messages, got: {system_msgs}"
        )

    async def test_session_scope_ignores_other_sessions(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With reflection_scope='session' (default), other-session reflections are not injected."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)  # default: session scope
        assert hook.config.reflection_scope == "session"

        msgs1: list[dict] = [{"role": "user", "content": "task one"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs1))
        session1_id = hook.current_session_id

        import numpy as np
        cur = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) "
            "VALUES (?, ?, ?)",
            (session1_id, "Important lesson from session one", 1.0),
        )
        ref_id = cur.lastrowid
        hook.db.commit()
        fake_vec = np.ones(hook.config.embedding.target_dims, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        hook.db.execute(
            "INSERT INTO reflections_vec (reflection_id, embedding) VALUES (?, ?)",
            (ref_id, fake_vec.tobytes()),
        )
        hook.db.commit()

        msgs2: list[dict] = [{"role": "user", "content": "different task"}]
        hook.archiver.archive_and_embed(msgs2)
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs2))

        # No global injection should have happened (session scope)
        system_msgs = [m["content"] for m in msgs2 if m.get("role") == "system"]
        combined = " ".join(system_msgs)
        assert "Important lesson from session one" not in combined

    async def test_reflections_vec_table_exists(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reflections_vec vec0 table is created on open_db."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        # Should not raise
        rows = hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec"
        ).fetchone()
        assert rows[0] == 0

    async def test_reflect_tool_embeds_to_reflections_vec(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """ReflectTool with global scope writes an embedding to reflections_vec."""
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop, config={"reflection_scope": "global"})
        msgs: list[dict] = [{"role": "user", "content": "test task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))
        assert hook.current_session_id is not None

        tool = loop.tools.get("reflect")
        result = await tool.execute(content="When in doubt, check the docs first.")
        assert result.startswith("ok"), result

        # Await the scheduled embed task(s). The vec write now runs off the
        # event loop via asyncio.to_thread, so yielding a couple of ticks no
        # longer guarantees completion — wait on the real tasks instead.
        import asyncio
        if hook._reflection_embed_tasks:
            await asyncio.gather(*hook._reflection_embed_tasks)

        row_count = hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec"
        ).fetchone()[0]
        assert row_count == 1, (
            f"Expected 1 row in reflections_vec after global reflect, got {row_count}"
        )

    async def test_purge_cleans_reflections_vec_orphans(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """purge_older_than removes orphan rows from reflections_vec."""
        import time as _time
        import numpy as np
        from nano_hermes.session.db import purge_older_than

        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        old_ts = _time.time() - 60 * 86400
        cur = hook.db.execute(
            "INSERT INTO sessions (session_key, started_at, ended_at) VALUES (?, ?, ?)",
            ("old:vec:cleanup", old_ts, old_ts),
        )
        old_session_id = cur.lastrowid

        cur2 = hook.db.execute(
            "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
            (old_session_id, "old reflection content", old_ts),
        )
        old_ref_id = cur2.lastrowid

        fake_vec = np.ones(hook.config.embedding.target_dims, dtype=np.float32)
        fake_vec /= np.linalg.norm(fake_vec)
        hook.db.execute(
            "INSERT INTO reflections_vec (reflection_id, embedding) VALUES (?, ?)",
            (old_ref_id, fake_vec.tobytes()),
        )
        hook.db.commit()

        assert hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec WHERE reflection_id = ?",
            (old_ref_id,),
        ).fetchone()[0] == 1

        purge_older_than(hook.db, days=30)

        assert hook.db.execute(
            "SELECT COUNT(*) FROM reflections_vec WHERE reflection_id = ?",
            (old_ref_id,),
        ).fetchone()[0] == 0, "reflections_vec orphan not cleaned by purge_older_than"

    async def test_edit_preserves_status(
        self, loop: AgentLoop, tmp_path: Path
    ) -> None:
        """propose_skill(action='edit') keeps the existing status (draft or active)."""
        hook = nano_hermes.install(loop)
        tool = loop.tools.get("propose_skill")

        # Create skill and manually promote it to active
        await tool.execute(name="active-skill", description="original", body="body")
        hook.db.execute(
            "UPDATE skill_stats SET status = 'active' WHERE name = ?",
            ("active-skill",),
        )
        hook.db.commit()

        # Edit should keep status='active'
        out = await tool.execute(
            action="edit",
            name="active-skill",
            description="revised description",
            body="new body content",
        )
        assert out.startswith("ok"), out

        row = hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?",
            ("active-skill",),
        ).fetchone()
        assert row[0] == "active", (
            f"status should remain 'active' after edit, got '{row[0]}'"
        )


# ---------------------------------------------------------------------------
# Phase 6: reflect empty content after strip
# ---------------------------------------------------------------------------

class TestReflectValidation:
    async def test_reflect_empty_after_strip_returns_error(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        msgs: list[dict] = [{"role": "user", "content": "task"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs))
        tool = loop.tools.get("reflect")
        out = await tool.execute(content="       ")
        assert "Error" in out
        assert "empty" in out.lower()


# ---------------------------------------------------------------------------
# Phase 2.5: recent_limit caps reflection injection
# ---------------------------------------------------------------------------

class TestRecentLimit:
    async def test_recent_limit_caps_reflection_injection(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(
            loop, config={"reflection": {"recent_limit": 3}}
        )

        messages: list[dict] = [{"role": "user", "content": "start"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=messages))
        session_id = hook.current_session_id
        assert session_id is not None

        import time as _time
        for i in range(6):
            hook.db.execute(
                "INSERT INTO reflections (session_id, content, created_at) VALUES (?, ?, ?)",
                (session_id, f"Reflection number {i}", _time.time()),
            )
        hook.db.commit()

        await hook.after_iteration(AgentHookContext(iteration=0, messages=messages))

        messages.append({"role": "assistant", "content": "reply"})
        before_len = len(messages)
        await hook.before_iteration(AgentHookContext(iteration=1, messages=messages))

        injected = [m for m in messages[before_len:] if m.get("role") == "system"]
        assert len(injected) == 1
        content = injected[0]["content"]
        bullet_count = content.count("\n- ")
        assert bullet_count == 3, f"expected 3 reflections, got {bullet_count}: {content}"
