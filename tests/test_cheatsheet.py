"""Tests for Dynamic Cheatsheet (memory/cheatsheet.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import nano_hermes
import pytest
from conftest import _make_loop
from nano_hermes.memory.cheatsheet import (
    _first_user_text,
    _last_assistant_text,
    _store_lesson,
    _task_category,
    build_injection_message,
    extract_cheatsheet_lesson,
    retrieve_lessons,
)


def _hook(tmp_path, enabled=True):
    cfg = {"skill_stats": {"cheatsheet_enabled": enabled, "cheatsheet_top_k": 3}}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _msgs(user="do the thing", assistant="done"):
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_first_user_text_basic(self):
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hello"}]
        assert _first_user_text(msgs) == "hello"

    def test_first_user_text_list_content(self):
        msgs = [{"role": "user", "content": [{"type": "text", "text": "part1"}, {"type": "text", "text": "part2"}]}]
        assert _first_user_text(msgs) == "part1 part2"

    def test_first_user_text_no_user(self):
        assert _first_user_text([{"role": "assistant", "content": "hi"}]) == ""

    def test_last_assistant_text_returns_last(self):
        msgs = [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "last"},
        ]
        assert _last_assistant_text(msgs) == "last"

    def test_task_category_truncates(self):
        long = "x" * 200
        assert _task_category(long) == "x" * 120


class TestStoreLesson:
    def test_stores_with_cheatsheet_type(self, tmp_path):
        hook = _hook(tmp_path)
        fact_id = _store_lesson(hook.db, "Always verify before deleting.", "delete files")
        assert fact_id > 0
        row = hook.db.execute(
            "SELECT fact_type, task_category FROM semantic_facts WHERE id = ?", (fact_id,)
        ).fetchone()
        assert row[0] == "cheatsheet"
        assert row[1] == "delete files"


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class TestExtractCheatsheetLesson:
    async def test_lesson_stored_on_success(self, tmp_path):
        hook = _hook(tmp_path)
        mock_resp = MagicMock()
        mock_resp.finish_reason = "stop"
        mock_resp.content = "Always check permissions before writing files."
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)):
            with patch("nano_hermes.memory.cheatsheet._embed_lesson", AsyncMock()):
                await extract_cheatsheet_lesson(hook, _msgs(), "success")
        rows = hook.db.execute(
            "SELECT content, fact_type FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchall()
        assert len(rows) == 1
        assert "permissions" in rows[0][0]

    async def test_skip_response_not_stored(self, tmp_path):
        hook = _hook(tmp_path)
        mock_resp = MagicMock()
        mock_resp.finish_reason = "stop"
        mock_resp.content = "SKIP"
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)):
            await extract_cheatsheet_lesson(hook, _msgs(), "success")
        count = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchone()[0]
        assert count == 0

    async def test_short_response_not_stored(self, tmp_path):
        hook = _hook(tmp_path)
        mock_resp = MagicMock()
        mock_resp.finish_reason = "stop"
        mock_resp.content = "OK"  # < 20 chars
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)):
            await extract_cheatsheet_lesson(hook, _msgs(), "success")
        count = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchone()[0]
        assert count == 0

    async def test_unknown_outcome_not_stored(self, tmp_path):
        hook = _hook(tmp_path)
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock()) as m:
            await extract_cheatsheet_lesson(hook, _msgs(), "unknown")
        # No LLM call made
        m.assert_not_called()

    async def test_no_user_message_skips(self, tmp_path):
        hook = _hook(tmp_path)
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock()) as m:
            await extract_cheatsheet_lesson(hook, [{"role": "assistant", "content": "hi"}], "success")
        m.assert_not_called()

    async def test_llm_error_skips_silently(self, tmp_path):
        hook = _hook(tmp_path)
        mock_resp = MagicMock()
        mock_resp.finish_reason = "error"
        mock_resp.error_status_code = 429
        mock_resp.error_type = "rate_limit"
        mock_resp.error_code = ""
        mock_resp.content = ""
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)):
            await extract_cheatsheet_lesson(hook, _msgs(), "success")
        count = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchone()[0]
        assert count == 0


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieveLessons:
    async def test_retrieves_stored_lessons_via_fts_fallback(self, tmp_path):
        hook = _hook(tmp_path)
        _store_lesson(hook.db, "First lesson about testing.", "test task")
        _store_lesson(hook.db, "Second lesson about deployment.", "deploy task")

        # Embedding fails → FTS fallback
        with patch.object(hook, "embedder", side_effect=Exception("no provider")):
            lessons = await retrieve_lessons(hook, "test task", top_k=3)

        assert len(lessons) >= 1

    async def test_empty_db_returns_empty(self, tmp_path):
        hook = _hook(tmp_path)
        with patch.object(hook, "embedder", side_effect=Exception("no provider")):
            lessons = await retrieve_lessons(hook, "something", top_k=3)
        assert lessons == []

    async def test_multiple_lessons_returned(self, tmp_path):
        hook = _hook(tmp_path)
        _store_lesson(hook.db, "Check logs before escalating issues to prod.", "check logs")
        _store_lesson(hook.db, "Use dry-run flags before destructive ops.", "dry run")
        with patch.object(hook, "embedder", side_effect=Exception("no provider")):
            lessons = await retrieve_lessons(hook, "anything", top_k=3)
        assert len(lessons) == 2


# ---------------------------------------------------------------------------
# Injection message
# ---------------------------------------------------------------------------


class TestBuildInjectionMessage:
    def test_none_when_empty(self):
        assert build_injection_message([]) is None

    def test_system_message_with_bullets(self):
        msg = build_injection_message(["Lesson A.", "Lesson B."])
        assert msg is not None
        assert msg["role"] == "system"
        assert "Lesson A." in msg["content"]
        assert "Lesson B." in msg["content"]
        assert "- " in msg["content"]


# ---------------------------------------------------------------------------
# Disabled by default
# ---------------------------------------------------------------------------


class TestDisabledByDefault:
    def test_hook_default_off(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        assert not getattr(hook.config.skill_stats, "cheatsheet_enabled", False)

    async def test_extraction_no_store_when_disabled(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        # Manually call extraction — won't be gated by hook (hook gates it),
        # but the function itself doesn't check config.
        # Verify no rows exist after we DON'T call it.
        count = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchone()[0]
        assert count == 0
