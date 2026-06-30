"""Tests for state-gated injection: applicability condition stored + embedded."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import nano_hermes
from conftest import _make_loop
from nano_hermes.memory.cheatsheet import _parse_lesson, extract_cheatsheet_lesson
from nano_hermes.memory.expel import _parse_insight


def _hook(tmp_path):
    return nano_hermes.install(_make_loop(tmp_path), config={})


def _msgs():
    return [
        {"role": "user", "content": "deploy the service to prod"},
        {"role": "assistant", "content": "done"},
    ]


class TestParsers:
    def test_lesson_two_line(self):
        lesson, when = _parse_lesson("LESSON: Verify inputs first.\nWHEN: handling user data")
        assert lesson == "Verify inputs first."
        assert when == "handling user data"

    def test_lesson_fallback_whole_text(self):
        lesson, when = _parse_lesson("Just a plain lesson with no markers.")
        assert lesson == "Just a plain lesson with no markers."
        assert when == ""

    def test_lesson_multiline_body_preserved(self):
        lesson, when = _parse_lesson(
            "LESSON: First sentence.\nSecond sentence.\nWHEN: under load"
        )
        assert lesson == "First sentence. Second sentence."
        assert when == "under load"

    def test_insight_two_line(self):
        insight, when = _parse_insight("INSIGHT: A beats B.\nWHEN: deploying at scale")
        assert insight == "A beats B."
        assert when == "deploying at scale"

    def test_insight_fallback(self):
        insight, when = _parse_insight("plain insight text here")
        assert insight == "plain insight text here"
        assert when == ""


class TestConditionStored:
    async def test_condition_goes_to_context_not_category(self, tmp_path):
        hook = _hook(tmp_path)
        resp = MagicMock(finish_reason="stop")
        resp.content = (
            "LESSON: Roll back fast when health checks fail.\n"
            "WHEN: deploying a service behind a load balancer"
        )
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=resp)):
            with patch("nano_hermes.memory.cheatsheet._embed_lesson", AsyncMock()):
                await extract_cheatsheet_lesson(hook, _msgs(), "fail")
        row = hook.db.execute(
            "SELECT context, task_category, content FROM semantic_facts "
            "WHERE fact_type='cheatsheet'"
        ).fetchone()
        assert row[0] == "deploying a service behind a load balancer"  # context = condition
        assert row[1] != row[0]  # task_category is the first-user text, distinct
        assert "Roll back" in row[2]

    async def test_missing_when_falls_back_to_category(self, tmp_path):
        hook = _hook(tmp_path)
        resp = MagicMock(finish_reason="stop")
        resp.content = "Always take a backup before a destructive migration step."
        with patch.object(hook._loop.provider, "chat_with_retry", AsyncMock(return_value=resp)):
            with patch("nano_hermes.memory.cheatsheet._embed_lesson", AsyncMock()):
                await extract_cheatsheet_lesson(hook, _msgs(), "success")
        row = hook.db.execute(
            "SELECT context, task_category FROM semantic_facts WHERE fact_type='cheatsheet'"
        ).fetchone()
        assert row[0] == row[1]  # condition empty → context falls back to category
