"""Background LLM callers read a refusal or provider error as no answer."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.reconstruction import _MIN_BODY_CHARS, check_reconstruction
from nano_hermes.skills.step_localize import localize_failure_step
from nano_hermes.utils.error_classifier import response_text


def _resp(content: str, finish: str = "stop") -> MagicMock:
    r = MagicMock()
    r.content = content
    r.finish_reason = finish
    r.error_status_code = None
    r.error_type = r.error_code = r.error_kind = None
    return r


def _hook(tmp_path, resp):
    hook = nano_hermes.install(_make_loop(tmp_path))
    hook._loop.provider = MagicMock()
    hook._loop.provider.chat_with_retry = AsyncMock(return_value=resp)
    return hook


def test_response_text():
    assert response_text(_resp("  YES \n"), "t") == "YES"
    assert response_text(_resp("No, I can't help with that.", "refusal"), "t") is None
    assert response_text(_resp("NO", "content_filter"), "t") is None
    assert response_text(_resp("Error: overloaded", "error"), "t") is None


async def test_refused_audit_does_not_block_the_skill(tmp_path):
    # Read as an answer, "NO, I can't…" blocked promotion.
    hook = _hook(tmp_path, _resp("NO, I can't review this.", "refusal"))
    ok = await check_reconstruction(
        hook, skill_name="s", description="d", body="A" * _MIN_BODY_CHARS,
    )
    assert ok is True


async def test_error_text_is_not_a_critique(tmp_path):
    hook = _hook(tmp_path, _resp("Error calling LLM: 500 internal", "error"))
    out = await localize_failure_step(
        hook, skill_name="s", current_body="# s", failure_context="f",
    )
    assert out is None
