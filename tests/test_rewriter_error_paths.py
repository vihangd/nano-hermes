"""Tests for rewriter.py error-classifier paths, skip filter, and edit-failed path.

Covers lines previously uncovered: critic abort, main-LLM abort/exception/empty,
skip filter in run_rewriter, critic-blocks path, and edit-failed path.
"""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.rewriter import (
    RewriteCandidate,
    _run_critic,
    rewrite_skill,
    run_rewriter,
)
from nano_hermes.utils.error_classifier import EvolutionAbortError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_hook(tmp_path, config=None):
    return nano_hermes.install(_make_loop(tmp_path), config=config or {})


def _seed_skill(hook, name):
    hook.db.execute(
        "INSERT OR REPLACE INTO skill_stats "
        "(name, status, use_count, success_count, origin) VALUES (?, 'active', 10, 2, 'agent')",
        (name,),
    )
    hook.db.commit()
    d = hook.workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"# {name}\n## Description\nOriginal.")


def _seed_failed_session(hook, skill_name, text="failure"):
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s_{skill_name}_{time.time()}", time.time()),
    )
    sid = cur.lastrowid
    hook.db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (sid, text, time.time()),
    )
    hook.db.execute(
        "INSERT INTO trajectories (session_id, task, outcome, skills_used, created_at) "
        "VALUES (?, 'test task', 'fail', ?, ?)",
        (sid, json.dumps([skill_name]), time.time()),
    )
    hook.db.commit()


def _error_resp(status_code):
    resp = MagicMock()
    resp.finish_reason = "error"
    resp.error_status_code = status_code
    resp.error_type = ""
    resp.error_code = ""
    resp.content = ""
    resp.error_kind = ""
    return resp


def _ok_resp(content):
    resp = MagicMock()
    resp.finish_reason = "stop"
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# _run_critic — error classifier paths
# ---------------------------------------------------------------------------


class TestRunCriticErrorPaths:
    async def test_billing_abort_raises_evolution_abort_error(self, tmp_path):
        """Billing error from critic LLM call → EvolutionAbortError propagates."""
        hook = _make_hook(tmp_path)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_error_resp(402))

        with pytest.raises(EvolutionAbortError) as exc_info:
            await _run_critic(
                hook,
                skill_name="s",
                original_body="orig",
                new_body="new",
                failure_context="failed",
            )
        assert exc_info.value.classified.should_abort is True

    async def test_rate_limit_on_critic_returns_false_after_retries(self, tmp_path):
        """Rate-limit (non-abort) on critic → skip attempt, exhaust retries → False."""
        hook = _make_hook(tmp_path)
        # Both retry attempts rate-limited → fall through and return False
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_error_resp(429))

        result = await _run_critic(
            hook,
            skill_name="s",
            original_body="orig",
            new_body="new",
            failure_context="failed",
        )
        assert result is False


# ---------------------------------------------------------------------------
# rewrite_skill — main LLM call error paths
# ---------------------------------------------------------------------------


class TestRewriteSkillErrorPaths:
    def _setup(self, tmp_path, skill_name="s"):
        hook = _make_hook(
            tmp_path,
            config={
                "skill_stats": {
                    "rewrite_critic_enabled": False,
                    "rewrite_step_localization_enabled": False,
                    "rewrite_replay_gate_enabled": False,
                }
            },
        )
        _seed_skill(hook, skill_name)
        _seed_failed_session(hook, skill_name)
        return hook

    async def test_billing_abort_on_main_llm_raises(self, tmp_path):
        hook = self._setup(tmp_path)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_error_resp(402))

        candidate = RewriteCandidate("s", use_count=10, success_count=2)
        with pytest.raises(EvolutionAbortError):
            await rewrite_skill(hook, candidate)

    async def test_rate_limit_on_main_llm_returns_none(self, tmp_path):
        hook = self._setup(tmp_path)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_error_resp(429))

        candidate = RewriteCandidate("s", use_count=10, success_count=2)
        result = await rewrite_skill(hook, candidate)
        assert result is None

    async def test_exception_on_main_llm_returns_none(self, tmp_path):
        hook = self._setup(tmp_path)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=RuntimeError("network down")
        )

        candidate = RewriteCandidate("s", use_count=10, success_count=2)
        result = await rewrite_skill(hook, candidate)
        assert result is None

    async def test_empty_llm_response_returns_none(self, tmp_path):
        """Empty content from LLM → return None (line 305-306)."""
        hook = self._setup(tmp_path)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_ok_resp(""))

        candidate = RewriteCandidate("s", use_count=10, success_count=2)
        result = await rewrite_skill(hook, candidate)
        assert result is None

    async def test_critic_blocking_returns_none(self, tmp_path):
        """When critic returns False → rewrite_skill returns None (line 324-325)."""
        hook = _make_hook(
            tmp_path,
            config={
                "skill_stats": {
                    "rewrite_critic_enabled": True,
                    "rewrite_step_localization_enabled": False,
                    "rewrite_replay_gate_enabled": False,
                }
            },
        )
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s")

        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_ok_resp("# s\nnew body"))

        with patch(
            "nano_hermes.skills.rewriter._run_critic",
            new=AsyncMock(return_value=False),
        ):
            candidate = RewriteCandidate("s", use_count=10, success_count=2)
            result = await rewrite_skill(hook, candidate)
        assert result is None


# ---------------------------------------------------------------------------
# run_rewriter — skip filter, rewrite-fails, edit-fails
# ---------------------------------------------------------------------------


class TestRunRewriterEdgePaths:
    def _setup(self, tmp_path, skill_name="s"):
        hook = _make_hook(
            tmp_path,
            config={
                "skill_stats": {
                    "rewrite_critic_enabled": False,
                    "rewrite_step_localization_enabled": False,
                    "rewrite_replay_gate_enabled": False,
                }
            },
        )
        _seed_skill(hook, skill_name)
        _seed_failed_session(hook, skill_name)
        return hook

    async def test_skip_filter_excludes_already_evolved(self, tmp_path):
        """Skills in skip= set should not be rewritten (line 373)."""
        hook = self._setup(tmp_path, "s")
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_ok_resp("# s\nnew"))

        result = await run_rewriter(hook, skip=frozenset(["s"]))
        assert result == []
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_rewrite_skill_returns_none_skips_candidate(self, tmp_path):
        """If rewrite_skill returns None (e.g. rate-limited), skill not in result."""
        hook = self._setup(tmp_path, "s")
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_error_resp(429))

        result = await run_rewriter(hook)
        assert result == []

    async def test_edit_failure_not_in_result(self, tmp_path):
        """When _edit returns 'Error: ...', skill not in returned list (line 421)."""
        hook = self._setup(tmp_path, "s")
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_ok_resp("# s\nnew body fixed"))

        with patch(
            "nano_hermes.skills.propose_tool.ProposeSkillTool._edit",
            new=AsyncMock(return_value="Error: skill locked"),
        ):
            result = await run_rewriter(hook)
        assert result == []
