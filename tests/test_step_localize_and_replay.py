"""Tests for AgentPRM-lite step localization and ASG-SI replay gate."""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.replay_gate import (
    _parse_verdict,
    gather_failure_trajectories,
    replay_passes_gate,
)
from nano_hermes.skills.rewriter import RewriteCandidate, rewrite_skill
from nano_hermes.skills.step_localize import localize_failure_step


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = text
    return resp


def _make_hook(tmp_path):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    hook._loop.provider = MagicMock()
    hook._loop.model = "test-model"
    return hook


def _seed_failing_trajectory(hook, skill_name: str, *, n_chunks: int = 3, outcome: str = "fail") -> int:
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s_{skill_name}_{time.time()}", time.time()),
    )
    sid = int(cur.lastrowid)
    for i in range(n_chunks):
        hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, ?, 'user', ?, ?)",
            (sid, i, f"chunk content {i}", time.time()),
        )
    hook.db.execute(
        "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
        "VALUES (?, 'task', ?, ?, ?)",
        (sid, json.dumps([skill_name]), outcome, time.time()),
    )
    hook.db.commit()
    return sid


# ---------------------------------------------------------------------------
# Step localization
# ---------------------------------------------------------------------------

class TestLocalizeFailureStep:
    async def test_returns_first_line_trimmed(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("Step 3 fails because the parser expects JSON but got XML.\n\nExtra garbage.")
        )
        result = await localize_failure_step(
            hook,
            skill_name="parser",
            current_body="# parser\n## Steps\n1. parse it",
            failure_context="some failures",
        )
        assert result == "Step 3 fails because the parser expects JSON but got XML."

    async def test_strips_surrounding_quotes(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response('"Quoted localization output."')
        )
        result = await localize_failure_step(
            hook,
            skill_name="x",
            current_body="body",
            failure_context="ctx",
        )
        assert result == "Quoted localization output."

    async def test_returns_none_on_empty_response(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response(""))
        result = await localize_failure_step(
            hook,
            skill_name="x",
            current_body="body",
            failure_context="ctx",
        )
        assert result is None

    async def test_returns_none_on_exception(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(side_effect=RuntimeError("boom"))
        result = await localize_failure_step(
            hook,
            skill_name="x",
            current_body="body",
            failure_context="ctx",
        )
        assert result is None

    async def test_returns_none_when_provider_missing(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider = None
        result = await localize_failure_step(
            hook,
            skill_name="x",
            current_body="body",
            failure_context="ctx",
        )
        assert result is None


# ---------------------------------------------------------------------------
# Replay gate
# ---------------------------------------------------------------------------

class TestParseVerdict:
    def test_improved(self):
        assert _parse_verdict("IMPROVED") == "IMPROVED"

    def test_lowercase(self):
        assert _parse_verdict("improved") == "IMPROVED"

    def test_with_trailing_punctuation(self):
        assert _parse_verdict("IMPROVED.") == "IMPROVED"

    def test_unrecognised(self):
        assert _parse_verdict("MAYBE") is None

    def test_empty(self):
        assert _parse_verdict("") is None


class TestGatherFailureTrajectories:
    def test_returns_chronological_trace(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_failing_trajectory(hook, "doomed", n_chunks=4)
        traces = gather_failure_trajectories(hook.db, "doomed", chunks_per_trajectory=4)
        assert len(traces) == 1
        # Chronological — chunk 0 first.
        lines = traces[0].splitlines()
        assert "chunk content 0" in lines[0]
        assert "chunk content 3" in lines[-1]

    def test_respects_max_trajectories(self, tmp_path):
        hook = _make_hook(tmp_path)
        for _ in range(5):
            _seed_failing_trajectory(hook, "many-fails", n_chunks=2)
        traces = gather_failure_trajectories(hook.db, "many-fails", max_trajectories=2)
        assert len(traces) == 2

    def test_skips_ok_sessions(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_failing_trajectory(hook, "mixed", outcome="ok")
        traces = gather_failure_trajectories(hook.db, "mixed")
        assert traces == []


class TestReplayPassesGate:
    async def test_fails_open_when_no_trajectories(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response("IMPROVED"))
        result = await replay_passes_gate(
            hook,
            skill_name="never-failed",
            original_body="x", new_body="y",
        )
        assert result is True

    async def test_fails_open_when_no_provider(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider = None
        _seed_failing_trajectory(hook, "stuck")
        result = await replay_passes_gate(
            hook,
            skill_name="stuck",
            original_body="x", new_body="y",
        )
        assert result is True

    async def test_passes_when_all_improved(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_failing_trajectory(hook, "fixme")
        _seed_failing_trajectory(hook, "fixme")
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=_mock_response("IMPROVED"))
        result = await replay_passes_gate(
            hook, skill_name="fixme",
            original_body="old", new_body="new",
            max_trajectories=2,
        )
        assert result is True

    async def test_blocks_on_any_worse_verdict(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_failing_trajectory(hook, "regress")
        _seed_failing_trajectory(hook, "regress")
        # First trace IMPROVED, second WORSE → single WORSE vetoes.
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=[_mock_response("IMPROVED"), _mock_response("WORSE")]
        )
        result = await replay_passes_gate(
            hook, skill_name="regress",
            original_body="old", new_body="new",
            max_trajectories=2,
        )
        assert result is False

    async def test_blocks_when_below_pass_rate(self, tmp_path):
        hook = _make_hook(tmp_path)
        for _ in range(3):
            _seed_failing_trajectory(hook, "iffy")
        # 1/3 IMPROVED, 2 SAME → 33% < 60%, blocked. No WORSE.
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=[_mock_response("IMPROVED"), _mock_response("SAME"), _mock_response("SAME")]
        )
        result = await replay_passes_gate(
            hook, skill_name="iffy",
            original_body="old", new_body="new",
            min_pass_rate=0.6,
            max_trajectories=3,
        )
        assert result is False

    async def test_passes_at_pass_rate_threshold(self, tmp_path):
        hook = _make_hook(tmp_path)
        for _ in range(3):
            _seed_failing_trajectory(hook, "borderline")
        # 2/3 IMPROVED ≈ 66% ≥ 60%.
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=[_mock_response("IMPROVED"), _mock_response("IMPROVED"), _mock_response("SAME")]
        )
        result = await replay_passes_gate(
            hook, skill_name="borderline",
            original_body="old", new_body="new",
            min_pass_rate=0.6,
            max_trajectories=3,
        )
        assert result is True


# ---------------------------------------------------------------------------
# rewrite_skill integration — full path with new gates
# ---------------------------------------------------------------------------

class TestRewriteWithGates:
    async def _setup_failing_skill(self, hook, name: str = "broken"):
        skill_dir = hook.workspace / "skills" / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {name}\n## Steps\n1. old")
        _seed_failing_trajectory(hook, name, n_chunks=3)
        return RewriteCandidate(skill_name=name, use_count=10, success_count=2)

    async def test_replay_block_aborts_rewrite(self, tmp_path):
        # critic OFF so the only gates are localization + replay.
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop, config={"skill_stats": {"rewrite_critic_enabled": False}})
        hook._loop.provider = MagicMock()
        hook._loop.model = "test-model"
        candidate = await self._setup_failing_skill(hook)
        new_body = "# broken\n## Steps\n1. new and improved"
        # Three calls: localize, rewrite, replay (WORSE).
        hook._loop.provider.chat_with_retry = AsyncMock(side_effect=[
            _mock_response("Step 1 fails: too vague."),
            _mock_response(new_body),
            _mock_response("WORSE"),
        ])
        result = await rewrite_skill(hook, candidate)
        assert result is None
        # version snapshot should NOT have been written.
        rows = hook.db.execute("SELECT COUNT(*) FROM skill_versions").fetchone()
        assert rows[0] == 0

    async def test_localized_critique_in_rewrite_prompt(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop,
            config={"skill_stats": {
                "rewrite_critic_enabled": False,
                "rewrite_replay_gate_enabled": False,
            }},
        )
        hook._loop.provider = MagicMock()
        hook._loop.model = "test-model"
        candidate = await self._setup_failing_skill(hook, "leaky")
        new_body = "# leaky\n## Steps\n1. better"
        hook._loop.provider.chat_with_retry = AsyncMock(side_effect=[
            _mock_response("Step 2 leaks state across calls."),
            _mock_response(new_body),
        ])
        result = await rewrite_skill(hook, candidate)
        assert result == new_body
        # Inspect the rewrite (second) call's prompt for the localized critique.
        calls = hook._loop.provider.chat_with_retry.call_args_list
        rewrite_call = calls[1]
        msgs = rewrite_call.kwargs.get("messages") or rewrite_call.args[0]
        prompt = msgs[0]["content"]
        assert "LOCALIZED DEFECT" in prompt
        assert "leaks state" in prompt
