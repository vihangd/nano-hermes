"""Tests for GEPA error-classifier paths and write-approval gate in gepa.py.

Covers lines previously uncovered: classifier abort/skip paths in mutation
and evaluation calls, empty/unchanged mutation guard, and write-approval staging.
"""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.gepa import GepaCandidate, evolve_skill, run_gepa
from nano_hermes.utils.error_classifier import ClassifiedError, FailoverReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_skill(hook, name, use_count=10, success_count=5):
    hook.db.execute(
        "INSERT OR REPLACE INTO skill_stats "
        "(name, status, use_count, success_count, origin) VALUES (?, 'active', ?, ?, 'agent')",
        (name, use_count, success_count),
    )
    hook.db.commit()


def _write_skill_file(workspace, name, body):
    d = workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(body)


def _seed_failed_session(hook, skill_name, chunk_text):
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s_{skill_name}_{time.time()}", time.time()),
    )
    session_id = cur.lastrowid
    hook.db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (session_id, chunk_text, time.time()),
    )
    hook.db.execute(
        "INSERT INTO trajectories (session_id, task, outcome, skills_used, created_at) "
        "VALUES (?, 'test task', 'fail', ?, ?)",
        (session_id, f'["{skill_name}"]', time.time()),
    )
    hook.db.commit()


def _error_resp(status_code=None, reason=FailoverReason.rate_limit):
    """Build a MagicMock that looks like a failed LLMResponse."""
    resp = MagicMock()
    resp.finish_reason = "error"
    resp.error_status_code = status_code
    resp.error_type = ""
    resp.error_code = ""
    resp.content = ""
    resp.error_kind = ""
    return resp


def _billing_resp():
    resp = _error_resp(status_code=402)
    return resp


def _rate_limit_resp():
    resp = _error_resp(status_code=429)
    return resp


def _make_provider(*resps):
    """Provider returning resps in order; raises StopIteration if exhausted."""
    it = iter(resps)
    provider = MagicMock()

    async def chat(**kw):
        return next(it)

    provider.chat_with_retry = chat
    return provider


def _ok_resp(content):
    resp = MagicMock()
    resp.finish_reason = "stop"
    resp.content = content
    return resp


# ---------------------------------------------------------------------------
# evolve_skill — mutation call error paths
# ---------------------------------------------------------------------------


class TestEvolveSkillMutationErrors:
    async def test_rate_limit_on_mutation_skips_round(self, tmp_path):
        """Rate-limit on mutation → continue (skip that round), eventually return None."""
        hook = nano_hermes.install(_make_loop(tmp_path))
        _write_skill_file(tmp_path, "s", "# s\nbody")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        # Both rounds return rate-limit error → no improvement
        hook._loop.provider = _make_provider(
            _rate_limit_resp(), _rate_limit_resp()
        )
        candidate = GepaCandidate("s", use_count=10, success_count=5)
        result = await evolve_skill(hook, candidate, max_rounds=2, minibatch_size=3)
        assert result is None

    async def test_billing_abort_on_mutation_raises(self, tmp_path):
        """Billing abort on mutation → EvolutionAbortError propagates out."""
        from nano_hermes.utils.error_classifier import EvolutionAbortError

        hook = nano_hermes.install(_make_loop(tmp_path))
        _write_skill_file(tmp_path, "s", "# s\nbody")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        hook._loop.provider = _make_provider(_billing_resp())
        candidate = GepaCandidate("s", use_count=10, success_count=5)
        with pytest.raises(EvolutionAbortError) as exc_info:
            await evolve_skill(hook, candidate, max_rounds=1, minibatch_size=3)
        assert exc_info.value.classified.should_abort is True

    async def test_empty_mutation_body_skips_round(self, tmp_path):
        """Empty mutation response → continue (unchanged mutation guard: line 253)."""
        hook = nano_hermes.install(_make_loop(tmp_path))
        original_body = "# s\nbody"
        _write_skill_file(tmp_path, "s", original_body)
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        # First call returns empty body, second returns same as original
        hook._loop.provider = _make_provider(
            _ok_resp(""),        # empty → skip
            _ok_resp(original_body),  # unchanged → skip
        )
        candidate = GepaCandidate("s", use_count=10, success_count=5)
        result = await evolve_skill(hook, candidate, max_rounds=2, minibatch_size=3)
        assert result is None


# ---------------------------------------------------------------------------
# evolve_skill — evaluation call error paths
# ---------------------------------------------------------------------------


class TestEvolveSkillEvalErrors:
    async def test_rate_limit_on_eval_skips_round(self, tmp_path):
        """Rate-limit on evaluation → continue (skip that round)."""
        hook = nano_hermes.install(_make_loop(tmp_path))
        _write_skill_file(tmp_path, "s", "# s\nbody")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        hook._loop.provider = _make_provider(
            _ok_resp("# s\nimproved body"),  # mutation succeeds
            _rate_limit_resp(),              # eval rate-limited → skip
        )
        candidate = GepaCandidate("s", use_count=10, success_count=5)
        result = await evolve_skill(hook, candidate, max_rounds=1, minibatch_size=3)
        assert result is None

    async def test_billing_abort_on_eval_raises(self, tmp_path):
        """Billing abort on evaluation → EvolutionAbortError."""
        from nano_hermes.utils.error_classifier import EvolutionAbortError

        hook = nano_hermes.install(_make_loop(tmp_path))
        _write_skill_file(tmp_path, "s", "# s\nbody")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        hook._loop.provider = _make_provider(
            _ok_resp("# s\nimproved body"),  # mutation ok
            _billing_resp(),                 # eval aborts
        )
        candidate = GepaCandidate("s", use_count=10, success_count=5)
        with pytest.raises(EvolutionAbortError):
            await evolve_skill(hook, candidate, max_rounds=1, minibatch_size=3)

    async def test_exception_on_eval_skips_round(self, tmp_path):
        """Generic exception on evaluation call → continue."""
        hook = nano_hermes.install(_make_loop(tmp_path))
        _write_skill_file(tmp_path, "s", "# s\nbody")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        call_count = 0
        provider = MagicMock()

        async def chat(**kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _ok_resp("# s\nimproved body")
            raise RuntimeError("eval network failure")

        provider.chat_with_retry = chat
        hook._loop.provider = provider

        candidate = GepaCandidate("s", use_count=10, success_count=5)
        result = await evolve_skill(hook, candidate, max_rounds=1, minibatch_size=3)
        assert result is None


# ---------------------------------------------------------------------------
# run_gepa — write-approval gate and edit-failed path
# ---------------------------------------------------------------------------


class TestRunGepaGate:
    async def test_write_approval_gate_stages_instead_of_writing(self, tmp_path):
        """Under write_approval=approve, run_gepa stages but returns empty list."""
        from nano_hermes.governance import write_approval as wa

        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={
                "skill_stats": {
                    "gepa_enabled": True,
                    "gepa_max_mutations": 1,
                    "write_approval": "approve",
                }
            },
        )
        original_body = "# s\n## Description\nOriginal."
        _write_skill_file(tmp_path, "s", original_body)
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed because unclear")

        improved_body = "# s\n## Description\nImproved with clearer steps."
        hook._loop.provider = _make_provider(
            _ok_resp(improved_body), _ok_resp("Y")
        )
        result = await run_gepa(hook)
        # Gate active → skill not reported as evolved
        assert result == []
        # But a pending row should exist
        rows = wa.list_pending(hook.db)
        assert any(r["skill_name"] == "s" and r["origin"] == "gepa" for r in rows)
        # SKILL.md must be unchanged
        assert (tmp_path / "skills" / "s" / "SKILL.md").read_text() == original_body

    async def test_edit_failure_not_added_to_evolved(self, tmp_path):
        """When ProposeSkillTool._edit returns Error..., skill not in returned list."""
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"gepa_enabled": True, "gepa_max_mutations": 1}},
        )
        _write_skill_file(tmp_path, "s", "# s\n## Description\nOriginal.")
        _seed_skill(hook, "s")
        _seed_failed_session(hook, "s", "failed")

        improved_body = "# s\n## Description\nImproved."
        hook._loop.provider = _make_provider(
            _ok_resp(improved_body), _ok_resp("Y")
        )
        with patch(
            "nano_hermes.skills.propose_tool.ProposeSkillTool._edit",
            new=AsyncMock(return_value="Error: permission denied"),
        ):
            result = await run_gepa(hook)
        assert result == []
