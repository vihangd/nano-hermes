"""Tests for OPRO prompt meta-optimization (governance/prompt_optimizer.py)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import nano_hermes
from conftest import _make_loop
from nano_hermes.governance.prompt_optimizer import (
    _score_prompt,
    _validate_prompt,
    get_active_prompt,
    record_gepa_rounds,
    run_opro,
)


def _hook(tmp_path, extra_cfg=None):
    cfg = {"skill_stats": {"opro_enabled": True, **(extra_cfg or {})}}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _seed_gepa_improvements(hook, n: int) -> None:
    t = time.time()
    for i in range(n):
        hook.db.execute(
            "INSERT INTO skill_versions (skill_name, body, reason, created_at) "
            "VALUES ('s', 'b', 'gepa: Pareto-best', ?)",
            (t - i,),
        )
    hook.db.commit()


class TestGetActivePrompt:
    def test_returns_fallback_when_no_row(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        result = get_active_prompt(hook.db, "gepa_mutation", "FALLBACK")
        assert result == "FALLBACK"

    def test_returns_db_row_when_active(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        # Must carry every required variable — get_active_prompt re-validates
        # on read and falls back on a prompt promoted under an older contract.
        custom = (
            "CUSTOM PROMPT {skill_name} {round_n} {max_rounds} {lens} "
            "{current_body} {failure_context}"
        )
        hook.db.execute(
            "INSERT INTO prompt_versions (prompt_name, body, active, created_at) "
            "VALUES ('gepa_mutation', ?, 1, ?)",
            (custom, time.time()),
        )
        hook.db.commit()
        result = get_active_prompt(hook.db, "gepa_mutation", "FALLBACK")
        assert result == custom

    def test_inactive_row_ignored(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        hook.db.execute(
            "INSERT INTO prompt_versions (prompt_name, body, active, created_at) "
            "VALUES ('gepa_mutation', 'INACTIVE', 0, ?)",
            (time.time(),),
        )
        hook.db.commit()
        result = get_active_prompt(hook.db, "gepa_mutation", "FALLBACK")
        assert result == "FALLBACK"


class TestScorePrompt:
    def test_returns_none_with_insufficient_data(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        assert _score_prompt(hook.db) is None

    def test_returns_none_below_threshold(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        _seed_gepa_improvements(hook, 4)  # < 5 minimum
        assert _score_prompt(hook.db) is None

    def test_returns_score_with_enough_data(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        _seed_gepa_improvements(hook, 10)
        record_gepa_rounds(hook.db, 20)  # 10/20 = 50%
        score = _score_prompt(hook.db)
        assert score is not None
        assert 45.0 <= score <= 55.0  # ~50%


class TestValidatePrompt:
    def test_valid_prompt_passes(self):
        body = (
            "skill: {skill_name} round {round_n}/{max_rounds} {lens} "
            "body: {current_body} failures: {failure_context}"
        )
        assert _validate_prompt(body, "gepa_mutation") is True

    def test_missing_variable_fails(self):
        body = "skill: {skill_name} round {round_n}/{max_rounds} failures: {failure_context}"
        # Missing {current_body}
        assert _validate_prompt(body, "gepa_mutation") is False

    def test_unknown_prompt_name_passes(self):
        # No required vars defined → always valid
        assert _validate_prompt("anything", "unknown_prompt") is True


class TestRunOproDisabled:
    def test_disabled_by_default_returns_false(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        result = asyncio.run(run_opro(hook))
        assert result is False

    def test_enabled_but_wrong_cycle_count_skips(self, tmp_path):
        hook = _hook(tmp_path, {"opro_cycle_interval": 20})
        hook._evolution_cycle_count = 5  # not a multiple of 20
        result = asyncio.run(run_opro(hook))
        assert result is False

    def test_enabled_cycle_zero_skips(self, tmp_path):
        hook = _hook(tmp_path)
        hook._evolution_cycle_count = 0
        result = asyncio.run(run_opro(hook))
        assert result is False


class TestRunOproInsufficientData:
    def test_skips_when_no_gepa_data(self, tmp_path):
        hook = _hook(tmp_path, {"opro_cycle_interval": 1})
        hook._evolution_cycle_count = 1
        result = asyncio.run(run_opro(hook))
        assert result is False


class TestRunOproGeneratesCandidates:
    async def test_generates_candidates_with_sufficient_data(self, tmp_path):
        hook = _hook(tmp_path, {"opro_cycle_interval": 1, "opro_candidates_per_round": 2})
        hook._evolution_cycle_count = 1
        _seed_gepa_improvements(hook, 10)
        record_gepa_rounds(hook.db, 15)

        valid_body = (
            "Improve {skill_name} round {round_n}/{max_rounds}. {lens} "
            "Current: {current_body}. Failures: {failure_context}."
        )
        mock_resp = MagicMock()
        mock_resp.finish_reason = "stop"
        mock_resp.content = valid_body

        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ):
            result = await run_opro(hook)

        assert result is True
        count = hook.db.execute(
            "SELECT COUNT(*) FROM prompt_versions WHERE prompt_name='gepa_mutation' AND active=0"
        ).fetchone()[0]
        assert count == 2

    async def test_promotes_higher_scored_candidate(self, tmp_path):
        # current rate = 8/20 = 40%; a pre-scored candidate at 80% must win.
        hook = _hook(tmp_path, {"opro_cycle_interval": 1, "opro_candidates_per_round": 1})
        hook._evolution_cycle_count = 1
        _seed_gepa_improvements(hook, 8)
        record_gepa_rounds(hook.db, 20)
        valid_body = (
            "Better {skill_name} {round_n}/{max_rounds}: {lens} "
            "{current_body} // {failure_context}"
        )
        hook.db.execute(
            "INSERT INTO prompt_versions "
            "(prompt_name, body, active, score, scored_at, created_at) "
            "VALUES ('gepa_mutation', ?, 0, 80.0, ?, ?)",
            (valid_body, time.time(), time.time()),
        )
        hook.db.commit()

        empty = MagicMock(finish_reason="stop", content="")
        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=empty)
        ):
            await run_opro(hook)

        active = hook.db.execute(
            "SELECT body FROM prompt_versions "
            "WHERE prompt_name='gepa_mutation' AND active=1"
        ).fetchone()
        assert active is not None and active[0] == valid_body

    async def test_rejects_candidate_missing_variables(self, tmp_path):
        hook = _hook(tmp_path, {"opro_cycle_interval": 1, "opro_candidates_per_round": 1})
        hook._evolution_cycle_count = 1
        _seed_gepa_improvements(hook, 10)
        record_gepa_rounds(hook.db, 15)

        mock_resp = MagicMock()
        mock_resp.finish_reason = "stop"
        mock_resp.content = "improve {skill_name} only"  # missing required vars

        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ):
            result = await run_opro(hook)

        assert result is False  # no valid candidates inserted
        count = hook.db.execute(
            "SELECT COUNT(*) FROM prompt_versions"
        ).fetchone()[0]
        assert count == 0


class TestRequiredVarsContractHolds:
    """The meta-prompt tells the LLM which variables to copy; the validator
    decides which are mandatory. If they drift apart, every generated
    candidate is rejected and OPRO silently stops promoting anything.
    """

    def test_meta_prompt_advertises_every_required_var(self):
        from nano_hermes.governance.prompt_optimizer import (
            _META_PROMPT_TEMPLATE,
            _REQUIRED_VARS,
        )
        for var in _REQUIRED_VARS["gepa_mutation"]:
            # The meta-prompt escapes braces for .format(), so {x} appears {{x}}.
            assert f"{{{var}}}" in _META_PROMPT_TEMPLATE, (
                f"{var} is required by the validator but the meta-prompt never "
                "tells the generator to include it"
            )


class TestActivePromptRevalidatedOnRead:
    def test_stale_active_prompt_falls_back(self, tmp_path):
        import sqlite3 as _sq
        from nano_hermes.governance.prompt_optimizer import get_active_prompt

        db = _sq.connect(":memory:")
        db.execute(
            "CREATE TABLE prompt_versions (prompt_name TEXT, body TEXT, "
            "active INT, scored_at REAL)"
        )
        # Promoted under the older contract — no {lens}.
        db.execute(
            "INSERT INTO prompt_versions VALUES ('gepa_mutation', ?, 1, 1.0)",
            ("{skill_name} {round_n} {max_rounds} {current_body} {failure_context}",),
        )
        db.commit()

        out = get_active_prompt(db, "gepa_mutation", "BUILTIN")
        assert out == "BUILTIN"

    def test_conforming_active_prompt_is_used(self, tmp_path):
        import sqlite3 as _sq
        from nano_hermes.governance.prompt_optimizer import get_active_prompt

        db = _sq.connect(":memory:")
        db.execute(
            "CREATE TABLE prompt_versions (prompt_name TEXT, body TEXT, "
            "active INT, scored_at REAL)"
        )
        good = (
            "{skill_name} {round_n} {max_rounds} {lens} "
            "{current_body} {failure_context}"
        )
        db.execute(
            "INSERT INTO prompt_versions VALUES ('gepa_mutation', ?, 1, 1.0)",
            (good,),
        )
        db.commit()

        assert get_active_prompt(db, "gepa_mutation", "BUILTIN") == good
