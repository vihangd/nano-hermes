"""Tests for the MIND-Skill reconstruction check (skills/reconstruction.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import nano_hermes
from conftest import _make_loop

from nano_hermes.skills.reconstruction import check_reconstruction, _MIN_BODY_CHARS


def _make_hook(tmp_path):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop)
    hook._loop.provider = MagicMock()
    return hook


class TestCheckReconstruction:
    def _yes_response(self):
        r = MagicMock()
        r.content = "YES"
        return r

    def _no_response(self):
        r = MagicMock()
        r.content = "NO"
        return r

    async def test_passes_when_llm_says_yes(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=self._yes_response())

        ok = await check_reconstruction(
            hook,
            skill_name="my-skill",
            description="Fetch and parse a web page",
            body="A" * _MIN_BODY_CHARS,
        )
        assert ok is True

    async def test_blocks_when_llm_says_no(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=self._no_response())

        ok = await check_reconstruction(
            hook,
            skill_name="misleading-skill",
            description="Fetch and parse a web page",
            body="A" * _MIN_BODY_CHARS,
        )
        assert ok is False

    async def test_skips_check_for_short_body(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock()

        ok = await check_reconstruction(
            hook,
            skill_name="stub-skill",
            description="Something",
            body="too short",  # < _MIN_BODY_CHARS
        )
        assert ok is True
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_retries_once_on_llm_exception(self, tmp_path):
        hook = _make_hook(tmp_path)
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=RuntimeError("network error")
        )

        ok = await check_reconstruction(
            hook,
            skill_name="flaky-skill",
            description="something",
            body="A" * _MIN_BODY_CHARS,
        )
        # Fail open after 2 attempts
        assert ok is True
        assert hook._loop.provider.chat_with_retry.call_count == 2

    async def test_fails_open_on_ambiguous_response(self, tmp_path):
        hook = _make_hook(tmp_path)
        r = MagicMock()
        r.content = "MAYBE"
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=r)

        ok = await check_reconstruction(
            hook,
            skill_name="ambiguous-skill",
            description="something",
            body="A" * _MIN_BODY_CHARS,
        )
        assert ok is True


class TestReconstructionGateIntegration:
    """Test that _filter_reconstruction_blocked gates promotion correctly."""

    def _setup_draft_skill(self, hook, skill_name: str, description: str, body: str) -> None:
        skill_dir = hook.workspace / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n"
        )
        hook.db.execute(
            "INSERT OR REPLACE INTO skill_stats "
            "(name, status, use_count, success_count) VALUES (?, 'draft', 10, 3)",
            (skill_name,),
        )
        hook.db.commit()

    async def test_filter_passes_skill_when_check_passes(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        hook._loop.provider = MagicMock()

        r = MagicMock()
        r.content = "YES"
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=r)

        self._setup_draft_skill(
            hook, "good-skill", "Fetches web content", "A" * _MIN_BODY_CHARS
        )

        result = await hook._filter_reconstruction_blocked(["good-skill"])
        assert "good-skill" in result

    async def test_filter_blocks_skill_when_check_fails(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        hook._loop.provider = MagicMock()

        r = MagicMock()
        r.content = "NO"
        hook._loop.provider.chat_with_retry = AsyncMock(return_value=r)

        self._setup_draft_skill(
            hook, "bad-skill", "Fetches web content", "A" * _MIN_BODY_CHARS
        )

        result = await hook._filter_reconstruction_blocked(["bad-skill"])
        assert "bad-skill" not in result

    async def test_filter_skips_check_for_active_skills(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock()

        hook.db.execute(
            "INSERT OR REPLACE INTO skill_stats "
            "(name, status, use_count, success_count) VALUES ('active-skill', 'active', 10, 8)",
        )
        hook.db.commit()

        result = await hook._filter_reconstruction_blocked(["active-skill"])
        assert "active-skill" in result
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_filter_skips_check_for_below_threshold_drafts(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock()

        # success_count = 1, promotion_threshold = 3 → not yet eligible
        hook.db.execute(
            "INSERT OR REPLACE INTO skill_stats "
            "(name, status, use_count, success_count) VALUES ('low-skill', 'draft', 5, 1)",
        )
        hook.db.commit()

        result = await hook._filter_reconstruction_blocked(["low-skill"])
        assert "low-skill" in result
        hook._loop.provider.chat_with_retry.assert_not_called()

    async def test_reconstruction_check_disabled_via_config(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(
            loop, config={"skill_stats": {"reconstruction_check_enabled": False}}
        )
        hook._loop.provider = MagicMock()
        hook._loop.provider.chat_with_retry = AsyncMock()

        self._setup_draft_skill(
            hook, "no-check-skill", "Something", "A" * _MIN_BODY_CHARS
        )

        await hook._check_promotions(["no-check-skill"])
        hook._loop.provider.chat_with_retry.assert_not_called()
