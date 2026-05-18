"""Tests for PrincipleTool and principle injection (item 7 / EvolveR)."""
from __future__ import annotations

import time

import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.principle_tool import PrincipleTool


def _make_hook(tmp_path):
    loop = _make_loop(tmp_path)
    return nano_hermes.install(loop)


class TestPrincipleToolRecord:
    async def test_records_principle_with_outcome(self, tmp_path):
        hook = _make_hook(tmp_path)
        tool = PrincipleTool(hook=hook)

        result = await tool.execute(
            condition="When deploying to a systemd server",
            action="Check the service file exists first",
            expected_outcome="Avoids service-not-found errors",
        )
        assert result.startswith("ok: principle #")

        row = hook.db.execute("SELECT condition, action, expected_outcome FROM principles").fetchone()
        assert row is not None
        assert "systemd" in row[0]
        assert "service file" in row[1]
        assert "errors" in row[2]

    async def test_records_principle_without_outcome(self, tmp_path):
        hook = _make_hook(tmp_path)
        tool = PrincipleTool(hook=hook)

        result = await tool.execute(
            condition="When parsing paginated API results",
            action="Always check for next_page token",
        )
        assert result.startswith("ok:")

    async def test_requires_condition(self, tmp_path):
        hook = _make_hook(tmp_path)
        tool = PrincipleTool(hook=hook)
        result = await tool.execute(action="do something")
        assert result.startswith("Error")

    async def test_requires_action(self, tmp_path):
        hook = _make_hook(tmp_path)
        tool = PrincipleTool(hook=hook)
        result = await tool.execute(condition="when something happens")
        assert result.startswith("Error")

    async def test_indexes_in_fts(self, tmp_path):
        hook = _make_hook(tmp_path)
        tool = PrincipleTool(hook=hook)
        await tool.execute(
            condition="When deploying to production server",
            action="Always run smoke tests first",
        )

        rows = hook.db.execute(
            "SELECT content_id FROM principles_fts WHERE principles_fts MATCH 'deploy'"
        ).fetchall()
        assert len(rows) == 1


class TestPrincipleInjection:
    async def _seed_principle(self, hook, condition: str, action: str) -> int:
        tool = PrincipleTool(hook=hook)
        await tool.execute(condition=condition, action=action)
        return hook.db.execute(
            "SELECT id FROM principles ORDER BY id DESC LIMIT 1"
        ).fetchone()[0]

    async def test_injects_matching_principle(self, tmp_path):
        hook = _make_hook(tmp_path)
        await self._seed_principle(
            hook,
            "When deploying a Python server to production",
            "Run health checks before marking deploy complete",
        )

        messages = [{"role": "user", "content": "Deploy the Python app to production"}]
        injections = hook._reflection_coord.get_principle_injections(messages)
        assert len(injections) == 1
        assert "deploy" in injections[0]["content"].lower()
        assert "health checks" in injections[0]["content"]

    async def test_does_not_inject_unrelated_principle(self, tmp_path):
        hook = _make_hook(tmp_path)
        await self._seed_principle(
            hook,
            "When parsing CSV files with pandas",
            "Always specify dtype=str to avoid silent type coercion",
        )

        messages = [{"role": "user", "content": "Write a shell script to restart nginx"}]
        injections = hook._reflection_coord.get_principle_injections(messages)
        assert injections == []

    async def test_empty_when_no_principles(self, tmp_path):
        hook = _make_hook(tmp_path)
        messages = [{"role": "user", "content": "Do something"}]
        injections = hook._reflection_coord.get_principle_injections(messages)
        assert injections == []

    async def test_empty_when_no_user_message(self, tmp_path):
        hook = _make_hook(tmp_path)
        await self._seed_principle(hook, "when deploying", "check first")
        messages = [{"role": "system", "content": "You are a helpful assistant"}]
        injections = hook._reflection_coord.get_principle_injections(messages)
        assert injections == []

    async def test_increments_use_count_on_match(self, tmp_path):
        hook = _make_hook(tmp_path)
        pid = await self._seed_principle(
            hook,
            "When deploying a server",
            "Run smoke tests",
        )
        initial = hook.db.execute(
            "SELECT use_count FROM principles WHERE id = ?", (pid,)
        ).fetchone()[0]

        messages = [{"role": "user", "content": "Deploy the server now"}]
        hook._reflection_coord.get_principle_injections(messages)

        updated = hook.db.execute(
            "SELECT use_count FROM principles WHERE id = ?", (pid,)
        ).fetchone()[0]
        assert updated == initial + 1

    async def test_respects_limit(self, tmp_path):
        hook = _make_hook(tmp_path)
        for i in range(5):
            await self._seed_principle(
                hook,
                f"When deploying to server in scenario {i}",
                f"action {i}",
            )

        messages = [{"role": "user", "content": "Deploy to server"}]
        injections = hook._reflection_coord.get_principle_injections(messages, limit=2)
        # Should not return more than 2
        assert len(injections) <= 1  # they're packed in one message
        # But the content should reference at most 2 principles
        if injections:
            bullets = injections[0]["content"].count("• If:")
            assert bullets <= 2
