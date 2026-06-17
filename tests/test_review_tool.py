"""Tests for PendingReviewTool — covers governance/review_tool.py."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import nano_hermes
from conftest import _make_loop
from nano_hermes.governance import write_approval as wa
from nano_hermes.governance.review_tool import PendingReviewTool


def _make_hook(tmp_path, config=None):
    return nano_hermes.install(_make_loop(tmp_path), config=config or {})


def _seed_skill(hook, name="my_skill"):
    d = hook.workspace / "skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: does {name}\n---\n\nbody\n"
    )
    hook.db.execute(
        "INSERT OR IGNORE INTO skill_stats (name, status, origin) VALUES (?, 'active', 'agent')",
        (name,),
    )
    hook.db.commit()


def _tool(hook):
    return PendingReviewTool(hook=hook)


class TestListAction:
    def test_empty_returns_no_pending_message(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="list"))
        assert result == "No pending writes."

    def test_returns_rows_when_pending(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="new body",
            reason="test reason", origin="rewriter",
        )
        result = asyncio.run(_tool(hook).execute(action="list"))
        assert "my_skill" in result
        assert "rewriter" in result


class TestMissingId:
    def test_diff_without_id_returns_error(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="diff"))
        assert "required" in result.lower()

    def test_reject_without_id_returns_error(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="reject"))
        assert "required" in result.lower()

    def test_approve_without_id_returns_error(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="approve"))
        assert "required" in result.lower()


class TestDiffAction:
    def test_diff_nonexistent_id(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="diff", id=999))
        assert "no pending write" in result.lower()

    def test_diff_existing_skill(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="proposed body",
            reason="perf", origin="gepa",
        )
        result = asyncio.run(_tool(hook).execute(action="diff", id=pid))
        assert "proposed body" in result
        assert "my_skill" in result


class TestRejectAction:
    def test_reject_marks_row_rejected(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="body",
            reason="r", origin="rewriter",
        )
        result = asyncio.run(_tool(hook).execute(action="reject", id=pid))
        assert "rejected" in result
        assert wa.get_pending(hook.db, pid)["status"] == "rejected"

    def test_reject_nonexistent_id(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="reject", id=999))
        assert "no open pending write" in result


class TestApproveSkillAction:
    def test_approve_skill_replays_write(self, tmp_path):
        hook = _make_hook(tmp_path, config={"skill_stats": {"write_approval": "approve"}})
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="approved body",
            reason="r", origin="rewriter",
        )
        result = asyncio.run(_tool(hook).execute(action="approve", id=pid))
        assert "approved" in result
        content = (hook.workspace / "skills" / "my_skill" / "SKILL.md").read_text()
        assert "approved body" in content

    def test_approve_nonexistent_id(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="approve", id=999))
        assert "no open pending write" in result

    def test_approve_stale_skill_refused(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_skill(hook)
        pid = wa.stage_skill_write(
            hook, skill_name="my_skill", description="desc", body="old proposal",
            reason="r", origin="rewriter",
        )
        # mutate skill after staging
        (hook.workspace / "skills" / "my_skill" / "SKILL.md").write_text("changed")
        result = asyncio.run(_tool(hook).execute(action="approve", id=pid))
        assert "refused" in result or "stale" in result


class TestApprovePrincipleAction:
    async def test_approve_principle_delegates_to_approve_principles(self, tmp_path):
        hook = _make_hook(tmp_path, config={"principles": {"enabled": True}})
        ops = [{"op": "add", "condition": "when X", "action": "do Y"}]
        pid = wa.stage_principle_ops(hook, ops=ops, reason="test")
        applied = AsyncMock(return_value={"added": 1})
        with patch("nano_hermes.skills.principle_curator.apply_ops", applied):
            result = await _tool(hook).execute(action="approve", id=pid)
        assert "approved" in result
        assert wa.get_pending(hook.db, pid)["status"] == "approved"


class TestProperties:
    def test_name_property(self, tmp_path):
        hook = _make_hook(tmp_path)
        assert _tool(hook).name == "pending_review"

    def test_description_property(self, tmp_path):
        hook = _make_hook(tmp_path)
        assert "approve" in _tool(hook).description.lower()


class TestUnknownAction:
    def test_unknown_action_returns_error(self, tmp_path):
        hook = _make_hook(tmp_path)
        result = asyncio.run(_tool(hook).execute(action="frobnicate", id=1))
        assert "unknown action" in result.lower()
