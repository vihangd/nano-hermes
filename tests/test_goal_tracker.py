"""Tests for the goal-state tracker and its hook integration."""
from __future__ import annotations

import pytest
from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop
from nano_hermes.coordinator.goal import (
    GoalTracker,
    extract_objective,
    goal_active,
)


def _system(text: str) -> dict:
    return {"role": "system", "content": text}


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


class TestGoalActive:
    def test_marker_present_in_string_content(self):
        msgs = [_system("Some preamble\nGoal (active):\nfinish the report\n")]
        assert goal_active(msgs)

    def test_marker_absent(self):
        msgs = [_system("Plain system text")]
        assert not goal_active(msgs)

    def test_marker_only_in_user_role_ignored(self):
        msgs = [_user("Goal (active): not me")]
        assert not goal_active(msgs)

    def test_marker_in_list_content_block(self):
        msgs = [{
            "role": "system",
            "content": [{"type": "text", "text": "Goal (active):\nship it"}],
        }]
        assert goal_active(msgs)


class TestExtractObjective:
    def test_returns_lines_until_blank(self):
        msgs = [_system(
            "Pre-goal preamble.\n"
            "Goal (active):\n"
            "Finish the quarterly report by Friday.\n"
            "\n"
            "Other unrelated lines.\n"
        )]
        assert extract_objective(msgs) == "Finish the quarterly report by Friday."

    def test_returns_lines_until_summary_prefix(self):
        msgs = [_system(
            "Goal (active):\n"
            "Migrate database to Postgres.\n"
            "Summary: blocked on auth.\n"
        )]
        assert extract_objective(msgs) == "Migrate database to Postgres."

    def test_clips_long_objective(self):
        long = "x" * 500
        msgs = [_system(f"Goal (active):\n{long}\n")]
        out = extract_objective(msgs)
        assert out is not None
        assert out.endswith("…")
        assert len(out) <= 245  # 240 + ellipsis + tolerance

    def test_no_marker_returns_none(self):
        assert extract_objective([_system("no goal here")]) is None

    def test_empty_objective_returns_none(self):
        msgs = [_system("Goal (active):\n\nSummary: x\n")]
        assert extract_objective(msgs) is None


class TestGoalTrackerTransitions:
    def test_start_event(self):
        tracker = GoalTracker()
        out = tracker.update([_system("Goal (active):\nship it now")])
        assert out == "started"
        assert tracker.active is True
        assert tracker.last_objective == "ship it now"

    def test_no_event_on_steady_state(self):
        tracker = GoalTracker()
        msgs = [_system("Goal (active):\ndo X")]
        assert tracker.update(msgs) == "started"
        # Same messages again — no transition.
        assert tracker.update(msgs) is None
        assert tracker.active is True

    def test_completion_event(self):
        tracker = GoalTracker()
        tracker.update([_system("Goal (active):\nfinish")])
        out = tracker.update([_system("(no goal now)")])
        assert out == "completed"
        assert tracker.active is False
        # Objective is preserved across completion so the caller can quote it.
        assert tracker.last_objective == "finish"

    def test_objective_updates_in_place(self):
        tracker = GoalTracker()
        tracker.update([_system("Goal (active):\nplan A")])
        tracker.update([_system("Goal (active):\nplan B")])
        assert tracker.active is True
        assert tracker.last_objective == "plan B"

    def test_completion_then_new_goal(self):
        tracker = GoalTracker()
        tracker.update([_system("Goal (active):\nA")])
        assert tracker.update([_system("no goal")]) == "completed"
        # New goal arrives — emits 'started' again.
        out = tracker.update([_system("Goal (active):\nB")])
        assert out == "started"
        assert tracker.last_objective == "B"


class TestHookIntegration:
    async def test_completion_queues_goal_nudge_under_own_header(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Iteration 0 — goal active.
        msgs_active = [
            _user("start the goal"),
            _system("Goal (active):\nfinish the migration"),
        ]
        ctx0 = AgentHookContext(iteration=0, messages=msgs_active)
        await hook.before_iteration(ctx0)
        await hook.after_iteration(ctx0)
        assert hook._goal_tracker.active is True

        # Iteration 1 — goal completes (marker disappears).
        msgs_done = [
            _user("start the goal"),
            _system("Runtime context with no goal active."),
            _user("any followup"),
        ]
        ctx1 = AgentHookContext(iteration=1, messages=msgs_done)
        await hook.before_iteration(ctx1)
        await hook.after_iteration(ctx1)
        assert hook._goal_tracker.active is False
        # Dedicated goal-completion channel is loaded; skill_suggestions
        # is NOT polluted with the goal-completion message.
        assert hook._reflection_coord._goal_completion_objective == "finish the migration"
        assert hook._reflection_coord._skill_suggestions == []
        # Salience was bumped past threshold so the Reflexion nudge will arm
        # on the next score_iteration call.
        assert hook._reflection_coord._salience_score >= (
            hook.config.reflection.threshold
        )

        # Iteration 2 — before_iteration drains the goal-completion nudge
        # into the conversation under "## Goal completed", not under the
        # skill-quality header.
        ctx2 = AgentHookContext(iteration=2, messages=msgs_done + [_user("ok")])
        await hook.before_iteration(ctx2)
        injected = [
            m for m in ctx2.messages
            if m.get("role") == "system"
            and isinstance(m.get("content"), str)
            and "Goal completed" in m["content"]
        ]
        assert len(injected) == 1
        assert "finish the migration" in injected[0]["content"]
        assert "Skill quality signals" not in injected[0]["content"]
        # And the slot is cleared after drain.
        assert hook._reflection_coord._goal_completion_objective is None

    async def test_no_transition_no_side_effect(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        msgs = [_user("hi")]
        ctx = AgentHookContext(iteration=0, messages=msgs)
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        assert hook._goal_tracker.active is False
        assert hook._goal_tracker.last_objective is None
        assert hook._reflection_coord._goal_completion_objective is None
        assert hook._reflection_coord._skill_suggestions == []
