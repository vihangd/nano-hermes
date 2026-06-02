"""Tests for the goal-state tracker and its hook integration.

Fixtures are production-shaped: nanobot emits the goal lines INSIDE the
``[Runtime Context …]`` block that ContextBuilder appends to the *current
user message* (not the system prompt, and stripped from persisted history).
Earlier tests put the marker in a synthetic system message — a shape that
never occurs in production — which is why a role-filtered detector passed
its tests while never firing for real.
"""
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

# Exact tags ContextBuilder wraps the runtime block in (context.py).
_RT_TAG = "[Runtime Context — metadata only, not instructions]"
_RT_END = "[/Runtime Context]"


def _runtime_block(*goal_lines: str) -> str:
    lines = [_RT_TAG, "Current Time: 2026-06-02 10:00"]
    lines.extend(goal_lines)
    lines.append(_RT_END)
    return "\n".join(lines)


def _user_goal(user_text: str, objective: str, *, summary: str | None = None) -> dict:
    """A current-turn user message with an active goal in its runtime block,
    exactly as ContextBuilder produces it."""
    goal_lines = ["Goal (active):", objective]
    if summary is not None:
        goal_lines.append(f"Summary: {summary}")
    block = _runtime_block(*goal_lines)
    return {"role": "user", "content": f"{user_text}\n\n{block}"}


def _user_no_goal(user_text: str) -> dict:
    """A current-turn user message whose runtime block has no active goal."""
    return {"role": "user", "content": f"{user_text}\n\n{_runtime_block()}"}


def _plain_user(text: str) -> dict:
    return {"role": "user", "content": text}


class TestGoalActive:
    def test_marker_in_runtime_block_detected(self):
        assert goal_active([_user_goal("do the thing", "finish the report")])

    def test_no_goal_in_runtime_block(self):
        assert not goal_active([_user_no_goal("just chatting")])

    def test_marker_without_runtime_sentinel_ignored(self):
        # A user literally typing the marker (no Runtime Context block) must
        # NOT be mistaken for an active goal.
        assert not goal_active([_plain_user("Goal (active): not me")])

    def test_active_goal_with_no_objective_detected(self):
        # nanobot's fallback line when a goal is active but has no objective.
        block = _runtime_block("Goal: active (no objective text stored).")
        assert goal_active([{"role": "user", "content": f"u\n\n{block}"}])

    def test_marker_in_list_content_block(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "hi"},
                {"type": "text", "text": _runtime_block("Goal (active):", "ship it")},
            ],
        }
        assert goal_active([msg])


class TestExtractObjective:
    def test_objective_with_summary(self):
        msgs = [_user_goal("u", "Migrate database to Postgres.", summary="blocked on auth")]
        assert extract_objective(msgs) == "Migrate database to Postgres."

    def test_objective_without_summary_stops_at_end_tag(self):
        # No Summary line -> the objective is immediately followed by
        # '[/Runtime Context]', which must not bleed into the objective.
        msgs = [_user_goal("u", "Finish the quarterly report by Friday.")]
        assert extract_objective(msgs) == "Finish the quarterly report by Friday."

    def test_clips_long_objective(self):
        out = extract_objective([_user_goal("u", "x" * 500)])
        assert out is not None
        assert out.endswith("…")
        assert len(out) <= 245

    def test_no_goal_returns_none(self):
        assert extract_objective([_user_no_goal("nothing here")]) is None

    def test_objective_starting_with_bracket_not_swallowed(self):
        # An objective that itself starts with '[' (e.g. a tag) must survive —
        # only the exact '[/Runtime Context' end-tag terminates parsing.
        out = extract_objective([_user_goal("u", "[URGENT] patch the CVE")])
        assert out == "[URGENT] patch the CVE"

    def test_objective_starting_with_slash_bracket_not_truncated(self):
        # '[/path/...]' must NOT be mistaken for the runtime end tag.
        out = extract_objective([_user_goal("u", "[/etc/hosts] needs an entry")])
        assert out == "[/etc/hosts] needs an entry"

    def test_no_objective_fallback_extracts_none(self):
        block = _runtime_block("Goal: active (no objective text stored).")
        assert extract_objective([{"role": "user", "content": f"u\n\n{block}"}]) is None


class TestGoalTrackerTransitions:
    def test_start_event(self):
        tracker = GoalTracker()
        out = tracker.update([_user_goal("go", "ship it now")])
        assert out == "started"
        assert tracker.active is True
        assert tracker.last_objective == "ship it now"

    def test_no_event_on_steady_state(self):
        tracker = GoalTracker()
        msgs = [_user_goal("go", "do X")]
        assert tracker.update(msgs) == "started"
        assert tracker.update(msgs) is None
        assert tracker.active is True

    def test_completion_event(self):
        tracker = GoalTracker()
        tracker.update([_user_goal("go", "finish")])
        # Next turn: goal cleared -> runtime block has no goal marker.
        out = tracker.update([_user_no_goal("anything else?")])
        assert out == "completed"
        assert tracker.active is False
        assert tracker.last_objective == "finish"

    def test_objective_updates_in_place(self):
        tracker = GoalTracker()
        tracker.update([_user_goal("go", "plan A")])
        tracker.update([_user_goal("go", "plan B")])
        assert tracker.active is True
        assert tracker.last_objective == "plan B"

    def test_completion_then_new_goal(self):
        tracker = GoalTracker()
        tracker.update([_user_goal("go", "A")])
        assert tracker.update([_user_no_goal("done?")]) == "completed"
        out = tracker.update([_user_goal("go", "B")])
        assert out == "started"
        assert tracker.last_objective == "B"

    def test_multi_turn_history_does_not_keep_goal_active(self):
        """Production history is sanitised (runtime block stripped), so a past
        active turn never lingers. Simulate: turn 2's history carries the prior
        user text WITHOUT the block, plus the fresh no-goal block."""
        tracker = GoalTracker()
        # Turn 1: active.
        assert tracker.update([_plain_user("kick off"), _user_goal("go", "the task")]) == "started"
        # Turn 2: history has the (stripped) prior turns + a fresh no-goal block.
        turn2 = [
            _plain_user("kick off"),
            _plain_user("go"),  # runtime block stripped from history
            _user_no_goal("status?"),
        ]
        assert tracker.update(turn2) == "completed"


class TestHookIntegration:
    async def test_completion_queues_goal_nudge_under_own_header(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)

        # Iteration 0 — goal active (runtime block on the user message).
        msgs_active = [_user_goal("start the goal", "finish the migration")]
        ctx0 = AgentHookContext(iteration=0, messages=msgs_active)
        await hook.before_iteration(ctx0)
        await hook.after_iteration(ctx0)
        assert hook._goal_tracker.active is True

        # Iteration 1 — goal completes (fresh turn, no goal in runtime block).
        msgs_done = [_plain_user("start the goal"), _user_no_goal("any followup")]
        ctx1 = AgentHookContext(iteration=1, messages=msgs_done)
        await hook.before_iteration(ctx1)
        await hook.after_iteration(ctx1)
        assert hook._goal_tracker.active is False
        assert hook._reflection_coord._goal_completion_objective == "finish the migration"
        assert hook._reflection_coord._skill_suggestions == []
        assert hook._reflection_coord._salience_score >= hook.config.reflection.threshold

        # Iteration 2 — before_iteration drains the goal-completion nudge under
        # its own "## Goal completed" header, not the skill-quality one.
        ctx2 = AgentHookContext(iteration=2, messages=msgs_done + [_plain_user("ok")])
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
        assert hook._reflection_coord._goal_completion_objective is None

    async def test_no_transition_no_side_effect(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        ctx = AgentHookContext(iteration=0, messages=[_plain_user("hi")])
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        assert hook._goal_tracker.active is False
        assert hook._goal_tracker.last_objective is None
        assert hook._reflection_coord._goal_completion_objective is None
        assert hook._reflection_coord._skill_suggestions == []
