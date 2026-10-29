"""Session outcome labels (session/outcome.py) and their boundary wiring."""
from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from nanobot.agent.hook import AgentHookContext, AgentRunHookContext

import nano_hermes
from conftest import _make_loop
from nano_hermes.memory.expel import find_contrasting_session
from nano_hermes.session.db import _apply_migrations, open_db
from nano_hermes.session.outcome import (
    OutcomeSignals,
    OutcomeTracker,
    classify_user_text,
    label,
)

_RT = "[Runtime Context — metadata only, not instructions]"


def _user(text: str) -> dict:
    return {"role": "user", "content": f"{text}\n\n{_RT}\n[/Runtime Context]"}


async def _turn(hook, msgs, message, key, *, error=None, tool_iterations=0):
    """One user turn: iteration 0, then *tool_iterations* more on the same list."""
    msgs.append(message)
    ctx = AgentHookContext(iteration=0, messages=msgs, session_key=key)
    await hook.before_iteration(ctx)
    msgs[:] = [m for m in msgs if m.get("role") != "system"]
    for i in range(1, tool_iterations + 1):
        msgs.append({"role": "assistant", "content": f"tool step {i}"})
        await hook.after_iteration(AgentHookContext(iteration=i - 1, messages=msgs, session_key=key))
        await hook.before_iteration(AgentHookContext(iteration=i, messages=msgs, session_key=key))
    msgs.append({"role": "assistant", "content": "done"})
    await hook.after_iteration(
        AgentHookContext(iteration=tool_iterations, messages=msgs, session_key=key, error=error)
    )


def _outcomes(hook) -> list[str]:
    return [r[0] for r in hook.db.execute("SELECT outcome FROM trajectories ORDER BY id")]


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize(
    "text, verdict",
    [
        ("thanks!", "positive"),
        ("no, that's perfect", "positive"),
        ("that works", "positive"),
        ("no that is wrong", "correction"),
        ("no, use the other file", "correction"),
        ("thanks, but it's still broken", "correction"),
        ("it still doesn't work", "correction"),
        ("try again", "correction"),
        ("no", "neutral"),          # answer to a yes/no question
        ("actually, add tests", "neutral"),
        ("can you add a flag", "neutral"),
        (f"thanks\n{_RT}\nChannel: telegram", "positive"),
        ("[Message Time: 1]\nold summary says it's wrong\n\nthanks", "positive"),
        ("You have an active sustained goal:\n\nship it", None),
        ("", None),
    ],
)
def test_classify_user_text(text, verdict):
    assert classify_user_text(text) == verdict


# --- rule label -------------------------------------------------------------

def _sig(*reactions, **kw) -> OutcomeSignals:
    sig = OutcomeSignals(**kw)
    sig.reactions.extend(reactions)
    return sig


@pytest.mark.parametrize(
    "sig, expected",
    [
        (_sig(), "unknown"),
        (_sig(tool_calls=9, tool_errors=4), "unknown"),  # tool errors alone decide nothing
        (_sig("positive"), "ok"),
        (_sig("correction", "positive"), "ok"),
        (_sig("correction"), "partial"),
        (_sig("correction", "correction"), "fail"),
        (_sig("positive", "correction"), "partial"),
        (_sig("neutral", rated_failure=True), "partial"),
        (_sig("positive", last_stop="error"), "fail"),
        (_sig(last_stop="tool_error"), "fail"),
        (_sig(llm_error=True), "fail"),
        (_sig(last_stop="max_iterations"), "fail"),
        (_sig(last_stop="max_iterations", goal_active=True), "unknown"),
        (_sig(goal="complete"), "ok"),
        (_sig("correction", goal="complete"), "partial"),
        (_sig(goal="cancel"), "unknown"),
        (_sig(goal="block"), "fail"),
    ],
)
def test_label(sig, expected):
    assert label(sig) == expected


def test_label_reads_a_recent_window_not_session_totals():
    tracker = OutcomeTracker()
    texts = ["start", "wrong", "no that is wrong", "ok next", "next", "that works"]
    tracker.record_user_messages("k", texts)
    assert label(tracker.take("k")) == "ok"


def test_cron_and_heartbeat_sessions_are_never_scored():
    tracker = OutcomeTracker()
    tracker.record_user_messages("cron:abc", ["x", "thanks"])
    tracker.record_user_messages("heartbeat", ["x", "thanks"])
    assert label(tracker.take("cron:abc")) == "unknown"
    assert label(tracker.take("heartbeat")) == "unknown"


# --- hook wiring ------------------------------------------------------------

def _hook(tmp_path, **config):
    return nano_hermes.install(_make_loop(tmp_path), config=config or None)


async def test_silent_session_is_unknown_and_thanks_is_ok(tmp_path):
    hook = _hook(tmp_path)
    await _turn(hook, [], _user("deploy the app"), "k")
    await _turn(hook, [], _user("new task"), "k")      # /new: ends session 1
    msgs: list = []
    await _turn(hook, msgs, _user("write a haiku"), "k")
    await _turn(hook, msgs, _user("thanks, perfect"), "k")
    await _turn(hook, [], _user("another"), "k")       # ends session 3
    assert _outcomes(hook) == ["unknown", "unknown", "ok"]


async def test_a_correction_is_scored_once_across_tool_iterations(tmp_path):
    hook = _hook(tmp_path)
    msgs: list = []
    await _turn(hook, msgs, _user("fix the bug"), "k")
    await _turn(hook, msgs, _user("that's wrong"), "k", tool_iterations=3)
    await _turn(hook, [], _user("next"), "k")
    assert _outcomes(hook) == ["partial"]  # two scorings would make it "fail"


async def test_the_next_sessions_opener_credits_the_ended_session(tmp_path):
    hook = _hook(tmp_path)
    await _turn(hook, [], _user("migrate the db"), "k")
    await _turn(hook, [], _user("thanks, that worked"), "k")  # opens session 2
    assert _outcomes(hook) == ["ok"]


async def test_a_model_error_fails_the_session(tmp_path):
    hook = _hook(tmp_path)
    msgs: list = []
    await _turn(hook, msgs, _user("go"), "k")
    await _turn(hook, msgs, _user("thanks"), "k", error="boom")
    await _turn(hook, [], _user("next"), "k")
    assert _outcomes(hook) == ["fail"]


async def test_run_stop_reason_from_on_error_and_on_finally(tmp_path):
    hook = _hook(tmp_path)
    msgs: list = []
    await _turn(hook, msgs, _user("go"), "")
    # 0.2.2 reports a raised run only through on_error.
    await hook.on_error(AgentRunHookContext(messages=[], stop_reason="tool_error"))
    await _turn(hook, [], _user("next"), "")
    # A goal's max_iterations is a budget boundary, not a failure.
    hook._outcomes.set_goal_active("", True)
    await hook.on_finally(AgentRunHookContext(messages=[], stop_reason="max_iterations"))
    await _turn(hook, [], _user("again"), "")
    assert _outcomes(hook)[:2] == ["fail", "unknown"]


async def test_update_goal_complete_is_ok_and_block_is_fail(tmp_path):
    hook = _hook(tmp_path)
    for action in ("complete", "block"):
        msgs: list = []
        await _turn(hook, msgs, _user(f"long task {action}"), "k")
        call = SimpleNamespace(name="update_goal", arguments={"action": action})
        await hook.after_iteration(AgentHookContext(
            iteration=0, messages=msgs, session_key="k",
            tool_calls=[call], tool_events=[{"name": "update_goal", "status": "ok"}],
        ))
    await _turn(hook, [], _user("next"), "k")
    assert _outcomes(hook) == ["ok", "fail"]


async def test_a_failed_skill_rating_makes_the_session_partial(tmp_path):
    hook = _hook(tmp_path)
    await _turn(hook, [], _user("use the skill"), "")
    hook.record_skill_rating("some-skill", success=False)
    hook.record_skill_rating("other-skill", success=True)  # self-grade: no effect
    await _turn(hook, [], _user("next"), "")
    assert _outcomes(hook) == ["partial"]


async def test_unknown_moves_no_counters_but_keeps_coactivations(tmp_path):
    hook = _hook(tmp_path)
    db, coord = hook.db, hook._reflection_coord
    now = time.time()
    pid = db.execute(
        "INSERT INTO principles (condition, action, created_at) VALUES ('c', 'a', ?)", (now,)
    ).lastrowid
    rids = [
        db.execute(
            "INSERT INTO reflections (content, created_at, utility) VALUES (?, ?, 0.5)",
            (f"r{i}", now),
        ).lastrowid
        for i in range(2)
    ]
    fid = db.execute(
        "INSERT INTO semantic_facts (content, source_chunk_ids, created_at) VALUES ('l', '[]', ?)",
        (now,),
    ).lastrowid
    db.commit()
    await _turn(hook, [], _user("task"), "k")
    coord._conv("k").injected_principle_ids.add(pid)
    coord._conv("k").injected_reflection_ids.update(rids)
    hook._lessons("k")[fid] = 1.0
    await _turn(hook, [], _user("next"), "k")  # unknown
    assert _outcomes(hook) == ["unknown"]
    assert db.execute("SELECT success_count, harmful_count FROM principles").fetchone() == (0, 0)
    assert {u for (u,) in db.execute("SELECT utility FROM reflections")} == {0.5}
    assert db.execute("SELECT trust_score FROM semantic_facts").fetchone()[0] == 1.0
    assert db.execute("SELECT count(*) FROM reflection_coactivations").fetchone()[0] == 1


def test_expel_does_not_contrast_unlabelled_sessions(tmp_path):
    hook = _hook(tmp_path)
    assert find_contrasting_session(hook.db, 1, "unknown") is None
    assert find_contrasting_session(hook.db, 1, "partial") is None


# --- migration --------------------------------------------------------------

def test_legacy_ok_becomes_unknown_and_reruns_spare_new_labels(tmp_path):
    db = open_db(tmp_path, 4)
    db.execute(
        "INSERT INTO trajectories (task, outcome, created_at) VALUES ('old', 'ok', 0), "
        "('old-fail', 'fail', 0)"
    )
    db.execute(
        "INSERT INTO trajectories (task, outcome, outcome_source, created_at) "
        "VALUES ('new', 'ok', 'rules', 0)"
    )
    db.commit()
    db.execute("PRAGMA user_version = 0")  # unstamped: every statement re-runs
    _apply_migrations(db)
    rows = dict(db.execute("SELECT task, outcome FROM trajectories").fetchall())
    assert rows == {"old": "unknown", "old-fail": "fail", "new": "ok"}
    db.close()


async def test_writer_stamps_source_and_signals(tmp_path):
    import json

    hook = _hook(tmp_path)
    await _turn(hook, [], _user("task"), "k")
    await _turn(hook, [], _user("next"), "k")
    source, signals = hook.db.execute(
        "SELECT outcome_source, outcome_signals FROM trajectories"
    ).fetchone()
    assert source == "rules"
    assert json.loads(signals)["user_messages"] == 1


def test_a_sessions_first_message_reacts_to_nothing():
    tracker = OutcomeTracker()
    tracker.record_user_messages("k", ["it still doesn't work, fix the login"])
    assert label(tracker.take("k")) == "unknown"
