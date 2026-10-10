"""Session-outcome state belongs to one conversation (F4).

One hook serves every chat the loop runs, but a session boundary is per
conversation. With process-wide state, one chat's boundary credited or
penalised the other's injected principles, reflections and lessons and reset
its skill tracking; its /goal tracker read the other chat's window as a goal
"completed"; and a Reflexion nudge went to whichever chat iterated next.
"""
from __future__ import annotations

import time

from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop

_RT = "[Runtime Context — metadata only, not instructions]"


def _user_goal(text: str, objective: str) -> dict:
    block = f"{_RT}\nGoal (active):\n{objective}\n[/Runtime Context]"
    return {"role": "user", "content": f"{text}\n\n{block}"}


def _user(text: str) -> dict:
    return {"role": "user", "content": f"{text}\n\n{_RT}\n[/Runtime Context]"}


async def _turn(hook, msgs, message, key, *, error=None):
    msgs.append(message)
    ctx = AgentHookContext(iteration=0, messages=msgs, session_key=key)
    await hook.before_iteration(ctx)
    injected = [m["content"] for m in msgs if m.get("role") == "system"]
    msgs[:] = [m for m in msgs if m.get("role") != "system"]
    msgs.append({"role": "assistant", "content": "ack"})
    after = AgentHookContext(iteration=0, messages=msgs, session_key=key, error=error)
    await hook.after_iteration(after)
    return injected


def _hook(tmp_path):
    return nano_hermes.install(_make_loop(tmp_path), config={})


async def test_switching_chats_does_not_complete_a_goal(tmp_path):
    hook = _hook(tmp_path)
    a, b = [], []
    await _turn(hook, a, _user_goal("start", "finish the migration"), "chan:a")
    seen = []
    for i in range(3):
        seen += await _turn(hook, b, _user(f"b {i}"), "chan:b")
        seen += await _turn(hook, a, _user_goal(f"a {i}", "finish the migration"), "chan:a")
    assert not [s for s in seen if "Goal completed" in s]
    assert hook._goal_tracker("chan:a").active is True


async def test_a_reflexion_nudge_goes_to_the_chat_that_earned_it(tmp_path):
    hook = _hook(tmp_path)
    coord = hook._reflection_coord
    a, b = [], []
    await _turn(hook, a, _user("a 0"), "chan:a")
    await _turn(hook, b, _user("b 0"), "chan:b")
    coord.add_salience(hook.config.reflection.threshold + 1.0, "chan:a")
    await _turn(hook, a, _user("a 1"), "chan:a")  # crosses the threshold
    b_seen = await _turn(hook, b, _user("b 1"), "chan:b")
    a_seen = await _turn(hook, a, _user("a 2"), "chan:a")
    is_nudge = lambda s: "reflect" in s.lower() and "Memory check-in" not in s  # noqa: E731
    assert not [s for s in b_seen if is_nudge(s)], b_seen
    assert [s for s in a_seen if is_nudge(s)], a_seen


async def test_a_boundary_scores_and_resets_only_its_own_conversation(tmp_path):
    hook = _hook(tmp_path)
    coord, db = hook._reflection_coord, hook.db
    now = time.time()
    pa, pb = (
        db.execute(
            "INSERT INTO principles (condition, action, created_at) VALUES (?, ?, ?)",
            (f"when {x}", f"do {x}", now),
        ).lastrowid
        for x in ("a", "b")
    )
    db.commit()
    a, b = [], []
    await _turn(hook, a, _user("a 0"), "chan:a")
    await _turn(hook, b, _user("b 0"), "chan:b")
    for key, pid, skill, lesson in (("chan:a", pa, "a-skill", 1), ("chan:b", pb, "b-skill", 2)):
        coord._conv(key).injected_principle_ids.add(pid)
        coord._conv(key).injected_reflection_ids.add(100 + pid)
        hook._skill_tracker._state(key).session_skills_used.add(skill)
        hook._lessons(key)[lesson] = 1.0

    # /new in chat A: a fresh window under the same key ends A's session.
    a2 = []
    await _turn(hook, a2, _user("fresh start"), "chan:a")

    credited = dict(db.execute("SELECT id, success_count FROM principles").fetchall())
    assert credited == {pa: 1, pb: 0}
    st_b = coord._conv("chan:b")
    assert st_b.injected_principle_ids == {pb}
    assert st_b.injected_reflection_ids == {100 + pb}
    assert hook._skill_tracker._state("chan:b").session_skills_used == {"b-skill"}
    assert hook._lessons("chan:b") == {2: 1.0}
    assert coord._conv("chan:a").injected_principle_ids == set()
    assert hook._lessons("chan:a") == {}
    skills = db.execute("SELECT skills_used FROM trajectories").fetchall()
    assert len(skills) == 1 and "a-skill" in skills[0][0] and "b-skill" not in skills[0][0]


async def test_a_tool_call_lands_in_its_own_conversation(tmp_path):
    """Tools have no hook context; they find their conversation through
    nanobot's per-turn request contextvar."""
    from types import SimpleNamespace

    import nanobot.agent.tools.context as ctx_mod

    hook = _hook(tmp_path)
    real = ctx_mod.current_request_context
    try:
        ctx_mod.current_request_context = lambda: SimpleNamespace(
            session_key="chan:a", persist_session=True
        )
        hook.record_skill_rating("rated-in-a")
        hook._reflection_coord.add_salience(1.0)
    finally:
        ctx_mod.current_request_context = real
    assert hook._skill_tracker._state("chan:a").session_skills_used == {"rated-in-a"}
    assert hook._reflection_coord._conv("chan:a").salience_score == 1.0
    assert hook._skill_tracker._state("chan:b").session_skills_used == set()


async def test_injected_lessons_are_recorded_under_the_injecting_chat(
    tmp_path, monkeypatch
):
    """The cheatsheet path records what it injected so the boundary can move
    those lessons' trust by the session outcome — for this chat only."""
    import nano_hermes.memory.cheatsheet as cheatsheet

    async def fake_retrieve(hook, task_text, top_k=3):
        return [(7, f"lesson for {task_text}", 0.9)]

    monkeypatch.setattr(cheatsheet, "retrieve_lessons", fake_retrieve)
    hook = nano_hermes.install(
        _make_loop(tmp_path), config={"skill_stats": {"cheatsheet_enabled": True}}
    )
    injected = await _turn(hook, [], _user("deploy the app"), "chan:a")
    assert any("lesson for deploy the app" in s for s in injected), injected
    assert hook._lessons("chan:a") == {7: 0.9}
    assert hook._lessons("chan:b") == {}


async def test_boundary_lessons_come_from_the_session_that_ended(tmp_path):
    """The boundary fires on the window that crossed it — after /new, the next
    task's first prompt. Lesson extraction must read the ended session's own
    transcript, not that window."""
    import asyncio

    hook = nano_hermes.install(
        _make_loop(tmp_path), config={"skill_stats": {"cheatsheet_enabled": True}}
    )
    seen: list[list[dict]] = []

    async def capture(messages, outcome):
        seen.append(messages)

    hook._extract_cheatsheet_lesson = capture
    a = []
    await _turn(hook, a, _user("migrate the database"), "chan:a")
    await _turn(hook, [], _user("write a haiku"), "chan:a")  # /new
    await asyncio.gather(*hook._lesson_tasks)
    assert len(seen) == 1
    text = " ".join(str(m["content"]) for m in seen[0])
    assert "migrate the database" in text and "haiku" not in text


async def test_a_save_after_eviction_does_not_bring_an_early_nudge(tmp_path):
    """An evicted conversation that saves to memory used to reset the keyless
    chat instead, then re-hydrate on its next turn to interval-1 and nudge
    right after saving. A save is a fresh start for the saving chat."""
    hook = nano_hermes.install(
        _make_loop(tmp_path), config={"reflection": {"memory_save_nudge_interval": 3}}
    )
    coord = hook._reflection_coord
    coord.observe_user_turn("chan:a", window_user_turns=1)
    for i in range(40):  # push chat A out of the 32-conversation LRU
        coord.observe_user_turn(f"chan:idle{i}", window_user_turns=1)
    assert "chan:a" not in coord._conversations
    coord.note_memory_save("chan:a")
    coord.observe_user_turn("chan:a", window_user_turns=10)  # long window
    assert coord._conv("chan:a").turns_since_save == 1
    assert coord.take_save_nudge("chan:a") is None
    assert coord.peek("") is None  # the keyless chat was not touched


async def test_reading_status_creates_no_conversation_state(tmp_path):
    """Read-only access must not insert an entry: in a full LRU that evicts a
    live conversation's injected ids before its boundary can score them."""
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config={})
    coord = hook._reflection_coord
    for i in range(32):
        coord._conv(f"chan:{i}").injected_principle_ids.add(i)
    before = dict(coord._conversations)
    out = await loop.tools.get("nano_status").execute()
    assert "salience: 0.0" in out
    assert coord._conversations == before
