"""Conversation identity comes from nanobot's ``AgentHookContext.session_key``.

Before this was threaded through, identity was keyed on ``id(messages)``, so
two interleaved chats crossed a session boundary on *every* switch: one bogus
``sessions`` row and one bogus trajectory per turn, plus a
``_completed_session_count`` bump that drives the LLM-backed evolution and
principle-curation cadences.
"""
from __future__ import annotations

from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop


def _make_hook(tmp_path):
    return nano_hermes.install(_make_loop(tmp_path), config={})


async def _turn(hook, msgs, text, session_key):
    msgs.append({"role": "user", "content": text})
    ctx = AgentHookContext(iteration=0, messages=msgs, session_key=session_key)
    await hook.before_iteration(ctx)
    msgs.append({"role": "assistant", "content": "ack"})
    await hook.after_iteration(ctx)


def _counts(hook):
    return (
        hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0],
        hook.db.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0],
    )


class TestInterleavedConversations:
    async def test_two_chats_produce_two_sessions(self, tmp_path):
        hook = _make_hook(tmp_path)
        a, b = [], []
        for i in range(20):
            await _turn(hook, a, f"a {i}", "cli:alice")
            await _turn(hook, b, f"b {i}", "cli:bob")
        await hook.archiver.drain(timeout=2.0)
        # 0 trajectories is the intended semantics, not a gap: neither chat
        # was /new-ed or compacted, so neither has a session boundary yet.
        # A boundary is the only thing that writes a trajectory — see the
        # archiver module docstring.
        assert _counts(hook) == (2, 0)
        # The boundary block is what schedules evolution + principle curation.
        assert hook._completed_session_count == 0

    async def test_distinct_keys_do_not_share_a_session(self, tmp_path):
        hook = _make_hook(tmp_path)
        a, b = [], []
        await _turn(hook, a, "hi", "cli:alice")
        await _turn(hook, b, "hi", "cli:bob")
        sids = [
            r[0]
            for r in hook.db.execute(
                "SELECT DISTINCT session_id FROM chunks ORDER BY session_id"
            )
        ]
        assert len(sids) == 2


class TestSessionKeyIsRecorded:
    async def test_real_key_lands_in_sessions(self, tmp_path):
        hook = _make_hook(tmp_path)
        await _turn(hook, [], "hi", "discord:12345")
        keys = [r[0] for r in hook.db.execute("SELECT session_key FROM sessions")]
        assert keys == ["discord:12345"]

    async def test_missing_key_falls_back_to_synthetic(self, tmp_path):
        hook = _make_hook(tmp_path)
        await _turn(hook, [], "hi", None)
        keys = [r[0] for r in hook.db.execute("SELECT session_key FROM sessions")]
        assert len(keys) == 1
        assert keys[0].startswith("loop:")


class TestFreshConversationUnderAReusedKey:
    async def test_new_starts_a_second_session(self, tmp_path):
        """``/new`` in the same channel reuses the key; the list shrinks.

        The shrink rule in ``_archive`` is the only thing that notices.
        """
        hook = _make_hook(tmp_path)
        msgs = []
        for i in range(3):
            await _turn(hook, msgs, f"before {i}", "cli:direct")
        msgs.clear()  # /new
        await _turn(hook, msgs, "after", "cli:direct")
        assert hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2


class TestFreshListObjectPerTurn:
    async def test_same_key_new_list_each_turn_stays_one_session(self, tmp_path):
        """A caller that rebuilds the history list each turn must not restart.

        This is what keying on ``id(messages)`` got wrong: the list object is
        not conversation identity, only the key is.
        """
        hook = _make_hook(tmp_path)
        history: list[dict] = []
        for i in range(10):
            history = list(history)  # fresh object, same conversation
            await _turn(hook, history, f"turn {i}", "cli:direct")
        assert hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1


class TestKeylessFallback:
    async def test_keyless_interleaving_still_over_counts(self, tmp_path):
        """Documented limitation, not an aspiration.

        With no ``session_key`` the list object is the only identity available,
        so the coordinator treats any switch of list as "previous conversation
        finished". Two interleaved keyless chats therefore still cross a
        boundary on every switch. Reachable in production — nanobot passes
        ``session.key if session else None`` — but nothing in the hook context
        can tell interleaving apart from two consecutive chats, so the
        alternative is losing trajectories for keyless callers entirely.
        """
        hook = _make_hook(tmp_path)
        a, b = [], []
        for i in range(5):
            await _turn(hook, a, f"a {i}", None)
            await _turn(hook, b, f"b {i}", None)
        # One bogus boundary per switch, and each one prunes the archiver's
        # cached session id, so the next turn on that list opens yet another
        # row: 10 sessions for 10 turns. This is what the keyed path fixes.
        sessions = hook.db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        assert sessions == 10
        assert hook._completed_session_count == 9
