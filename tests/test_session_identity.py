"""Conversation identity comes from nanobot's ``AgentHookContext.session_key``.

Before this was threaded through, identity was keyed on ``id(messages)``, so
two interleaved chats crossed a session boundary on *every* switch: one bogus
``sessions`` row and one bogus trajectory per turn, plus a
``_completed_session_count`` bump that drives the LLM-backed evolution and
principle-curation cadences.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
from nanobot.agent.hook import AgentHookContext

import nano_hermes
from conftest import _make_loop
from nano_hermes.session.archiver import SessionArchiver
from nano_hermes.session.db import open_db


@pytest.fixture()
def archiver_factory(tmp_path):
    """Build a SessionArchiver on a real DB with a no-op embedder."""

    class _Chain:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def embed(self, texts):
            return [np.zeros(4, dtype=np.float32) for _ in texts]

    def _factory():
        db = open_db(tmp_path, target_dims=4)
        archiver = SessionArchiver(
            db=db,
            embedder_factory=_Chain,
            target_dims=4,
            redact_secrets=False,
        )
        return archiver, db

    return _factory


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


    async def test_new_repeating_the_previous_prompt_still_starts_a_session(
        self, tmp_path
    ):
        """``/new`` must win even when the new conversation opens with the
        exact words the old one ended on.

        Message identity is the replayed text *including* nanobot's
        ``[Message Time: ...]`` prefix, which carries
        ``datetime.now().isoformat()`` — so two real user turns never collide
        even verbatim. Stripping that prefix is what made a daily cron job
        archive its first run and nothing after: same prompt body, same key,
        zero new chunks, and yesterday's session never ended so it never
        produced a trajectory.
        """
        from nanobot.session.manager import Session

        hook = _make_hook(tmp_path)
        prompt = "run the daily standup"
        probe = Session(key="probe")
        probe.add_message("user", prompt)
        if "[Message Time:" not in _history(probe, 12, True)[0]["content"]:
            pytest.skip(
                "this nanobot does not stamp replayed user turns, so two "
                "identical prompts are indistinguishable by content — the "
                "collision is a documented residual there, see archiver.py"
            )

        async def drive():
            for _ in range(2):
                session = Session(key="cron:daily")  # /new: fresh history
                session.add_message("user", prompt)
                msgs = [{"role": "system", "content": "sys"}] + _history(
                    session, 12, True
                )
                await hook.before_iteration(
                    AgentHookContext(
                        messages=msgs, iteration=0, session_key="cron:daily"
                    )
                )
            await hook.archiver.drain()

        await drive()
        db = hook.db
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == 2
        # The finished session is stamped and written out, which is the whole
        # point of noticing the boundary at all.
        assert (
            db.execute(
                "SELECT COUNT(*) FROM sessions WHERE ended_at IS NOT NULL"
            ).fetchone()[0]
            == 1
        )
        assert db.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] == 1


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


class TestSlidingWindowHistory:
    """nanobot hands the hook a bounded window, not a growing list.

    ``context.messages`` wraps ``Session.get_history(max_messages=...)``, which
    re-slices the unconsolidated tail every turn. A length-based watermark pins
    at the cap once the window saturates, and ``range(watermark, len(messages))``
    is empty from then on — every later message was silently dropped, so
    ``session_search`` went blind on any conversation longer than the cap.
    Uses the real ``nanobot`` Session so the window shape can't drift from
    upstream without this failing.
    """

    @pytest.mark.parametrize("tool_cycles", [0, 2])
    def test_saturated_window_still_archives_every_message(
        self, archiver_factory, tool_cycles
    ):
        from nanobot.session.manager import Session

        cap, turns = 12, 20
        archiver, db = archiver_factory()
        session = Session(key="chan:1")
        expected: list[str] = []

        async def drive():
            for t in range(turns):
                session.add_message("user", f"u{t}")
                expected.append(f"u{t}")
                msgs = [{"role": "system", "content": "sys"}] + session.get_history(
                    max_messages=cap
                )
                # The replayed window first, as ``before_iteration`` does.
                archiver.archive_and_embed(msgs, "chan:1")
                appended: list[tuple[str, str]] = []
                for k in range(tool_cycles):
                    appended.append(("assistant", f"call{t}.{k}"))
                    appended.append(("tool", f"res{t}.{k}"))
                appended.append(("assistant", f"a{t}"))
                for role, content in appended:
                    msg = {"role": role, "content": content}
                    if role == "tool":
                        msg["tool_call_id"] = f"c{t}{content}"
                    msgs.append(msg)
                    expected.append(content)
                # Then the mid-turn tail, as ``after_iteration`` does: these
                # have not been through ``get_history``, so the archiver may
                # not trust their fingerprints.
                archiver.archive_and_embed(msgs, "chan:1", mid_turn=True)
                for role, content in appended:
                    session.add_message(role, content)
            await archiver.drain()

        asyncio.run(drive())

        rows = [
            r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id").fetchall()
        ]
        # Every emitted message archived exactly once, despite the window
        # having slid many times over.
        assert rows == expected
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_repeated_content_does_not_skip_intervening_messages(
        self, archiver_factory
    ):
        """The anchor is relocated nearest its old position, not at the last
        match — otherwise two identical user turns would swallow everything
        between them."""
        archiver, db = archiver_factory()

        def a(text):
            return {"role": "assistant", "content": text}

        def u(text):
            return {"role": "user", "content": text}

        async def drive():
            archiver.archive_and_embed([u("q1"), a("ok")], "chan:1")
            # Window has slid off "q1", so the anchor is no longer at
            # ``watermark - 1`` and the scan runs. "ok" recurs at the tail.
            archiver.archive_and_embed(
                [a("ok"), u("q2"), a("r2"), u("q3"), a("r3"), u("q4"), a("ok")],
                "chan:1",
            )
            await archiver.drain()

        asyncio.run(drive())
        rows = [
            r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id").fetchall()
        ]
        # Anchoring on the *last* match would resume past the end of the
        # window and archive nothing at all.
        assert rows == ["q1", "ok", "q2", "r2", "q3", "r3", "q4", "ok"]


class TestStaleSessionIdAfterNew:
    def test_current_session_id_is_none_while_shrink_is_pending(self, archiver_factory):
        """On the first turn after ``/new`` under a reused key the mapped id
        still points at the *old* session. Reporting it would attach this
        turn's reflections to a session that ``sync`` is about to stamp
        ``ended_at`` on and ``purge_older_than`` will cascade-delete."""
        archiver, db = archiver_factory()
        old = [{"role": "user", "content": "u1"}, {"role": "assistant", "content": "a1"}]
        first = archiver.ensure_session(old, "chan:1")
        assert first is not None
        assert archiver.current_session_id(old, "chan:1") == first

        fresh = [{"role": "user", "content": "brand new"}]
        assert archiver.current_session_id(fresh, "chan:1") is None
        assert archiver.ensure_session(fresh, "chan:1") != first


class TestPerConversationStateIsBounded:
    def test_idle_conversations_are_evicted(self, archiver_factory):
        """Alignment state is capped per conversation, not per process lifetime.

        ``prune_session_by_id`` cannot do it: at a boundary ``_archive`` has
        already remapped the key, so it matches nothing, and an idle channel
        crosses no boundary at all. A gateway serving hundreds of channels would
        otherwise hold a fingerprint list for every one of them forever.
        """
        from nano_hermes.session.archiver import _MAX_TRACKED_CONVERSATIONS

        archiver, _db = archiver_factory()
        keys = [f"chan:{i}" for i in range(_MAX_TRACKED_CONVERSATIONS + 8)]
        for key in keys:
            archiver.ensure_session([{"role": "user", "content": f"hi {key}"}], key)
        assert len(archiver._fingerprints) == _MAX_TRACKED_CONVERSATIONS
        assert set(archiver._fingerprints) == set(keys[-_MAX_TRACKED_CONVERSATIONS:])
        # Evicted together, so no window ever aligns against an empty list
        # inside a live session.
        assert set(archiver._pending) <= set(archiver._fingerprints)
        assert set(archiver._session_ids) <= set(archiver._fingerprints)
        # The session-id-keyed state goes too: the popped id was its last
        # reference, so nothing else would ever release it.
        live_sids = set(archiver._session_ids.values())
        assert set(archiver._session_tasks) <= live_sids
        assert set(archiver._turn_counts) <= live_sids


def _history(session, cap: int, include_timestamps: bool) -> list[dict]:
    """``get_history`` kwargs, tolerating upstream signature drift.

    nanobot HEAD dropped ``include_timestamps`` (pinned 0.2.2 still has it) but
    kept ``_MESSAGE_TIME_PREFIX_RE``, so the prefix the archiver normalizes off
    its anchors is still produced — just no longer opt-in here. Passing the
    kwarg unconditionally would make the against-HEAD compat run fail on a
    signature change rather than on a real break.
    """
    import inspect

    kwargs: dict = {"max_messages": cap}
    if "include_timestamps" in inspect.signature(session.get_history).parameters:
        kwargs["include_timestamps"] = include_timestamps
    return session.get_history(**kwargs)


def _window(
    loop,
    session,
    cap: int,
    current_message: str,
    role: str = "user",
    include_timestamps: bool = True,
) -> list[dict]:
    """The window nanobot actually hands the hook, built by the real upstream code.

    Production is ``ContextBuilder.build_messages(history=get_history(...),
    current_message=...)`` — not ``get_history`` on its own. The difference is
    load-bearing: the loop persists the user message *before* the turn
    (``_persist_user_message_early``), so the last history entry already has the
    same role, and ``build_messages`` merges the current message plus the
    ``[Runtime Context …]`` block into it (``context.py:230-255``). The window's
    last message therefore never equals the form the next turn replays. Feeding
    fixtures straight from ``get_history`` hid that for three rounds; going
    through the real builder means the fixture can't drift from upstream without
    a test failing.
    """
    import inspect

    history = [dict(m) for m in _history(session, cap, include_timestamps)]
    kwargs = {
        "history": history,
        "current_message": current_message,
        "current_role": role,
        "channel": "cli",
        # Dropped on nanobot HEAD. Filtered rather than removed so the
        # against-HEAD compat run fails on a real break, not a signature change.
        "chat_id": "1",
    }
    accepted = inspect.signature(loop.context.build_messages).parameters
    return loop.context.build_messages(
        **{k: v for k, v in kwargs.items() if k in accepted}
    )


class TestSaturatedWindowThroughTheRealHook:
    """The archiver-only tests miss the ``before_iteration`` path.

    ``before_iteration`` sees the window *without* the turn's assistant reply,
    so once ``get_history`` is pinned at the cap it is always one message
    shorter than the preceding ``after_iteration`` left. A shrink rule keyed on
    length alone therefore declares a new conversation on every turn past the
    cap: 24 turns produced 19 sessions, 18 bogus trajectories, and 4.75x
    duplicated chunks. ``include_timestamps=True`` is production's setting
    (``loop.py:1180``) and re-prefixes replayed user turns, which can orphan
    the anchor independently.
    """

    @pytest.mark.parametrize(
        "reply",
        [
            "unique",
            # A constant reply used to make the one-message anchor match the
            # wrong occurrence of itself: 48 messages emitted, 12 stored, the
            # anchor never advanced again and nothing was logged.
            "constant",
            # Both of these hold a line nanobot's replay sanitizer deletes, so
            # the raw form the mid-turn pass archives is not the form the next
            # window hands back.
            "breadcrumb",
            "tool_echo",
        ],
    )
    def test_every_message_archived_once_with_monotonic_turn_index(self, loop, reply):
        from nanobot.session.manager import Session

        replies = {
            "unique": lambda t: f"a{t}",
            "constant": lambda t: "ok",
            "breadcrumb": lambda t: f"done\n[image: /tmp/p{t}.png]",
            "tool_echo": lambda t: f"done\nmessage(to='x{t}')",
        }
        reply_for = replies[reply]
        cap, turns = 12, 24
        hook = nano_hermes.install(loop)
        session = Session(key="chan:1")
        emitted = 0

        async def drive():
            nonlocal emitted
            for t in range(turns):
                session.add_message("user", f"u{t}")
                emitted += 1
                msgs = _window(loop, session, cap, f"u{t}")
                await hook.before_iteration(
                    AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
                )
                msgs.append({"role": "assistant", "content": reply_for(t)})
                emitted += 1
                await hook.after_iteration(
                    AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
                )
                session.add_message("assistant", reply_for(t))
            await hook.archiver.drain()

        asyncio.run(drive())

        db = hook.db
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == emitted
        # turn_index is transcript order (session/browse.py) and a turn count
        # (status/tool.py), so it has to be per session and monotonic. Taken
        # from the window index it ran 1..cap and then repeated.
        turn_indexes = [r[0] for r in db.execute("SELECT turn_index FROM chunks ORDER BY id")]
        assert turn_indexes == list(range(1, emitted + 1))
        # The live merged tip is cleaned before the insert: the corpus that FTS5
        # and the embedder see must not carry a runtime-metadata block per turn,
        # nor the prompt twice.
        contents = [r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id")]
        assert [c for c in contents if "[Runtime Context" in c] == []
        assert contents[0].endswith("u0")
        assert contents[0].count("u0") == 1

    def test_user_turn_is_archived_before_the_model_answers(self, loop):
        """``before_iteration`` archives, not just ``after_iteration``.

        The window it holds has been through ``get_history``, so those messages
        are in the form the archiver can fingerprint and re-find next turn —
        only the reply appended later has to be adopted by role. It also means a
        tool called *during* the turn can already search the turn that asked for
        it, and a run that dies before ``after_iteration`` still archived its
        user turn.
        """
        hook = nano_hermes.install(loop)
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
        ]

        async def drive():
            ctx = AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
            await hook.before_iteration(ctx)
            msgs.append({"role": "assistant", "content": "answer"})
            await hook.after_iteration(ctx)
            # Second turn, interrupted: only before_iteration runs. The session
            # is mapped by now, so the lazy bootstrap in _sync_session no longer
            # archives and this depends on before_iteration doing it.
            msgs.append({"role": "user", "content": "what did I ask"})
            await hook.before_iteration(
                AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
            )
            await hook.archiver.drain()

        asyncio.run(drive())
        rows = [r[0] for r in hook.db.execute("SELECT content FROM chunks ORDER BY id")]
        assert rows == ["first", "answer", "what did I ask"]

    def test_tool_loop_iterations_do_not_restart_the_conversation(self, loop):
        """``before_iteration`` fires once per *iteration*, not once per turn.

        nanobot's runner builds the window once and then mutates that same list
        in place — the assistant message and each tool result are appended to it
        (``runner.py:482``, ``:511``) before the next iteration's
        ``before_iteration``. So from iteration 1 on, the window holds raw
        mid-turn messages that ``get_history``'s replay sanitizer will rewrite.
        Confirming their fingerprints there made the next turn's window align
        nowhere: new session, full re-archive, on every tool-using turn.
        """
        from nanobot.session.manager import Session

        cap, turns = 12, 8
        hook = nano_hermes.install(loop)
        session = Session(key="chan:1")
        expected: list[str] = []

        async def drive():
            for t in range(turns):
                session.add_message("user", f"u{t}")
                expected.append(f"u{t}")
                msgs = _window(loop, session, cap, f"u{t}")
                appended: list[tuple[str, str]] = []
                # Iteration 0: fresh window, then the tool call and its result.
                # The reply carries a line the replay sanitizer deletes.
                per_iteration = [
                    [
                        ("assistant", f"calling\nmessage(to='x{t}')"),
                        ("tool", f"res{t}"),
                    ],
                    [("assistant", f"done\n[image: /tmp/p{t}.png]")],
                ]
                for it, produced in enumerate(per_iteration):
                    ctx = AgentHookContext(
                        messages=msgs, iteration=it, session_key="chan:1"
                    )
                    await hook.before_iteration(ctx)
                    for role, content in produced:
                        msg = {"role": role, "content": content}
                        if role == "tool":
                            msg["tool_call_id"] = f"c{t}"
                        msgs.append(msg)
                        expected.append(content)
                        appended.append((role, content))
                    await hook.after_iteration(ctx)
                for role, content in appended:
                    session.add_message(role, content)
            await hook.archiver.drain()

        asyncio.run(drive())

        db = hook.db
        rows = [r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id")]
        # Replayed user turns carry a ``[Message Time: ...]`` prefix, so compare
        # on the tail; what matters is one row per emitted message, in order.
        assert [r.rsplit("\n", 1)[-1] for r in rows] == [
            e.rsplit("\n", 1)[-1] for e in expected
        ]
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        turn_indexes = [
            r[0] for r in db.execute("SELECT turn_index FROM chunks ORDER BY id")
        ]
        assert turn_indexes == list(range(1, len(expected) + 1))

    @pytest.mark.parametrize("include_timestamps", [False, True])
    def test_one_conversation_stays_one_session(self, loop, include_timestamps):
        from nanobot.session.manager import Session

        cap, turns = 12, 24
        hook = nano_hermes.install(loop)
        session = Session(key="chan:1")
        emitted = 0

        async def drive():
            nonlocal emitted
            for t in range(turns):
                session.add_message("user", f"u{t}")
                emitted += 1
                msgs = _window(
                    loop, session, cap, f"u{t}", include_timestamps=include_timestamps
                )
                await hook.before_iteration(
                    AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
                )
                msgs.append({"role": "assistant", "content": f"a{t}"})
                emitted += 1
                await hook.after_iteration(
                    AgentHookContext(messages=msgs, iteration=0, session_key="chan:1")
                )
                session.add_message("assistant", f"a{t}")
            await hook.archiver.drain()

        asyncio.run(drive())

        db = hook.db
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        # Every message archived exactly once — no gaps, no duplicates.
        assert db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] == emitted
        assert db.execute("SELECT COUNT(*) FROM trajectories").fetchone()[0] == 0


class TestMidTurnMessagesAreAdoptedNotDuplicated:
    """A message archived mid-turn is re-read in a form that is not equal to it.

    ``after_iteration`` archives the reply the model just appended; the next
    turn sees that same reply after ``get_history`` has run it through nanobot's
    replay sanitizer, which deletes local-image breadcrumbs and tool-call echo
    lines outright. Fingerprinting the raw form and looking for it in the
    replayed window therefore never matched: the window aligned nowhere, was
    re-archived from zero, and ``chunks`` grew quadratically (12 messages
    emitted, 27 stored in six turns). Mid-turn messages are adopted by role
    instead, and their replayed fingerprints taken from the window.
    """

    @pytest.mark.parametrize(
        "reply",
        [
            "plain reply",
            "answer\n[image: /tmp/shot.png]",  # sanitizer drops that line
            "answer\nmessage(to='alice')",  # so does it drop a tool-call echo
        ],
    )
    def test_replay_transformed_reply_is_not_re_archived(self, archiver_factory, reply):
        from nanobot.session.manager import Session

        archiver, db = archiver_factory()
        session = Session(key="chan:1")

        async def drive():
            for turn, question in enumerate(("q1", "q2")):
                session.add_message("user", question)
                msgs = [{"role": "system", "content": "sys"}] + _history(
                    session, 12, True
                )
                archiver.archive_and_embed(msgs, "chan:1")
                if turn == 0:
                    # after_iteration: the reply is appended in its raw form,
                    # which get_history will not hand back unchanged.
                    msgs.append({"role": "assistant", "content": reply})
                    archiver.archive_and_embed(msgs, "chan:1", mid_turn=True)
                    session.add_message("assistant", reply)
            await archiver.drain()

        asyncio.run(drive())
        rows = [r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id")]
        assert len(rows) == 3, rows
        assert rows[1] == reply
        assert db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1

    def test_redacted_secret_does_not_orphan_the_anchor(self, tmp_path):
        """Anchors are stored pre-redaction; comparing a stored *redacted*
        anchor against freshly-extracted raw text would never match."""
        from unittest.mock import AsyncMock, MagicMock

        chain = MagicMock()
        chain.__aenter__ = AsyncMock(return_value=chain)
        chain.__aexit__ = AsyncMock(return_value=False)
        chain.embed = AsyncMock(return_value=[])
        db = open_db(tmp_path, target_dims=4)
        archiver = SessionArchiver(
            db=db, embedder_factory=lambda: chain, target_dims=4, redact_secrets=True
        )
        secret = "token sk-abcdefghijklmnopqrstuvwxyz1234567890"
        archiver.ensure_session([{"role": "user", "content": secret}], "chan:1")
        archiver.ensure_session(
            [{"role": "user", "content": secret}, {"role": "assistant", "content": "ok"}],
            "chan:1",
        )
        rows = [r[0] for r in db.execute("SELECT content FROM chunks ORDER BY id")]
        assert len(rows) == 2 and rows[1] == "ok"
        assert "abcdefghijklmnopqrstuvwxyz" not in rows[0]
