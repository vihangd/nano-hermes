"""Per-turn hook cost and startup cost of nano-hermes — run it on the target.

    uv run python -m evals.bench_hook --tool-calls 8          # one shape
    uv run python -m evals.bench_hook --tool-calls 1 --chats 2

Measures what nano-hermes adds to a nanobot turn: the time spent inside its
hook calls (before_iteration / after_iteration), on the window shape nanobot
really builds — history read before the user message is persisted, so the
tail is the appended current message. Also reports nano-hermes's import time
and RSS on top of an already-imported nanobot.

Run **one shape per process** and repeat it: running several shapes in one
process measured nothing reproducible in earlier audits. Numbers from a Pi
3B+ replace the x26 desktop extrapolation the audits have had to use.
Embedding providers are not called (no keys needed), so with the default
config — where embeddings run in background tasks — the hook time is what the
turn pays. Settings that embed on iteration 0 (``skill_stats.cheatsheet_enabled``,
``trajectory.inject_context``, ``reflection_scope="global"``) add one embedding
round trip per turn on top of these numbers.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _rss_mb() -> float:
    """Current RSS (not ru_maxrss, which is the peak and hides an import's cost
    when an earlier one peaked higher)."""
    try:
        pages = int(Path("/proc/self/statm").read_text().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
    except OSError:  # macOS: no /proc
        out = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
        ).stdout
        return int(out) / 1024


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool-calls", type=int, default=0)
    ap.add_argument("--chats", type=int, default=1)
    ap.add_argument("--turns", type=int, default=40)
    ap.add_argument("--cap", type=int, default=120, help="nanobot max_messages")
    args = ap.parse_args()
    logging.disable(logging.CRITICAL)

    import nanobot.agent.loop  # noqa: F401 — the host has nanobot loaded already

    rss0, t0 = _rss_mb(), time.perf_counter()
    import nano_hermes
    import_ms = (time.perf_counter() - t0) * 1000
    rss_import = _rss_mb() - rss0

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
    from conftest import _make_loop  # minimal real AgentLoop
    from nanobot.agent.hook import AgentHookContext, AgentRunHookContext
    from nanobot.agent.tools.context import (
        RequestContext,
        bind_request_context,
        reset_request_context,
    )
    from nanobot.providers.base import ToolCallRequest
    from nanobot.session.manager import Session

    loop = _make_loop(Path(tempfile.mkdtemp()))
    hook = nano_hermes.install(loop)
    sessions = [Session(key=f"bench:{c}") for c in range(args.chats)]
    per_turn: list[float] = []

    # Pass only the kwargs this nanobot accepts (HEAD dropped ``chat_id``).
    accepted = inspect.signature(loop.context.build_messages).parameters

    def build(**kwargs):
        return loop.context.build_messages(
            **{k: v for k, v in kwargs.items() if k in accepted}
        )

    async def timed(fn, ctx) -> float:
        start = time.perf_counter()
        await fn(ctx)
        return time.perf_counter() - start

    async def drive() -> None:
        # The hook sequence nanobot's runner really issues, inside the per-turn
        # request context its loop binds: per tool iteration before_iteration
        # -> (assistant message) -> before_execute_tools -> (tool results) ->
        # after_iteration, one context object each; then a final iteration;
        # then on_finally.
        for t in range(args.turns):
            session = sessions[t % args.chats]
            key = session.key
            hist = [dict(m) for m in session.get_history(max_messages=args.cap)]
            text = f"user turn {t} " + "x" * 200
            msgs = build(history=hist, current_message=text, channel="cli", chat_id="1")
            session.add_message("user", text)
            token = bind_request_context(
                RequestContext(channel="cli", chat_id="1", session_key=key)
            )
            try:
                spent = 0.0
                for i in range(args.tool_calls):
                    ctx = AgentHookContext(messages=msgs, iteration=i, session_key=key)
                    spent += await timed(hook.before_iteration, ctx)
                    call = {"id": f"c{t}_{i}", "type": "function",
                            "function": {"name": "nano_status", "arguments": "{}"}}
                    msgs.append({"role": "assistant", "content": f"calling {i}",
                                 "tool_calls": [call]})
                    ctx.tool_calls = [ToolCallRequest(
                        id=call["id"], name="nano_status", arguments={})]
                    spent += await timed(hook.before_execute_tools, ctx)
                    msgs.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": "ok " * 40})
                    spent += await timed(hook.after_iteration, ctx)
                    session.add_message("assistant", f"calling {i}", tool_calls=[call])
                    session.add_message("tool", "ok " * 40, tool_call_id=call["id"])
                ctx = AgentHookContext(
                    messages=msgs, iteration=args.tool_calls, session_key=key)
                spent += await timed(hook.before_iteration, ctx)
                reply = f"reply {t} " + "y" * 200
                msgs.append({"role": "assistant", "content": reply})
                ctx.final_content = reply
                spent += await timed(hook.after_iteration, ctx)
                spent += await timed(
                    hook.on_finally, AgentRunHookContext(messages=list(msgs)))
            finally:
                reset_request_context(token)
            session.add_message("assistant", reply)
            per_turn.append(spent * 1000)
        await hook.archiver.drain()

    asyncio.run(drive())
    p = sorted(per_turn)
    print(
        f"tool_calls={args.tool_calls} chats={args.chats} turns={args.turns}: "
        f"hook median={statistics.median(p):.3f} ms p90={p[int(len(p) * 0.9)]:.3f} ms "
        f"max={p[-1]:.3f} ms | import={import_ms:.0f} ms (+{rss_import:.1f} MB RSS) "
        f"| process RSS={_rss_mb():.0f} MB"
    )


if __name__ == "__main__":
    main()
