"""Shared fixtures and helpers for nano-hermes test suite."""
from __future__ import annotations

import contextlib
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nanobot.agent import loop as _loop_mod
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus

from nano_hermes.hook import NanoHermesHook


def _make_loop_attrs_settable(*names: str) -> None:
    """Let tests assign read-only ``AgentLoop`` properties across nanobot versions.

    Pinned nanobot 0.2.2 exposes ``provider`` / ``model`` as plain attributes,
    so the suite injects fakes with ``hook._loop.provider = MagicMock()``.
    Upstream turned both into read-only properties resolving through
    ``runtime_resolver``, which makes every such assignment raise
    AttributeError — swamping the against-HEAD compat run with ~100 failures
    that say nothing about nano-hermes (production code never assigns these;
    it only ever reads them).

    Re-adding setters in the test process keeps that compat signal meaningful
    without rewriting every call site. Reads fall through to the real resolver
    whenever a test hasn't overridden the value.
    """
    unset = object()

    for name in names:
        prop = getattr(AgentLoop, name, None)
        if not isinstance(prop, property) or prop.fset is not None:
            continue  # settable already (0.2.2) — nothing to do
        slot = f"_test_{name}"

        def _get(self, _fget=prop.fget, _slot=slot):
            # Sentinel, not None: some tests deliberately set these to None to
            # exercise the "not configured" path, which must not fall through
            # to the real resolver.
            override = getattr(self, _slot, unset)
            return _fget(self) if override is unset else override

        def _set(self, value, _slot=slot):
            setattr(self, _slot, value)

        setattr(AgentLoop, name, property(_get, _set))


_make_loop_attrs_settable("provider", "model")


# ---------------------------------------------------------------------------
# Loop factory + fixture
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: Path) -> AgentLoop:
    """Minimal AgentLoop. Mirrors nanobot/tests/agent/test_unified_session.py.

    The patches avoid heavy subsystems (SessionManager, SubagentManager, and
    the background dream/cron worker) during __init__ — everything else we
    need (MemoryStore, ContextBuilder, ToolRegistry, _extra_hooks) is
    constructed for real.

    Version-tolerant across nanobot releases: 0.2.0 spins up a ``Dream``
    object, while 0.2.1 (commit d1a94dae) replaced it with ``CronService``.
    We patch whichever the installed nanobot actually exposes so the suite
    stays green on both.
    """
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock(max_tokens=4096)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch("nanobot.agent.loop.SessionManager"))
        sub_mgr = stack.enter_context(patch("nanobot.agent.loop.SubagentManager"))
        for sym in ("Dream", "CronService"):
            if hasattr(_loop_mod, sym):
                stack.enter_context(patch(f"nanobot.agent.loop.{sym}"))
        sub_mgr.return_value.cancel_by_session = AsyncMock(return_value=0)
        return AgentLoop(bus=bus, provider=provider, workspace=tmp_path)


@pytest.fixture
def loop(tmp_path: Path) -> AgentLoop:
    return _make_loop(tmp_path)


# ---------------------------------------------------------------------------
# DB/chunk helpers
# ---------------------------------------------------------------------------

def _existing_hook(loop: AgentLoop) -> NanoHermesHook:
    for h in loop._extra_hooks:
        if isinstance(h, NanoHermesHook):
            return h
    raise RuntimeError("install() wasn't called on this loop")


def _seed_chunk(loop: AgentLoop, content: str) -> int:
    """Insert one session + one chunk via the hook's db; returns chunk_id."""
    hook = _existing_hook(loop)
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        ("test:1", 1_700_000_000.0),
    )
    session_id = cur.lastrowid
    cur = hook.db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (session_id, content, 1_700_000_000.0),
    )
    hook.db.commit()
    return int(cur.lastrowid)


# ---------------------------------------------------------------------------
# Embedding key helper
# ---------------------------------------------------------------------------

def _unset_embedding_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Bundled skill helpers
# ---------------------------------------------------------------------------

_BUNDLED_SKILLS = Path(__file__).parent.parent / "examples" / "skills"


def _copy_bundled_skill(name: str, workspace: Path) -> Path:
    src = _BUNDLED_SKILLS / name / "SKILL.md"
    assert src.exists(), f"bundled skill missing at {src}"
    dst_dir = workspace / "skills" / name
    dst_dir.mkdir(parents=True)
    dst = dst_dir / "SKILL.md"
    shutil.copy(src, dst)
    return dst


# ---------------------------------------------------------------------------
# Fake embedding infrastructure
# ---------------------------------------------------------------------------

_FAKE_DIMS = 512
_FAKE_VEC_SEARCH = np.zeros(_FAKE_DIMS, dtype=np.float32)
_FAKE_VEC_SEARCH[0] = 1.0
_FAKE_VEC_ACADEMIC = np.zeros(_FAKE_DIMS, dtype=np.float32)
_FAKE_VEC_ACADEMIC[1] = 1.0
_FAKE_VEC_UNRELATED = np.zeros(_FAKE_DIMS, dtype=np.float32)
_FAKE_VEC_UNRELATED[2] = 1.0

# Keep only markers UNIQUE to our test skills. Earlier entries like
# "search the web" / "web search" collide with nanobot 0.2.0's built-in
# 'my' skill, whose description contains the phrase "search the web?" as
# a diagnostic example. The fake embedder must distinguish test skills
# from built-ins; substring-matching on common phrases doesn't.
_FAKE_KEYWORDS: list[tuple[str, np.ndarray]] = [
    ("duckduckgo", _FAKE_VEC_SEARCH),
    ("arxiv", _FAKE_VEC_ACADEMIC),
]


async def _fake_embed(self, texts):  # signature: (self, texts)
    out = []
    for t in texts:
        matched = _FAKE_VEC_UNRELATED
        tl = t.lower()
        for kw, vec in _FAKE_KEYWORDS:
            if kw in tl:
                matched = vec
                break
        out.append(matched)
    return out


def _patch_embedding(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace EmbeddingChain.embed with the deterministic fake above.

    aiohttp.ClientSession is still created inside ``async with`` but
    never used, since ``embed`` is intercepted before ``_call`` runs.
    """
    monkeypatch.setattr(
        "nano_hermes.embedding.chain.EmbeddingChain.embed",
        _fake_embed,
    )
