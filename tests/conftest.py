"""Shared fixtures and helpers for nano-hermes test suite."""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus

from nano_hermes.hook import NanoHermesHook


# ---------------------------------------------------------------------------
# Loop factory + fixture
# ---------------------------------------------------------------------------

def _make_loop(tmp_path: Path) -> AgentLoop:
    """Minimal AgentLoop. Mirrors nanobot/tests/agent/test_unified_session.py.

    The patches avoid heavy subsystems (SessionManager, SubagentManager,
    Dream) during __init__ — everything else we need (MemoryStore,
    ContextBuilder, ToolRegistry, _extra_hooks) is constructed for real.
    """
    bus = MessageBus()
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = MagicMock(max_tokens=4096)

    with patch("nanobot.agent.loop.SessionManager"), \
         patch("nanobot.agent.loop.SubagentManager") as sub_mgr, \
         patch("nanobot.agent.loop.Dream"):
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

_FAKE_KEYWORDS: list[tuple[str, np.ndarray]] = [
    ("duckduckgo", _FAKE_VEC_SEARCH),
    ("search the web", _FAKE_VEC_SEARCH),
    ("web search", _FAKE_VEC_SEARCH),
    ("arxiv", _FAKE_VEC_ACADEMIC),
    ("academic", _FAKE_VEC_ACADEMIC),
    ("papers", _FAKE_VEC_ACADEMIC),
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
