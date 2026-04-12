"""Smoke tests for ``nano_hermes.install()`` and its two tools.

Runs against a real ``nanobot.agent.loop.AgentLoop`` with a mocked LLM
provider. The whole point is to catch interface drift between our code
and nanobot's actual classes *before* we run anything on the Pi:

- ``AgentHook`` lifecycle signatures (``NanoHermesHook`` lands in
  ``loop._extra_hooks`` without crashing its __init__).
- ``Tool`` ABC wiring (``memory_patch`` / ``session_search`` register as
  real Tool subclasses and ``loop.tools`` can find them by name).
- ``BudgetedMemory`` wraps the SAME ``MemoryStore`` instance that
  nanobot's ``ContextBuilder`` uses — no parallel state.
- SQLite schema bootstraps under ``<workspace>/nano_hermes/`` with
  FTS5 + sqlite-vec extensions loaded.
- ``memory_patch`` add/replace/remove work and enforce budgets with
  actionable overflow errors.
- ``session_search`` degrades cleanly to FTS5-only when every embedding
  provider is unreachable.
- ``hybrid_search`` returns the right chunk given a hand-crafted vector
  — validates the ``sqlite-vec`` vec0 MATCH syntax in-process.
"""
from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from nanobot.agent.hook import AgentHookContext
from nanobot.agent.loop import AgentLoop
from nanobot.bus.queue import MessageBus

import nano_hermes
from nano_hermes.hook import NanoHermesHook
from nano_hermes.memory.tool import MemoryPatchTool
from nano_hermes.reflect.salience import (
    correction_score,
    last_user_text,
    tool_burst_score,
)
from nano_hermes.reflect.tool import ReflectTool
from nano_hermes.session.search import SessionSearchTool, hybrid_search
from nano_hermes.skills.tool import SkillSearchTool


# ---------------------------------------------------------------------------
# Fixtures
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


def _existing_hook(loop: AgentLoop) -> NanoHermesHook:
    for h in loop._extra_hooks:
        if isinstance(h, NanoHermesHook):
            return h
    raise RuntimeError("install() wasn't called on this loop")


# ---------------------------------------------------------------------------
# install() wiring
# ---------------------------------------------------------------------------

class TestInstall:
    def test_hook_lands_in_extra_hooks(self, loop: AgentLoop) -> None:
        hook = nano_hermes.install(loop)
        assert isinstance(hook, NanoHermesHook)
        assert hook in loop._extra_hooks

    def test_tools_registered(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        assert "memory_patch" in loop.tools
        assert "session_search" in loop.tools
        assert "skill_search" in loop.tools
        assert "reflect" in loop.tools
        assert isinstance(loop.tools.get("memory_patch"), MemoryPatchTool)
        assert isinstance(loop.tools.get("session_search"), SessionSearchTool)
        assert isinstance(loop.tools.get("skill_search"), SkillSearchTool)
        assert isinstance(loop.tools.get("reflect"), ReflectTool)

    def test_state_db_lives_under_workspace(
        self, loop: AgentLoop, tmp_path: Path
    ) -> None:
        nano_hermes.install(loop)
        assert (tmp_path / "nano_hermes" / "state.db").exists()

    def test_budgeted_memory_wraps_nanobot_store(self, loop: AgentLoop) -> None:
        """Proves we're not holding a parallel MemoryStore — writes via
        memory_patch land in the exact same file ContextBuilder reads from."""
        hook = nano_hermes.install(loop)
        assert hook.budgeted_memory.store is loop.context.memory

    def test_config_override_applies_to_budgets(self, loop: AgentLoop) -> None:
        hook = nano_hermes.install(
            loop, config={"memory": {"memory_md_chars": 50}}
        )
        assert hook.budgeted_memory.budgets.memory_md_chars == 50
        # other budgets keep their defaults
        assert hook.budgeted_memory.budgets.user_md_chars == 1375


# ---------------------------------------------------------------------------
# memory_patch tool
# ---------------------------------------------------------------------------

class TestMemoryPatch:
    async def test_add_persists_via_nanobot_store(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")
        assert tool is not None

        out = await tool.execute(
            slot="memory",
            action="add",
            content="user prefers pickles on sandwiches",
        )
        assert out.startswith("ok")
        assert "pickles" in loop.context.memory.read_memory()

    async def test_over_budget_returns_actionable_error(
        self, loop: AgentLoop
    ) -> None:
        nano_hermes.install(loop, config={"memory": {"memory_md_chars": 50}})
        tool = loop.tools.get("memory_patch")

        out = await tool.execute(slot="memory", action="add", content="x" * 120)
        assert out.startswith("Error")
        # both the budget and the shortfall should surface
        assert "50" in out
        assert "70" in out

    async def test_replace_flow(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        await tool.execute(slot="user", action="add", content="prefers tabs")
        out = await tool.execute(
            slot="user", action="replace", needle="tabs", replacement="spaces"
        )
        assert out.startswith("ok")
        user_md = loop.context.memory.read_user()
        assert "spaces" in user_md
        assert "tabs" not in user_md

    async def test_remove_missing_needle_returns_error(
        self, loop: AgentLoop
    ) -> None:
        nano_hermes.install(loop)
        tool = loop.tools.get("memory_patch")

        out = await tool.execute(slot="soul", action="remove", needle="nope")
        assert out.startswith("Error")
        assert "soul" in out


# ---------------------------------------------------------------------------
# Session schema — FTS5 trigger + sqlite-vec vec0 MATCH
# ---------------------------------------------------------------------------

class TestSessionSchema:
    def test_fts_trigger_mirrors_inserts(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        _seed_chunk(loop, "remember the spice melange")
        hook = _existing_hook(loop)

        rows = hook.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'melange'"
        ).fetchall()
        assert rows, "chunks_ai trigger did not mirror the insert into chunks_fts"

    def test_vec0_match_roundtrip(self, loop: AgentLoop) -> None:
        """Validates sqlite-vec vec0 MATCH/k syntax with a hand-crafted vector."""
        nano_hermes.install(loop)
        chunk_id = _seed_chunk(loop, "the agent can cook pasta")
        hook = _existing_hook(loop)

        dims = hook.config.embedding.target_dims
        vec = np.ones(dims, dtype=np.float32)
        vec /= np.linalg.norm(vec)

        hook.db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (chunk_id, vec.tobytes()),
        )
        hook.db.commit()

        hits = hybrid_search(hook.db, "pasta", vec, hook.config.retrieval)
        assert hits, "hybrid_search returned nothing — check vec0 MATCH syntax"
        assert hits[0].chunk_id == chunk_id


# ---------------------------------------------------------------------------
# session_search tool — FTS5 fallback path
# ---------------------------------------------------------------------------

class TestSessionSearchFallback:
    async def test_falls_back_to_fts_when_all_providers_unreachable(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Strip every provider key; the embedding chain will raise
        # AllProvidersFailed and the tool should degrade to FTS5-only.
        monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
        monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        nano_hermes.install(loop)
        _seed_chunk(loop, "spice melange caravan")

        tool = loop.tools.get("session_search")
        assert tool is not None
        out = await tool.execute(query="melange")
        assert "melange" in out, f"FTS fallback returned nothing: {out!r}"


# ---------------------------------------------------------------------------
# after_iteration archival path
# ---------------------------------------------------------------------------

def _unset_embedding_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPINFRA_API_KEY", raising=False)
    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)


class TestArchiver:
    """``after_iteration`` should persist new messages and keep FTS current."""

    async def test_archive_inserts_chunks_and_populates_fts(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "what's the capital of Nauru"},
                {"role": "assistant", "content": "Yaren District is the de facto seat."},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        # drain the background embed task (will no-op because keys were unset)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT role, content FROM chunks ORDER BY id"
        ).fetchall()
        assert len(rows) == 2
        assert rows[0] == ("user", "what's the capital of Nauru")
        assert rows[1][0] == "assistant"
        assert "Yaren" in rows[1][1]

        matches = hook.db.execute(
            "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'Nauru'"
        ).fetchall()
        assert matches, "FTS trigger did not pick up archived chunks"

    async def test_tool_only_assistant_messages_are_skipped(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                # assistant fires a tool call with no text — nothing to archive
                {"role": "assistant", "content": None, "tool_calls": [{"name": "x"}]},
                {"role": "user", "content": "carry on"},
                # assistant with empty string — also skip
                {"role": "assistant", "content": "   "},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT role, content FROM chunks ORDER BY id"
        ).fetchall()
        assert rows == [("user", "carry on")]

    async def test_watermark_prevents_reinsert_on_second_iteration(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [
            {"role": "user", "content": "first question"},
        ]
        ctx1 = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx1)
        await hook.after_iteration(ctx1)

        # second iteration — same list, one new message appended
        messages.append({"role": "assistant", "content": "first answer"})
        ctx2 = AgentHookContext(iteration=1, messages=messages)
        await hook.after_iteration(ctx2)
        await hook.archiver.drain()

        rows = hook.db.execute(
            "SELECT content FROM chunks ORDER BY id"
        ).fetchall()
        assert [r[0] for r in rows] == ["first question", "first answer"], (
            f"watermark leak — archived {len(rows)} rows: {rows}"
        )

        # sanity: exactly one session row (both iterations share the list id)
        session_count = hook.db.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]
        assert session_count == 1

    async def test_archived_content_is_findable_via_session_search(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end: archive a turn, then retrieve it with session_search
        over the FTS fallback path (no embedding network needed)."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        ctx = AgentHookContext(
            iteration=0,
            messages=[
                {"role": "user", "content": "planning a trip to Reykjavik"},
                {"role": "assistant", "content": "Reykjavik is great in winter."},
            ],
        )
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        await hook.archiver.drain()

        tool = loop.tools.get("session_search")
        assert tool is not None
        out = await tool.execute(query="Reykjavik")
        assert "Reykjavik" in out, f"search returned: {out!r}"


# ---------------------------------------------------------------------------
# Ported skill: examples/skills/duckduckgo-search/SKILL.md
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


class TestSkillCoexistence:
    """A ported SKILL.md under ``workspace/skills/`` is picked up by
    nanobot's SkillsLoader, and nano-hermes tools stay registered
    alongside it — confirms there's no collision between our
    ``loop.tools`` registration and nanobot's skill injection pipeline.
    """

    def test_skills_loader_discovers_ported_skill(
        self, tmp_path: Path
    ) -> None:
        _copy_bundled_skill("duckduckgo-search", tmp_path)

        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)

        entries = loop.context.skills.list_skills(filter_unavailable=False)
        names = [e["name"] for e in entries]
        assert "duckduckgo-search" in names

        summary = loop.context.skills.build_skills_summary()
        assert "duckduckgo-search" in summary
        assert "DuckDuckGo" in summary  # description text surfaces in XML

        # nano-hermes tools survive skill discovery
        assert "memory_patch" in loop.tools
        assert "session_search" in loop.tools

    def test_frontmatter_parses_with_nanobot_parser(
        self, tmp_path: Path
    ) -> None:
        """Nanobot's SKILL.md frontmatter parser is line-based (not real
        YAML). Confirms our description survives the round-trip — easy
        to break by accidentally using nested YAML values."""
        _copy_bundled_skill("duckduckgo-search", tmp_path)

        loop = _make_loop(tmp_path)
        meta = loop.context.skills.get_skill_metadata("duckduckgo-search")
        assert meta is not None
        assert "DuckDuckGo" in meta.get("description", "")

    def test_skill_reports_available_without_requirements(
        self, tmp_path: Path
    ) -> None:
        """No ``requires`` in the frontmatter → skill is always available
        and ``build_skills_summary`` marks it ``available="true"``."""
        _copy_bundled_skill("duckduckgo-search", tmp_path)

        loop = _make_loop(tmp_path)
        summary = loop.context.skills.build_skills_summary()
        # the skill block for duckduckgo-search must be marked available
        assert 'available="true"' in summary
        # and the loader's filter-available pass keeps it in the list
        available_names = [
            e["name"]
            for e in loop.context.skills.list_skills(filter_unavailable=True)
        ]
        assert "duckduckgo-search" in available_names


# ---------------------------------------------------------------------------
# skill_search (Voyager-style embedding retrieval)
# ---------------------------------------------------------------------------

# Fake embedding map used to test the skill indexer without hitting the
# network. Substring matches (case-insensitive) select a vector; unmatched
# text falls through to _FAKE_VEC_UNRELATED.
_FAKE_DIMS = 512
_FAKE_VEC_SEARCH = np.zeros(_FAKE_DIMS, dtype=np.float32); _FAKE_VEC_SEARCH[0] = 1.0
_FAKE_VEC_ACADEMIC = np.zeros(_FAKE_DIMS, dtype=np.float32); _FAKE_VEC_ACADEMIC[1] = 1.0
_FAKE_VEC_UNRELATED = np.zeros(_FAKE_DIMS, dtype=np.float32); _FAKE_VEC_UNRELATED[2] = 1.0

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


class TestSkillSearch:
    """Voyager-style semantic retrieval over nanobot's SkillsLoader."""

    async def test_search_returns_relevant_skill(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _copy_bundled_skill("duckduckgo-search", tmp_path)
        _patch_embedding(monkeypatch)
        nano_hermes.install(loop)

        tool = loop.tools.get("skill_search")
        assert tool is not None
        out = await tool.execute(query="I want to search the web")
        assert "duckduckgo-search" in out
        assert "DuckDuckGo" in out  # description surfaces alongside the name

    async def test_search_ranks_by_similarity(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Competing skill arxiv-search should lose to duckduckgo-search on
        a web-search query — orthogonal fake vectors make this
        deterministic."""
        _copy_bundled_skill("duckduckgo-search", tmp_path)
        arxiv_dir = tmp_path / "skills" / "arxiv-search"
        arxiv_dir.mkdir(parents=True)
        (arxiv_dir / "SKILL.md").write_text(
            "---\n"
            "name: arxiv-search\n"
            "description: Find academic papers on arxiv\n"
            "---\n\nBody.\n"
        )
        _patch_embedding(monkeypatch)
        nano_hermes.install(loop)

        tool = loop.tools.get("skill_search")
        out = await tool.execute(query="I want to search the web")
        lines = out.splitlines()
        assert "duckduckgo-search" in lines[0], (
            f"expected duckduckgo-search first, got:\n{out}"
        )

    async def test_search_fails_cleanly_when_no_providers(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _copy_bundled_skill("duckduckgo-search", tmp_path)
        _unset_embedding_keys(monkeypatch)
        nano_hermes.install(loop)

        tool = loop.tools.get("skill_search")
        out = await tool.execute(query="anything")
        assert out.startswith("Error")
        assert "embedding" in out.lower()

    async def test_unchanged_skill_is_not_reindexed(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Note: nanobot's SkillsLoader auto-discovers its own BUILTIN_SKILLS_DIR
        # in addition to workspace/skills/, so the indexer sees both. We
        # assert on the duckduckgo-search row specifically, not on totals.
        _copy_bundled_skill("duckduckgo-search", tmp_path)
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        report = await hook.skill_indexer.refresh()
        assert report["reindexed"] >= 1  # at least ddg, possibly + builtins
        assert report["removed"] == 0
        first_indexed_at = hook.db.execute(
            "SELECT indexed_at FROM skill_stats WHERE name = ?",
            ("duckduckgo-search",),
        ).fetchone()[0]
        assert first_indexed_at is not None

        # Same content → second refresh reindexes nothing and the stored
        # indexed_at timestamp is unchanged.
        report2 = await hook.skill_indexer.refresh()
        assert report2["reindexed"] == 0
        assert report2["removed"] == 0
        second_indexed_at = hook.db.execute(
            "SELECT indexed_at FROM skill_stats WHERE name = ?",
            ("duckduckgo-search",),
        ).fetchone()[0]
        assert second_indexed_at == first_indexed_at

    async def test_removed_skill_is_cleaned_up(
        self,
        loop: AgentLoop,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        skill_path = _copy_bundled_skill("duckduckgo-search", tmp_path)
        _patch_embedding(monkeypatch)
        hook = nano_hermes.install(loop)

        await hook.skill_indexer.refresh()
        assert hook.db.execute(
            "SELECT COUNT(*) FROM skill_stats WHERE name = ?",
            ("duckduckgo-search",),
        ).fetchone()[0] == 1

        shutil.rmtree(skill_path.parent)
        report = await hook.skill_indexer.refresh()
        assert report["removed"] >= 1  # at least ddg
        # ddg specifically is gone from both skill_stats and skill_vec
        assert hook.db.execute(
            "SELECT COUNT(*) FROM skill_stats WHERE name = ?",
            ("duckduckgo-search",),
        ).fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Reflexion: reflect tool + salience nudges + per-session injection
# ---------------------------------------------------------------------------

class TestSalienceHeuristics:
    """Pure-function unit tests for reflect/salience.py."""

    def test_tool_burst_fires_at_threshold(self) -> None:
        assert tool_burst_score(0) == 0.0
        assert tool_burst_score(4) == 0.0
        assert tool_burst_score(5) > 0.0
        assert tool_burst_score(10) == tool_burst_score(5)  # flat above threshold

    def test_correction_score_detects_pushback_phrases(self) -> None:
        assert correction_score(None) == 0.0
        assert correction_score("thanks, that worked") == 0.0
        assert correction_score("no, that's wrong") > 0.0
        assert correction_score("Actually, I wanted Celsius") > 0.0
        assert correction_score("Try again — doesn't work") > 0.0

    def test_last_user_text_flattens_content_blocks(self) -> None:
        msgs: list[dict] = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "plain string"},
            {"role": "assistant", "content": "reply"},
        ]
        assert last_user_text(msgs) == "plain string"

        block_msgs: list[dict] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
            },
        ]
        assert last_user_text(block_msgs) == "hello  world"
        assert last_user_text([]) is None


class TestReflectTool:
    async def test_reflect_stores_reflection_for_current_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Bootstrap a session via before_iteration
        ctx = AgentHookContext(
            iteration=0,
            messages=[{"role": "user", "content": "kick things off"}],
        )
        await hook.before_iteration(ctx)
        assert hook.current_session_id is not None
        session_id = hook.current_session_id

        tool = loop.tools.get("reflect")
        assert tool is not None
        out = await tool.execute(
            content="Next time, check the config file before editing code."
        )
        assert out.startswith("ok")

        rows = hook.db.execute(
            "SELECT session_id, content FROM reflections"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == session_id
        assert "config file" in rows[0][1]

    async def test_reflect_fails_without_active_session(
        self, loop: AgentLoop, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)
        # Do NOT call before_iteration. current_session_id stays None.
        assert hook.current_session_id is None

        tool = loop.tools.get("reflect")
        out = await tool.execute(content="this should not land anywhere")
        assert out.startswith("Error")
        assert "no active session" in out


class TestReflectionInjection:
    async def test_new_reflection_is_injected_on_next_iteration(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [{"role": "user", "content": "start"}]

        # Iteration 0 — bootstrap + write a reflection
        ctx0 = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx0)
        assert hook.current_session_id is not None

        reflect = loop.tools.get("reflect")
        await reflect.execute(content="Lesson: prefer simpler tools first.")

        # Simulate the LLM appending an assistant response, then end-of-iter
        messages.append({"role": "assistant", "content": "understood"})
        await hook.after_iteration(ctx0)

        # Iteration 1 — before_iteration should inject the new reflection
        ctx1 = AgentHookContext(iteration=1, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx1)
        assert len(messages) == before_len + 1
        injected = messages[-1]
        assert injected["role"] == "system"
        assert "Reflections from earlier" in injected["content"]
        assert "simpler tools" in injected["content"]

        # Iteration 2 — already-injected reflection should NOT be re-added
        messages.append({"role": "assistant", "content": "ok"})
        await hook.after_iteration(ctx1)
        ctx2 = AgentHookContext(iteration=2, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx2)
        assert len(messages) == before_len  # no new injection

    async def test_reflections_scoped_to_session(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reflections from session A must NOT leak into session B."""
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        # Session A
        msgs_a: list[dict] = [{"role": "user", "content": "session A start"}]
        await hook.before_iteration(AgentHookContext(iteration=0, messages=msgs_a))
        session_a = hook.current_session_id
        assert session_a is not None

        reflect = loop.tools.get("reflect")
        await reflect.execute(content="Session A — secret lesson")
        await hook.after_iteration(AgentHookContext(iteration=0, messages=msgs_a))

        # Session B — a totally new messages list
        msgs_b: list[dict] = [{"role": "user", "content": "session B start"}]
        ctx_b = AgentHookContext(iteration=0, messages=msgs_b)
        before_len = len(msgs_b)
        await hook.before_iteration(ctx_b)
        session_b = hook.current_session_id
        assert session_b is not None and session_b != session_a

        # No reflections should have been injected — session B is empty
        # of reflections and session A's are off-limits.
        assert len(msgs_b) == before_len
        for msg in msgs_b:
            assert "secret lesson" not in str(msg.get("content", ""))


class TestSalienceNudge:
    async def test_errors_trigger_nudge_next_iteration(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        # Lower the threshold so a single error fires the nudge cleanly.
        hook = nano_hermes.install(loop, config={"reflection": {"threshold": 3.0}})

        messages: list[dict] = [{"role": "user", "content": "do the thing"}]
        ctx0 = AgentHookContext(iteration=0, messages=messages, error="boom")
        await hook.before_iteration(ctx0)
        await hook.after_iteration(ctx0)
        assert hook._nudge_pending is True

        # Iteration 1 — nudge should be injected and cleared
        ctx1 = AgentHookContext(iteration=1, messages=messages)
        before_len = len(messages)
        await hook.before_iteration(ctx1)
        assert hook._nudge_pending is False
        assert len(messages) == before_len + 1
        nudge = messages[-1]
        assert nudge["role"] == "system"
        assert "reflect(content" in nudge["content"]

    async def test_quiet_iterations_do_not_trigger_nudge(
        self,
        loop: AgentLoop,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        _unset_embedding_keys(monkeypatch)
        hook = nano_hermes.install(loop)

        messages: list[dict] = [{"role": "user", "content": "easy question"}]
        ctx = AgentHookContext(iteration=0, messages=messages)
        await hook.before_iteration(ctx)
        await hook.after_iteration(ctx)
        assert hook._nudge_pending is False
