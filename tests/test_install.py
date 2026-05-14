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

from pathlib import Path

from nanobot.agent.loop import AgentLoop

import nano_hermes
from nano_hermes.hook import NanoHermesHook
from nano_hermes.memory.tool import MemoryPatchTool
from nano_hermes.reflect.tool import ReflectTool
from nano_hermes.session.search import SessionSearchTool
from nano_hermes.skills.tool import SkillSearchTool


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
        for name in [
            "memory_patch", "session_search", "trajectory_search",
            "skill_search", "skill_stats", "propose_skill", "skill_rate",
            "reflect", "nano_status",
        ]:
            assert name in loop.tools, f"tool '{name}' not registered"
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
            loop, config={"memory": {"memory_md_tokens": 50}}
        )
        assert hook.budgeted_memory.budgets.memory_md_tokens == 50
        # other budgets keep their defaults
        assert hook.budgeted_memory.budgets.user_md_tokens == 320


# ---------------------------------------------------------------------------
# Phase 6: tool registration completeness
# ---------------------------------------------------------------------------

class TestToolRegistrationCompleteness:
    """Verify all 9 tools are registered by install()."""

    def test_all_tools_registered(self, loop: AgentLoop) -> None:
        nano_hermes.install(loop)
        expected = [
            "memory_patch",
            "session_search",
            "trajectory_search",
            "skill_search",
            "skill_stats",
            "propose_skill",
            "skill_rate",
            "reflect",
            "nano_status",
        ]
        for name in expected:
            assert name in loop.tools, f"tool '{name}' not registered"
