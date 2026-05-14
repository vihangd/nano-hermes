"""nano-hermes: memory budgets, session search, skill evolution, and
Reflexion-based self-improvement extensions for HKUDS nanobot.

Usage::

    from nanobot.agent.loop import AgentLoop
    import nano_hermes

    loop = AgentLoop(bus=bus, provider=provider, workspace=ws)
    hook = nano_hermes.install(loop)

``install`` attaches a lifecycle hook and registers ten agent-facing
tools: ``memory_patch``, ``session_search``, ``trajectory_search``,
``skill_search``, ``skill_stats``, ``propose_skill``, ``skill_rate``,
``reflect``, ``nano_status``, and ``workflow_suggest``.

It does NOT duplicate nanobot's existing memory/skill prompt injection —
nanobot's ContextBuilder already handles that via
``MemoryStore.get_memory_context()`` and
``SkillsLoader.build_skills_summary()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import NanoHermesConfig
from .hook import NanoHermesHook
from .memory.tool import MemoryPatchTool
from .reflect.tool import ReflectTool
from .session.search import SessionSearchTool
from .session.trajectory_search import TrajectorySearchTool
from .session.workflow_suggest import WorkflowSuggestTool
from .skills.propose_tool import ProposeSkillTool
from .skills.rate_tool import SkillRateTool
from .skills.stats_tool import SkillStatsTool
from .skills.tool import SkillSearchTool
from .status.tool import NanoStatusTool

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

__version__ = "0.8.0"
__all__ = [
    "install",
    "NanoHermesHook",
    "NanoHermesConfig",
    "MemoryPatchTool",
    "SessionSearchTool",
    "TrajectorySearchTool",
    "SkillSearchTool",
    "SkillStatsTool",
    "ProposeSkillTool",
    "SkillRateTool",
    "ReflectTool",
    "NanoStatusTool",
    "WorkflowSuggestTool",
    "__version__",
]


def install(
    loop: "AgentLoop",
    config: dict[str, Any] | NanoHermesConfig | None = None,
) -> NanoHermesHook:
    """Install nano-hermes on a nanobot ``AgentLoop``.

    Appends a ``NanoHermesHook`` to the loop's ``_extra_hooks`` list (same
    mechanism ``Nanobot.run(hooks=...)`` uses under the hood) and registers
    nine tools on the loop's ToolRegistry: ``memory_patch``,
    ``session_search``, ``trajectory_search``, ``skill_search``,
    ``skill_stats``, ``propose_skill``, ``skill_rate``, ``reflect``,
    ``nano_status``, and ``workflow_suggest``.

    Returns the installed hook so callers can inspect or later detach it.
    """
    if isinstance(config, NanoHermesConfig):
        cfg = config
    else:
        cfg = NanoHermesConfig.model_validate(config or {})

    hook = NanoHermesHook(config=cfg, loop=loop)
    loop._extra_hooks.append(hook)
    loop.tools.register(MemoryPatchTool(hook=hook))
    loop.tools.register(SessionSearchTool(hook=hook))
    loop.tools.register(TrajectorySearchTool(hook=hook))
    loop.tools.register(SkillSearchTool(hook=hook))
    loop.tools.register(SkillStatsTool(hook=hook))
    loop.tools.register(ProposeSkillTool(hook=hook))
    loop.tools.register(SkillRateTool(hook=hook))
    loop.tools.register(ReflectTool(hook=hook))
    loop.tools.register(NanoStatusTool(hook=hook))
    loop.tools.register(WorkflowSuggestTool(hook=hook))
    return hook
