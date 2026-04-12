"""nano-hermes: memory budgets, session search, and self-evolution
extensions for HKUDS nanobot.

Usage::

    from nanobot.agent.loop import AgentLoop
    import nano_hermes

    loop = AgentLoop(bus=bus, provider=provider, workspace=ws)
    hook = nano_hermes.install(loop)

``install`` attaches a lifecycle hook and registers the ``memory_patch``
and ``session_search`` tools. It does NOT duplicate nanobot's existing
memory/skill prompt injection — nanobot's ContextBuilder already handles
that via ``MemoryStore.get_memory_context()`` and
``SkillsLoader.build_skills_summary()``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .config import NanoHermesConfig
from .hook import NanoHermesHook
from .memory.tool import MemoryPatchTool
from .reflect.tool import ReflectTool
from .session.search import SessionSearchTool
from .skills.stats_tool import SkillStatsTool
from .skills.tool import SkillSearchTool

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop

__version__ = "0.1.0"
__all__ = [
    "install",
    "NanoHermesHook",
    "NanoHermesConfig",
    "MemoryPatchTool",
    "SessionSearchTool",
    "SkillSearchTool",
    "SkillStatsTool",
    "ReflectTool",
    "__version__",
]


def install(
    loop: "AgentLoop",
    config: dict[str, Any] | NanoHermesConfig | None = None,
) -> NanoHermesHook:
    """Install nano-hermes on a nanobot ``AgentLoop``.

    Appends a ``NanoHermesHook`` to the loop's ``_extra_hooks`` list (same
    mechanism ``Nanobot.run(hooks=...)`` uses under the hood) and registers
    ``memory_patch`` and ``session_search`` on the loop's ToolRegistry.

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
    loop.tools.register(SkillSearchTool(hook=hook))
    loop.tools.register(SkillStatsTool(hook=hook))
    loop.tools.register(ReflectTool(hook=hook))
    return hook
