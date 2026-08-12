"""nano-hermes: memory budgets, session search, skill evolution, and
Reflexion-based self-improvement extensions for HKUDS nanobot.

Usage::

    from nanobot.agent.loop import AgentLoop
    import nano_hermes

    loop = AgentLoop(bus=bus, provider=provider, workspace=ws)
    hook = nano_hermes.install(loop)

``install`` attaches a lifecycle hook and registers twelve agent-facing
tools: ``memory_patch``, ``session_browse``, ``session_search``,
``trajectory_search``, ``skill_search``, ``skill_stats``, ``propose_skill``,
``skill_rate``, ``reflect``, ``nano_status``, ``record_principle``, and
``workflow_suggest``.

It does NOT duplicate nanobot's existing memory/skill prompt injection —
nanobot's ContextBuilder already handles that via ``MemoryStore.read_memory()``
and ``SkillsLoader.build_skills_summary()``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import NanoHermesConfig

log = logging.getLogger(__name__)
from .hook import NanoHermesHook
from .memory.tool import MemoryPatchTool
from .reflect.tool import ReflectTool
from .session.browse import SessionBrowseTool
from .session.search import SessionSearchTool
from .session.trajectory_search import TrajectorySearchTool
from .session.workflow_suggest import WorkflowSuggestTool
from .skills.export_tool import SkillExportTool
from .skills.principle_tool import PrincipleTool
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
    "SessionBrowseTool",
    "SessionSearchTool",
    "TrajectorySearchTool",
    "SkillSearchTool",
    "SkillStatsTool",
    "ProposeSkillTool",
    "SkillRateTool",
    "SkillExportTool",
    "ReflectTool",
    "NanoStatusTool",
    "PrincipleTool",
    "WorkflowSuggestTool",
    "__version__",
]


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*; override wins on conflicts."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _load_config_files(workspace: Path) -> dict[str, Any]:
    """Load and merge user-level then workspace-level config files.

    Locations (later overrides earlier):
      1. ~/.nanobot/nano_hermes.json   — user-level defaults
      2. <workspace>/nano_hermes/config.json — workspace-specific overrides

    Missing files are silently skipped. Parse errors are logged at WARNING
    and that file is skipped (other files still apply).
    """
    user_cfg = Path.home() / ".nanobot" / "nano_hermes.json"
    workspace_cfg = workspace / "nano_hermes" / "config.json"

    merged: dict[str, Any] = {}
    for path in (user_cfg, workspace_cfg):
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
            if not isinstance(data, dict):
                log.warning("nano-hermes: %s must be a JSON object — skipping", path)
                continue
            merged = _deep_merge(merged, data)
            log.debug("nano-hermes: loaded config from %s", path)
        except Exception:
            log.warning("nano-hermes: failed to parse %s — skipping", path, exc_info=True)

    return merged


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
    elif config is not None:
        cfg = NanoHermesConfig.model_validate(config)
    else:
        cfg = NanoHermesConfig.model_validate(_load_config_files(loop.workspace))

    hook = NanoHermesHook(config=cfg, loop=loop)
    loop._extra_hooks.append(hook)
    if cfg.memory_loadtime_scan:
        from .memory.guard import install_loadtime_memory_scan  # noqa: PLC0415
        install_loadtime_memory_scan(loop.context.memory)
    loop.tools.register(MemoryPatchTool(hook=hook))
    loop.tools.register(SessionBrowseTool(hook=hook))
    loop.tools.register(SessionSearchTool(hook=hook))
    loop.tools.register(TrajectorySearchTool(hook=hook))
    loop.tools.register(SkillSearchTool(hook=hook))
    loop.tools.register(SkillStatsTool(hook=hook))
    loop.tools.register(PrincipleTool(hook=hook))
    loop.tools.register(ProposeSkillTool(hook=hook))
    loop.tools.register(SkillRateTool(hook=hook))
    loop.tools.register(ReflectTool(hook=hook))
    loop.tools.register(NanoStatusTool(hook=hook))
    loop.tools.register(WorkflowSuggestTool(hook=hook))
    loop.tools.register(SkillExportTool(hook=hook))
    from .governance.review_tool import PendingReviewTool  # noqa: PLC0415
    loop.tools.register(PendingReviewTool(hook=hook))
    return hook
