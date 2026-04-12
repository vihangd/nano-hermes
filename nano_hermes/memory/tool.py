"""The ``memory_patch`` agent-facing Tool.

Wraps ``BudgetedMemory`` as a proper ``nanobot.agent.tools.base.Tool``
subclass, registered via ``ToolRegistry.register(instance)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "slot": {
            "type": "string",
            "enum": ["memory", "user", "soul"],
            "description": "Which memory file to patch.",
        },
        "action": {
            "type": "string",
            "enum": ["add", "replace", "remove"],
            "description": "What to do with the slot.",
        },
        "content": {
            "type": "string",
            "description": "New entry to append (required for action=add).",
        },
        "needle": {
            "type": "string",
            "description": "Substring locator (required for action=replace|remove).",
        },
        "replacement": {
            "type": "string",
            "description": "Replacement text (required for action=replace).",
        },
    },
    "required": ["slot", "action"],
}


@tool_parameters(_SCHEMA)
class MemoryPatchTool(Tool):
    """Patch long-term memory (MEMORY.md / USER.md / SOUL.md).

    Frozen-snapshot semantics: writes land on disk immediately but only
    appear in the system prompt on the next session start — nanobot's
    ContextBuilder loads memory once per run (prefix-cache friendly).
    Per-slot character budgets are enforced; overflow errors report the
    exact shortfall so your next call can free space by removing or
    replacing lower-value entries.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "memory_patch"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        slot: str = kwargs["slot"]
        action: str = kwargs["action"]
        content = kwargs.get("content")
        needle = kwargs.get("needle")
        replacement = kwargs.get("replacement")

        mem = self._hook.budgeted_memory
        try:
            if action == "add":
                if content is None:
                    return "Error: action=add requires `content`"
                result = mem.add(slot, content)  # type: ignore[arg-type]
                if result == "duplicate":
                    return f"ok: entry already exists in {slot} (not re-added)"
                return f"ok: added {len(content.strip())} chars to {slot}"
            if action == "replace":
                if needle is None or replacement is None:
                    return "Error: action=replace requires `needle` and `replacement`"
                mem.replace(slot, needle, replacement)  # type: ignore[arg-type]
                return f"ok: replaced in {slot}"
            if action == "remove":
                if needle is None:
                    return "Error: action=remove requires `needle`"
                mem.remove(slot, needle)  # type: ignore[arg-type]
                return f"ok: removed from {slot}"
            return f"Error: unknown action {action!r}"
        except Exception as e:
            return f"Error: {e}"
