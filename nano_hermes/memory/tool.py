"""The ``memory_patch`` agent-facing Tool.

Wraps ``BudgetedMemory`` as a proper ``nanobot.agent.tools.base.Tool``
subclass, registered via ``ToolRegistry.register(instance)``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..redact import format_redaction_note, redact

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
            "enum": ["add", "replace", "remove", "consolidate", "distill"],
            "description": (
                "What to do with the slot. "
                "consolidate: embed entries, merge near-duplicates (cosine ≥ threshold), "
                "keep the longest entry per cluster. Call when memory feels bloated. "
                "distill: find recurring themes across successful sessions and surface "
                "them as candidate facts for you to add to memory. Slot is ignored for distill."
            ),
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
    "required": ["action"],
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
        slot: str = kwargs.get("slot", "memory")
        action: str = kwargs["action"]
        content = kwargs.get("content")
        needle = kwargs.get("needle")
        replacement = kwargs.get("replacement")

        # Redact secrets from the content the agent is about to persist.
        # `needle` is a locator (must match what's already on disk) so we
        # never redact it.
        redaction_note = ""
        if self._hook.config.redact_secrets:
            if action == "add" and content is not None:
                r = redact(content)
                content = r.text
                redaction_note = format_redaction_note(r)
            elif action == "replace" and replacement is not None:
                r = redact(replacement)
                replacement = r.text
                redaction_note = format_redaction_note(r)

        mem = self._hook.budgeted_memory
        try:
            if action == "add":
                if content is None:
                    return "Error: action=add requires `content`"
                result = mem.add(slot, content)  # type: ignore[arg-type]
                if result == "duplicate":
                    return f"ok: entry already exists in {slot} (not re-added)"
                return f"ok: added {len(content.strip())} chars to {slot}{redaction_note}"
            if action == "replace":
                if needle is None or replacement is None:
                    return "Error: action=replace requires `needle` and `replacement`"
                mem.replace(slot, needle, replacement)  # type: ignore[arg-type]
                return f"ok: replaced in {slot}{redaction_note}"
            if action == "remove":
                if needle is None:
                    return "Error: action=remove requires `needle`"
                mem.remove(slot, needle)  # type: ignore[arg-type]
                return f"ok: removed from {slot}"
            if action == "consolidate":
                return await self._consolidate(slot)  # type: ignore[arg-type]
            if action == "distill":
                return await self._distill()
            return f"Error: unknown action {action!r}"
        except Exception as e:
            return f"Error: {e}"

    async def _distill(self) -> str:
        from .consolidation import find_hub_clusters  # noqa: PLC0415

        cfg = self._hook.config.memory
        try:
            hubs = await find_hub_clusters(
                self._hook.db,
                min_sessions=cfg.distill_hub_min_sessions,
                max_chunks=cfg.distill_max_chunks,
                cluster_threshold=cfg.distill_cluster_threshold,
            )
        except Exception as e:
            return f"Error: embedding failed during distillation — {e}"

        if not hubs:
            return (
                "ok: no recurring cross-session hubs found "
                "(need ≥2 successful sessions with thematically overlapping content)"
            )

        lines = [f"Found {len(hubs)} hub cluster(s) from episodic memory:\n"]
        for i, hub in enumerate(hubs, 1):
            n_sess = len(hub["sessions"])
            lines.append(f"Hub {i} — spans {n_sess} session(s):")
            for j, sample in enumerate(hub["samples"], 1):
                snippet = sample.replace("\n", " ")[:200]
                lines.append(f"  [{j}] {snippet}")
            lines.append("")

        lines.append(
            "Review these hubs and distill durable facts with:\n"
            '  memory_patch(action="add", slot="memory", content="<fact>")'
        )
        return "\n".join(lines)

    async def _consolidate(self, slot: str) -> str:
        from .consolidation import consolidate_entries, split_entries  # noqa: PLC0415

        mem = self._hook.budgeted_memory
        text = mem.read(slot)  # type: ignore[arg-type]
        if not text.strip():
            return f"ok: {slot} is empty — nothing to consolidate"

        entries = split_entries(text)
        if len(entries) < 2:
            return f"ok: {slot} has only {len(entries)} entry — nothing to consolidate"

        threshold = self._hook.config.memory.consolidation_similarity_threshold
        try:
            surviving, n_removed = await consolidate_entries(
                entries, self._hook.embedder, threshold
            )
        except Exception as e:
            return f"Error: embedding failed during consolidation — {e}"

        if n_removed == 0:
            return f"ok: {slot} — no near-duplicates found at threshold {threshold:.2f}"

        new_text = "\n\n".join(surviving)
        mem.write(slot, new_text)  # type: ignore[arg-type]
        chars_freed = len(text) - len(new_text)
        return (
            f"ok: {slot} consolidated — {n_removed} entr{'y' if n_removed == 1 else 'ies'} merged, "
            f"{chars_freed:+d} chars ({len(entries)} → {len(surviving)})"
        )
