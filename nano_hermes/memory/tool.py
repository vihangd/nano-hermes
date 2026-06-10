"""The ``memory_patch`` agent-facing Tool.

Wraps ``BudgetedMemory`` as a proper ``nanobot.agent.tools.base.Tool``
subclass, registered via ``ToolRegistry.register(instance)``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

log = logging.getLogger(__name__)

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
            "enum": ["add", "replace", "remove", "consolidate", "distill", "audit"],
            "description": (
                "What to do with the slot. "
                "consolidate: embed entries, merge near-duplicates (cosine ≥ threshold), "
                "keep the longest entry per cluster. Call when memory feels bloated. "
                "distill: find recurring themes across successful sessions and surface "
                "them as candidate facts for you to add to memory. Slot is ignored for distill. "
                "audit: hygiene sweep over stored facts — retire ones an updated fact "
                "has made stale. Slot is ignored for audit."
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
                # Reset the cadence counter so the nudge doesn't fire
                # immediately after the agent already saved.
                self._hook._reflection_coord.note_memory_save()
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
            if action == "audit":
                return await self._audit()
            return f"Error: unknown action {action!r}"
        except Exception as e:
            return f"Error: {e}"

    async def _audit(self) -> str:
        """Hygiene sweep: retire stored facts an updated fact has superseded."""
        from .bitemporal import sweep_contradictions  # noqa: PLC0415

        mem_cfg = self._hook.config.memory
        n = await sweep_contradictions(
            self._hook,
            enabled=mem_cfg.bitemporal_invalidation_enabled,
            sim_threshold=mem_cfg.bitemporal_supersede_threshold,
            max_anchors=mem_cfg.contradiction_sweep_max_anchors,
        )
        if not mem_cfg.bitemporal_invalidation_enabled:
            return "ok: audit skipped (bitemporal invalidation disabled)"
        return f"ok: audit retired {n} stale fact(s)"

    async def _distill(self) -> str:
        import json as _json  # noqa: PLC0415
        import time as _time  # noqa: PLC0415
        from .consolidation import distill_hub_to_fact, find_hub_clusters  # noqa: PLC0415

        cfg = self._hook.config.memory
        try:
            hubs = await find_hub_clusters(
                self._hook.db,
                min_sessions=cfg.distill_hub_min_sessions,
                max_chunks=cfg.distill_max_chunks,
                cluster_threshold=cfg.distill_cluster_threshold,
            )
        except Exception as e:
            return f"Error: clustering failed — {e}"

        if not hubs:
            return (
                "ok: no recurring cross-session hubs found "
                "(need ≥2 successful sessions with thematically overlapping content)"
            )

        if not cfg.distill_llm_enabled:
            lines = [f"Found {len(hubs)} hub cluster(s) from episodic memory:\n"]
            for i, hub in enumerate(hubs, 1):
                n_sess = len(hub["sessions"])
                ids_preview = hub["chunk_ids"][:5]
                lines.append(
                    f"Hub {i} — spans {n_sess} session(s), "
                    f"chunk_ids (first 5): {ids_preview}:"
                )
                for j, sample in enumerate(hub["samples"], 1):
                    snippet = sample.replace("\n", " ")[:200]
                    lines.append(f"  [{j}] {snippet}")
                lines.append("")
            lines.append(
                "Review these hubs and distill durable facts with:\n"
                '  memory_patch(action="add", slot="memory", content="<fact>")'
            )
            return "\n".join(lines)

        from .links import link_new_fact  # noqa: PLC0415

        distilled: list[dict] = []
        for hub in hubs:
            fact_data = await distill_hub_to_fact(self._hook, hub)
            if fact_data is None:
                continue
            chunk_ids = hub["chunk_ids"]
            cur = self._hook.db.execute(
                "INSERT INTO semantic_facts "
                "(content, source_chunk_ids, created_at, keywords, tags, context, importance) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    fact_data["fact"],
                    _json.dumps(chunk_ids),
                    _time.time(),
                    _json.dumps(fact_data["keywords"]),
                    _json.dumps(fact_data["tags"]),
                    fact_data["context"],
                    fact_data["importance"],
                ),
            )
            fact_id = cur.lastrowid
            self._hook.db.commit()

            # Embed the fact + auto-link to similar prior facts (A-MEM).
            try:
                n_links = await link_new_fact(self._hook, fact_id, fact_data["fact"])
            except Exception as e:
                log.warning("link_new_fact failed for fact %s: %s", fact_id, e)
                n_links = 0

            # Bi-temporal: stamp invalid_at on any prior fact this one
            # supersedes (runs after linking so the vector is already stored).
            superseded: list[int] = []
            try:
                from .bitemporal import invalidate_superseded_facts  # noqa: PLC0415
                superseded = await invalidate_superseded_facts(
                    self._hook,
                    fact_id,
                    fact_data["fact"],
                    enabled=cfg.bitemporal_invalidation_enabled,
                    sim_threshold=cfg.bitemporal_supersede_threshold,
                )
            except Exception as e:
                log.warning("supersession check failed for fact %s: %s", fact_id, e)

            # Feed importance into the salience nudge so foundational
            # facts trigger a reflection prompt sooner.
            self._hook._reflection_coord.add_salience(  # noqa: SLF001
                fact_data["importance"] / 10.0
            )

            entry = dict(fact_data)
            entry["fact_id"] = fact_id
            entry["chunk_ids"] = chunk_ids
            entry["n_links"] = n_links
            entry["superseded"] = superseded
            distilled.append(entry)

        if not distilled:
            return (
                "ok: distillation found hubs but produced no usable facts "
                "(LLM call failed or returned empty for all hubs)"
            )

        lines = [f"Distilled {len(distilled)} semantic fact(s) from {len(hubs)} hub(s):\n"]
        for i, entry in enumerate(distilled, 1):
            ids_preview = entry["chunk_ids"][:5]
            tags = ",".join(entry["tags"]) or "—"
            lines.append(
                f"Fact {i} [id={entry['fact_id']} importance={entry['importance']} "
                f"tags={tags} links={entry['n_links']} chunk_ids={ids_preview}]:"
            )
            lines.append(f"  {entry['fact']}")
            if entry.get("superseded"):
                lines.append(
                    f"  ⚠ supersedes outdated fact(s) {entry['superseded']} — "
                    "update or remove the matching MEMORY.md entry if you promoted it."
                )
            lines.append("")
        lines.append(
            "Facts saved to semantic_facts table. Promote to long-term memory with:\n"
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
