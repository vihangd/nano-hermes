"""The ``skill_search`` agent-facing Tool."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..coordinator.session import request_persists
from ..embedding.chain import AllProvidersFailed
from .composition import get_compositions

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": (
                "Free-text description of the task you're trying to accomplish. "
                "Returns skills ranked by how well their description matches."
            ),
        },
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 20,
            "description": "Number of skills to return. Defaults to 5.",
        },
    },
    "required": ["query"],
}


@tool_parameters(_SCHEMA)
class SkillSearchTool(Tool):
    """Semantic search over available skills, ranked by embedding similarity.

    Use when you're facing a task and want to know which skill (if any)
    is most relevant to load — without reading the full skill library
    line by line. Complements nanobot's alphabetical skill list in the
    system prompt.

    The index refreshes on every call: unchanged skills skip
    re-embedding via a content-hash check, so the common case costs
    only the query embed. Returns one line per hit:

        [distance] name — description (path)

    You can pass ``path`` to ``read_file`` to load the full skill body.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "skill_search"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    def _hidden_skills(self) -> set[str]:
        """Skills nanobot keeps out of this turn's prompt (nanobot > 0.4.1:
        ``cron`` without the cron tool, ``memory`` in a session it won't
        keep). Filtered here, not at index refresh, whose cleanup would drop
        their stats rows."""
        loop = self._hook._loop
        excluded = getattr(type(loop.context), "_excluded_skills", None)
        if excluded is None:
            return set()
        try:
            return set(excluded(include_memory=request_persists(), has_cron=loop.tools.has("cron")))
        except Exception:  # noqa: BLE001 — a changed signature must not break search
            log.debug("skill_search: nanobot exclusions unavailable", exc_info=True)
            return set()

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        if not query.strip():
            return "Error: query must not be empty."
        k = int(kwargs.get("k") or 5)
        hidden = self._hidden_skills()
        try:
            # Over-fetch so hidden skills don't eat the top k.
            hits = await self._hook.skill_indexer.search(query, k=k + len(hidden))
        except AllProvidersFailed as e:
            return (
                f"Error: cannot search skills — every embedding provider "
                f"failed ({e}). Fall back to nanobot's static skill list "
                f"in your system prompt."
            )
        shown = [h for h in hits if h.name not in hidden][:k]
        if not shown:
            return "no matching skills" if hits else "no indexed skills (none have a description?)"
        hits = shown
        self._hook.record_skill_candidates([h.name for h in hits])
        lines = []
        if hits[0].lexical:
            lines.append("(lexical fallback — embedding providers are down)")
        for h in hits:
            line = f"[{h.distance:.3f}] {h.name} — {h.description[:160]} ({h.location})"
            partners = get_compositions(self._hook.db, h.name)
            if partners:
                line += f" [often used with: {', '.join(partners)}]"
            if h.siblings:
                line += f" [also similar: {', '.join(h.siblings)}]"
            lines.append(line)
        return "\n".join(lines)
