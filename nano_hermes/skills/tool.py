"""The ``skill_search`` agent-facing Tool."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from ..embedding.chain import AllProvidersFailed

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


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

    async def execute(self, **kwargs: Any) -> str:
        query: str = kwargs["query"]
        if not query.strip():
            return "Error: query must not be empty."
        k = int(kwargs.get("k") or 5)
        try:
            hits = await self._hook.skill_indexer.search(query, k=k)
        except AllProvidersFailed as e:
            return (
                f"Error: cannot search skills — every embedding provider "
                f"failed ({e}). Fall back to nanobot's static skill list "
                f"in your system prompt."
            )
        if not hits:
            return "no indexed skills (none have a description?)"
        self._hook.record_skill_candidates([h.name for h in hits])
        return "\n".join(
            f"[{h.distance:.3f}] {h.name} — {h.description[:160]} ({h.location})"
            for h in hits
        )
