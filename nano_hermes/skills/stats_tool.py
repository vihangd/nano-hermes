"""The ``skill_stats`` agent-facing Tool.

Read-only view of the skill usage statistics accumulated by nano-hermes.
Zero LLM cost — pure SQL query against skill_stats.
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": (
                "Skill name to query. Omit to list all skills with recorded usage."
            ),
        },
    },
    "required": [],
}


@tool_parameters(_SCHEMA)
class SkillStatsTool(Tool):
    """Query skill usage statistics tracked by nano-hermes.

    Returns use counts, success rates, and last-used timestamps for
    skills that have been retrieved via skill_search and then used in
    the same iteration. Useful for deciding which skills have proven
    reliable vs. which ones tend to fail.

    If a skill has fewer than ``min_uses_for_success_rate`` uses, the
    success rate is shown as 'n/a (too few uses)'.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "skill_stats"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        skill_name: str | None = kwargs.get("name")
        min_uses = self._hook.config.skill_stats.min_uses_for_success_rate

        if skill_name:
            rows = self._hook.db.execute(
                "SELECT name, use_count, success_count, status, last_used_at "
                "FROM skill_stats WHERE name = ?",
                (skill_name,),
            ).fetchall()
        else:
            rows = self._hook.db.execute(
                "SELECT name, use_count, success_count, status, last_used_at "
                "FROM skill_stats WHERE use_count > 0 "
                "ORDER BY use_count DESC",
            ).fetchall()

        if not rows:
            if skill_name:
                return f"No stats recorded for skill '{skill_name}'."
            return "No skill usage recorded yet. Use skill_search first."

        lines = []
        now = time.time()
        for name, use_count, success_count, status, last_used_at in rows:
            if use_count >= min_uses:
                pct = int(100 * success_count / use_count)
                rate = f"{success_count}/{use_count} ({pct}%)"
            else:
                rate = f"{success_count}/{use_count} (n/a — too few uses)"

            if last_used_at:
                age_s = now - last_used_at
                if age_s < 3600:
                    age = f"{int(age_s / 60)}m ago"
                elif age_s < 86400:
                    age = f"{int(age_s / 3600)}h ago"
                else:
                    age = f"{int(age_s / 86400)}d ago"
            else:
                age = "never"

            lines.append(
                f"{name} | uses: {use_count} | success: {rate} | "
                f"status: {status} | last: {age}"
            )

        return "\n".join(lines)
