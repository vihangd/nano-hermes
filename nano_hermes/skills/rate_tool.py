"""The ``skill_rate`` agent-facing Tool — explicitly records whether a skill
helped accomplish a task.

This is the clean-signal path for skill lifecycle management. The hook's
observed-use detection (read_file on SKILL.md) tracks *that* a skill was
consulted, but cannot determine *whether it helped*. The agent knows. Call
this after following a skill's instructions to record your judgment.

Only ``skill_rate`` writes ``success_count`` and triggers promotion/deprecation
checks — the automatic path intentionally does not, so lifecycle decisions
are always grounded in the agent's explicit assessment.
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
                "Name of the skill to rate — exactly as shown in skill_search results."
            ),
        },
        "outcome": {
            "type": "string",
            "enum": ["success", "failure"],
            "description": (
                "Whether the skill helped accomplish the task. "
                "'success': the skill's instructions worked and the task advanced. "
                "'failure': the skill was unhelpful, wrong, or caused errors."
            ),
        },
    },
    "required": ["name", "outcome"],
}


@tool_parameters(_SCHEMA)
class SkillRateTool(Tool):
    """Record whether a skill helped accomplish the current task.

    Call this AFTER following a skill's instructions and observing the result.
    You don't need to call it every time — only when you have a clear signal.
    Repeated ratings accumulate: skills with enough successes are promoted from
    draft to active; skills with chronically low success rates are deprecated
    and excluded from future search results.

    Outcome values:
    - "success": the skill worked — the task advanced as expected.
    - "failure": the skill didn't help — wrong approach, bad instructions, or errors.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "skill_rate"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        skill_name: str = kwargs.get("name", "")
        outcome: str = kwargs.get("outcome", "")

        skill_name = skill_name.strip()
        if not skill_name:
            return "Error: skill name must not be empty."

        if outcome not in ("success", "failure"):
            return (
                f"Error: outcome must be 'success' or 'failure', got {outcome!r}."
            )

        # Verify the skill exists before writing
        row = self._hook.db.execute(
            "SELECT status, use_count, success_count FROM skill_stats WHERE name = ?",
            (skill_name,),
        ).fetchone()
        if row is None:
            return (
                f"Error: skill {skill_name!r} not found in skill_stats. "
                "Use skill_search to find known skills, or propose_skill to create one."
            )

        is_success = outcome == "success"
        now = time.time()
        session_id = self._hook.current_session_id

        try:
            with self._hook.db:
                self._hook.db.execute(
                    "UPDATE skill_stats SET "
                    "use_count = use_count + 1, "
                    "success_count = success_count + CASE WHEN ? THEN 1 ELSE 0 END, "
                    "last_used_at = ?, "
                    "provenance = json_insert(COALESCE(provenance, '[]'), '$[#]', ?) "
                    "WHERE name = ?",
                    (is_success, now, session_id, skill_name),
                )
        except Exception as e:
            return f"Error: {e}"

        # Lifecycle check: this is the only place promotion/deprecation fires.
        self._hook._check_promotions([skill_name])

        # Register for trajectory tracking.
        self._hook.record_skill_rating(skill_name)

        # Read back to see if status changed so we can report it.
        new_row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        new_status = new_row[0] if new_row else row[0]
        old_status = row[0]

        msg = f"ok: {skill_name!r} rated as {outcome}"
        if new_status != old_status:
            msg += f" (status: {old_status} → {new_status})"
        return msg
