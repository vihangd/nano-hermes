"""The ``pending_review`` agent-facing Tool.

Review and resolve autonomous evolution writes staged by the write-approval
gate (``write_approval == "approve"``). Zero LLM cost for list/diff/reject;
approve replays the staged write (skills: write SKILL.md; principles: re-run
the curator ops).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from . import write_approval as wa

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "diff", "approve", "reject"],
            "description": (
                "list: show open pending writes. diff: show current-vs-proposed "
                "for one id. approve: apply it (refused if the skill changed "
                "since staging). reject: discard it."
            ),
        },
        "id": {
            "type": "integer",
            "description": "Pending write id (required for diff/approve/reject).",
        },
    },
    "required": ["action"],
}


@tool_parameters(_SCHEMA)
class PendingReviewTool(Tool):
    """Review autonomous skill/principle writes held by the write-approval gate.

    When ``write_approval`` is set to ``approve``, the background evolution loop
    stages its skill rewrites and curator edits instead of committing them. Use
    this tool to list what is waiting, inspect a proposed change, then approve
    (apply it) or reject (discard it). Approving a skill write is refused if the
    skill changed on disk since the write was staged.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "pending_review"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs.get("action", "")
        db = self._hook.db
        workspace = self._hook.workspace

        if action == "list":
            rows = wa.list_pending(db)
            if not rows:
                return "No pending writes."
            return "\n".join(
                f"#{r['id']} | {r['subsystem']} | {r['skill_name'] or '-'} | "
                f"{r['origin']} | {r['reason']}"
                for r in rows
            )

        pid = kwargs.get("id")
        if pid is None:
            return f"Error: 'id' is required for action='{action}'."
        pid = int(pid)

        if action == "diff":
            return wa.diff_pending(db, workspace, pid)
        if action == "reject":
            return wa.reject(db, pid)
        if action == "approve":
            rec = wa.get_pending(db, pid)
            if not rec or rec["status"] != "pending":
                return f"no open pending write #{pid}"
            if rec["subsystem"] == "principles":
                return await wa.approve_principles(self._hook, pid)
            return wa.approve_skill(
                db, workspace, pid,
                snapshot_retain=self._hook.config.skill_stats.snapshot_retain,
            )
        return f"Error: unknown action {action!r}."
