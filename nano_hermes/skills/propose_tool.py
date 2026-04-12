"""The ``propose_skill`` agent-facing Tool — writes a new SKILL.md to disk.

Skills start in ``draft`` status and are promoted to ``active`` after N
successful uses (tracked by the skill_stats table). Drafts behave exactly
like active skills in search — only ``deprecated`` skills are filtered out.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

from .guard import scan_skill_content

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["create", "edit"],
            "description": (
                "'create' (default) writes a new draft skill — fails if an "
                "active or draft skill with that name already exists. "
                "'edit' rewrites the SKILL.md of an existing active or draft "
                "skill without touching usage counters — use read_file to "
                "inspect current content before editing."
            ),
        },
        "name": {
            "type": "string",
            "description": (
                "Skill identifier — lowercase letters, digits, hyphens, and "
                "underscores only (e.g. 'fetch-webpage'). Used as the "
                "directory name under workspace/skills/."
            ),
        },
        "description": {
            "type": "string",
            "description": (
                "One-line summary of what the skill does. This is embedded "
                "for semantic search, so precision matters."
            ),
        },
        "body": {
            "type": "string",
            "description": (
                "Full Markdown body of the skill — when to use it, how to "
                "invoke it, examples, common failure modes."
            ),
        },
    },
    "required": ["name", "description", "body"],
}


@tool_parameters(_SCHEMA)
class ProposeSkillTool(Tool):
    """Propose a new skill and save it as a draft SKILL.md on disk.

    Use this when you've figured out a reusable procedure that would be
    valuable in future sessions — e.g. a reliable way to call an API,
    a multi-step data-processing recipe, or a debugging checklist.

    The skill starts as a *draft*. Use ``skill_rate`` after applying the skill
    to record whether it helped. After enough successful ratings (default 3),
    it promotes to *active*. Skills that chronically fail get *deprecated* and
    stop appearing in search results.

    Rules:
    - Name must be lowercase, use hyphens/underscores, no spaces or slashes.
    - Cannot overwrite an existing active or draft skill (use a new name,
      or wait for deprecation then re-propose).
    - Deprecated skills CAN be re-proposed with fresh content.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "propose_skill"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs.get("action") or "create"
        skill_name: str = kwargs.get("name", "").strip()
        description: str = kwargs.get("description", "").strip()
        body: str = kwargs.get("body", "").strip()

        # --- Common validation ---
        if not _NAME_RE.match(skill_name):
            return (
                "Error: invalid skill name — must be lowercase letters, "
                "digits, hyphens, or underscores, starting with a letter or "
                f"digit (got '{skill_name}')."
            )
        if not description:
            return "Error: description must not be empty."
        if not body:
            return "Error: body must not be empty."

        if action not in ("create", "edit"):
            return f"Error: unknown action {action!r} — must be 'create' or 'edit'."
        if action == "edit":
            return await self._edit(skill_name, description, body)
        return await self._create(skill_name, description, body)

    async def _create(self, skill_name: str, description: str, body: str) -> str:
        # --- Conflict check ---
        row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        if row:
            existing_status = row[0]
            if existing_status in ("active", "draft"):
                return (
                    f"Error: skill '{skill_name}' already exists with status "
                    f"'{existing_status}'. Choose a different name, or wait "
                    f"for the skill to be deprecated before re-proposing."
                )
            # deprecated -> fall through, allow overwrite

        return await self._write_skill_md(
            skill_name, description, body,
            upsert_sql=(
                "INSERT INTO skill_stats "
                "(name, status, use_count, success_count, last_used_at, provenance, content_hash) "
                "VALUES (?, 'draft', 0, 0, NULL, NULL, NULL) "
                "ON CONFLICT(name) DO UPDATE SET "
                "status = 'draft', "
                "use_count = 0, "
                "success_count = 0, "
                "last_used_at = NULL, "
                "provenance = NULL, "
                "content_hash = NULL, "
                "indexed_at = NULL"
            ),
            success_msg=(
                f"ok: created draft skill '{skill_name}' at "
                f"workspace/skills/{skill_name}/SKILL.md. "
                "It will appear in skill_search after the next search call triggers re-indexing."
            ),
        )

    async def _edit(self, skill_name: str, description: str, body: str) -> str:
        # --- Must exist and not be deprecated ---
        row = self._hook.db.execute(
            "SELECT status FROM skill_stats WHERE name = ?", (skill_name,)
        ).fetchone()
        if not row:
            return (
                f"Error: skill '{skill_name}' not found. "
                "Use action='create' to propose a new skill."
            )
        existing_status = row[0]
        if existing_status == "deprecated":
            return (
                f"Error: skill '{skill_name}' is deprecated. "
                "Use action='create' to re-propose it with fresh counters."
            )

        return await self._write_skill_md(
            skill_name, description, body,
            upsert_sql=(
                "UPDATE skill_stats SET content_hash = NULL, indexed_at = NULL WHERE name = ?"
            ),
            success_msg=(
                f"ok: updated skill '{skill_name}' at "
                f"workspace/skills/{skill_name}/SKILL.md. "
                "Usage stats preserved. Re-indexing will happen on next skill_search call."
            ),
            edit_mode=True,
        )

    async def _write_skill_md(
        self,
        skill_name: str,
        description: str,
        body: str,
        *,
        upsert_sql: str,
        success_msg: str,
        edit_mode: bool = False,
    ) -> str:
        # Security scan the body before writing to disk.
        err = scan_skill_content(body)
        if err:
            return f"Error: {err}"

        skill_dir = self._hook.workspace / "skills" / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_md = skill_dir / "SKILL.md"
        content = f"---\nname: {skill_name}\ndescription: {description}\n---\n\n{body}\n"
        skill_md.write_text(content, encoding="utf-8")

        try:
            with self._hook.db:
                # edit mode uses a plain UPDATE (single param); create uses UPSERT (tuple param)
                if edit_mode:
                    self._hook.db.execute(upsert_sql, (skill_name,))
                else:
                    self._hook.db.execute(upsert_sql, (skill_name,))
        except Exception as e:
            return f"Error: wrote SKILL.md but failed to update skill_stats: {e}"

        return success_msg
