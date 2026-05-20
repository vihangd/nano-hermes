"""``skill_export`` agent-facing tool.

Exports training data (skill text + session trajectories) for offline
GEPA/MIPROv2 optimisation.  Output: JSONL files written to
``<workspace>/nano_hermes/exports/``.  The agent then copies these files to
an off-device machine for DSPy optimisation and re-imports improved skill
text via ``propose_skill(action="edit", ...)``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nanobot.agent.tools.base import Tool, tool_parameters

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "skill_name": {
            "type": "string",
            "description": (
                'Name of the skill to export, or "all" to export every active skill '
                "that has reached the corpus maturity threshold."
            ),
        },
        "min_sessions": {
            "type": "integer",
            "description": (
                "Override the minimum distinct session count required before a skill "
                "is exported.  Defaults to the configured export_min_sessions (50). "
                "Lower values produce noisier training sets."
            ),
        },
    },
    "required": ["skill_name"],
}


@tool_parameters(_SCHEMA)
class SkillExportTool(Tool):
    """Export training data for offline GEPA/MIPROv2 skill optimisation.

    Writes one JSONL file per eligible skill to
    ``<workspace>/nano_hermes/exports/``.  Each line is one training example:
    skill text, session context, and trajectory outcome (ok / partial / fail).

    A skill must have ≥ ``export_min_sessions`` (default 50) distinct sessions
    before it qualifies — low-data skills produce noisy optimisations.

    Workflow:
      1. Call ``skill_export(skill_name="all")`` to dump eligible skills.
      2. Copy the JSONL files to your laptop / cloud box.
      3. Run DSPy / GEPA optimisation off-device.
      4. Import the improved skill text with ``propose_skill(action="edit", ...)``.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "skill_export"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    async def execute(self, **kwargs: Any) -> str:
        from .training_export import (  # noqa: PLC0415
            count_skill_sessions,
            export_mature_skills,
            export_skill_training_data,
        )

        skill_name: str = kwargs["skill_name"]
        if skill_name != "all" and not _NAME_RE.match(skill_name):
            return f"error: invalid skill name {skill_name!r} — must match [a-z0-9][a-z0-9_-]{{0,63}}"
        cfg = self._hook.config.skill_stats
        min_sessions: int = int(kwargs.get("min_sessions") or cfg.export_min_sessions)

        workspace = Path(self._hook._loop.workspace)
        exports_dir = workspace / "nano_hermes" / "exports"

        if skill_name == "all":
            written = export_mature_skills(
                self._hook.db,
                workspace,
                exports_dir,
                min_sessions=min_sessions,
            )
            if not written:
                return (
                    f"ok: no active skills have ≥{min_sessions} distinct sessions yet. "
                    f"Run more sessions and try again, or lower min_sessions."
                )
            lines = [f"Exported {len(written)} skill(s) to {exports_dir}:\n"]
            for p in written:
                lines.append(f"  {p}")
            lines.append(
                "\nCopy these files off-device, optimise with DSPy/GEPA, then re-import:\n"
                '  propose_skill(action="edit", name="<skill>", content="<new SKILL.md>")'
            )
            return "\n".join(lines)

        # Single skill export
        n_sessions = count_skill_sessions(self._hook.db, skill_name)
        if n_sessions < min_sessions:
            return (
                f"ok: skill {skill_name!r} has only {n_sessions} distinct session(s) "
                f"(need ≥{min_sessions}).  Accumulate more sessions before exporting."
            )

        import time as _time  # noqa: PLC0415
        records = export_skill_training_data(
            self._hook.db, workspace, skill_name
        )
        if not records:
            return f"ok: skill {skill_name!r} has no trajectory rows — nothing to export"

        exports_dir.mkdir(parents=True, exist_ok=True)
        ts = int(_time.time())
        out_path = exports_dir / f"{skill_name}.{ts}.jsonl"

        import json as _json  # noqa: PLC0415
        with out_path.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

        return (
            f"Exported {len(records)} training row(s) for {skill_name!r} → {out_path}\n"
            f"(sessions: {n_sessions}, outcomes: "
            + str({r['outcome'] for r in records})
            + ")\n\n"
            "Copy off-device, optimise, then re-import:\n"
            f'  propose_skill(action="edit", name="{skill_name}", content="<new SKILL.md>")'
        )
