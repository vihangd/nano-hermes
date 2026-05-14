"""The ``workflow_suggest`` agent-facing Tool.

Surfaces recurring task clusters from successful session trajectories so
the agent can distill them into reusable workflow skills.  Agent-invoked
(not automatic) — the agent calls this when it suspects recurring patterns
are worth capturing, then calls propose_skill to create the workflow skill.

Off by default: requires ``workflow_induction.enabled = true`` in config.
"""
from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING, Any

import numpy as np
from nanobot.agent.tools.base import Tool, tool_parameters

from ..memory.consolidation import greedy_cluster

if TYPE_CHECKING:
    from ..hook import NanoHermesHook


_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "k": {
            "type": "integer",
            "minimum": 1,
            "maximum": 10,
            "description": "Maximum number of workflow clusters to return. Defaults to 5.",
        },
    },
    "required": [],
}


@tool_parameters(_SCHEMA)
class WorkflowSuggestTool(Tool):
    """Identify recurring task patterns across successful sessions.

    Clusters past successful trajectories by semantic similarity and
    returns groups of related tasks. Use the output to recognise a
    reusable procedure and create a workflow skill with propose_skill.

    Requires workflow_induction.enabled = true in nano-hermes config.
    The more successful sessions you have, the better the signal.
    """

    def __init__(self, *, hook: "NanoHermesHook") -> None:
        self._hook = hook

    @property
    def name(self) -> str:
        return "workflow_suggest"

    @property
    def description(self) -> str:
        return (type(self).__doc__ or "").strip()

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        cfg = self._hook.config.workflow_induction
        if not cfg.enabled:
            return (
                "workflow_suggest is disabled. "
                "Set workflow_induction.enabled=true in nano-hermes config to activate."
            )

        k = int(kwargs.get("k") or 5)
        clusters = _find_workflow_clusters(
            self._hook.db,
            max_trajectories=cfg.max_trajectories,
            cluster_threshold=cfg.cluster_threshold,
            min_cluster_size=cfg.min_cluster_size,
        )

        if not clusters:
            return (
                "No recurring workflow patterns found. "
                f"Need ≥{cfg.min_cluster_size} similar successful sessions. "
                "Run more sessions or lower workflow_induction.min_cluster_size."
            )

        clusters = clusters[:k]
        lines = [f"Found {len(clusters)} recurring pattern(s) across successful sessions:\n"]
        for i, cl in enumerate(clusters, 1):
            n = len(cl["tasks"])
            skill_counts: dict[str, int] = {}
            for skills in cl["skills_used"]:
                for s in skills:
                    skill_counts[s] = skill_counts.get(s, 0) + 1
            common_skills = sorted(skill_counts, key=lambda s: -skill_counts[s])[:5]

            lines.append(f"Pattern {i} — {n} session(s):")
            for t in cl["tasks"][:3]:
                lines.append(f"  - {t[:120]}")
            if n > 3:
                lines.append(f"  ... and {n - 3} more")
            if common_skills:
                lines.append(f"  Common skills: {', '.join(common_skills)}")
            lines.append("")

        lines.append(
            "To capture a pattern as a reusable workflow skill:\n"
            "  propose_skill(action=\"create\", name=\"...\", body=\"...\")"
        )
        return "\n".join(lines)


def _find_workflow_clusters(
    db: sqlite3.Connection,
    *,
    max_trajectories: int = 100,
    cluster_threshold: float = 0.85,
    min_cluster_size: int = 3,
) -> list[dict[str, Any]]:
    """Cluster successful trajectories by embedding similarity.

    Returns list of cluster dicts with ``tasks`` (list of task strings)
    and ``skills_used`` (list of skill-name lists), ordered by cluster size
    descending.  Only trajectories with stored embeddings in
    ``trajectories_vec`` participate; others are silently skipped.
    """
    rows = db.execute(
        "SELECT id, task, skills_used FROM trajectories "
        "WHERE outcome = 'ok' "
        "ORDER BY created_at DESC LIMIT ?",
        (max_trajectories,),
    ).fetchall()

    if not rows:
        return []

    traj_ids = [r[0] for r in rows]
    placeholders = ",".join("?" * len(traj_ids))
    emb_rows = db.execute(
        f"SELECT trajectory_id, embedding FROM trajectories_vec "
        f"WHERE trajectory_id IN ({placeholders})",
        traj_ids,
    ).fetchall()
    emb_by_id = {r[0]: np.frombuffer(r[1], dtype=np.float32) for r in emb_rows}

    aligned = [(r[0], r[1], r[2]) for r in rows if r[0] in emb_by_id]
    if not aligned:
        return []

    vecs = [emb_by_id[traj_id] for traj_id, _, _ in aligned]
    raw_clusters = greedy_cluster(vecs, cluster_threshold)

    result: list[dict[str, Any]] = []
    for cl in raw_clusters:
        if len(cl) < min_cluster_size:
            continue
        tasks = [aligned[i][1] for i in cl]
        skills_used = [
            json.loads(aligned[i][2]) if aligned[i][2] else []
            for i in cl
        ]
        result.append({"tasks": tasks, "skills_used": skills_used})

    result.sort(key=lambda c: -len(c["tasks"]))
    return result
