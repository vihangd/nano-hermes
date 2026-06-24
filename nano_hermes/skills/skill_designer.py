"""MemSkill designer loop (arXiv:2602.02474).

Detects sessions where NO skill was used and the outcome was fail/partial,
clusters them by task-embedding cosine similarity, and for each cluster of
``>= min_cluster_size`` sessions proposes a NEW skill to fill the coverage gap.

Complements GEPA/rewriter (which improve *existing* failing skills) by
addressing the orthogonal problem: tasks for which the agent has *no skill at
all*.  One LLM call per cluster; zero calls when there are no gaps or the
feature is disabled.

Called from ``_run_evolution_cycle`` after Ratchet, before OPRO.
Default-off (``skill_designer_enabled = False``).
"""
from __future__ import annotations

import logging
import re
import time
from typing import TYPE_CHECKING

from .guard import scan_skill_content

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Immutable prompt — never derived from skill content
# ---------------------------------------------------------------------------

_DESIGNER_PROMPT = """\
You are designing a new AI agent skill to handle a task category that the agent currently has no skill for.

SIMILAR FAILED TASKS (the agent had no skill to help with these):
---
{task_examples}
---

Design a new SKILL.md file that would help the agent handle tasks like these.
Use this exact format:
---
name: <skill_name_snake_case>
description: <one-line description>
---

## When to Use
<conditions>

## Steps
<numbered steps>

Output ONLY the SKILL.md content. Skill name must be unique and descriptive.
"""

_NAME_RE = re.compile(r"^name:\s*([a-z0-9][a-z0-9_-]{0,63})\s*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Clustering helper
# ---------------------------------------------------------------------------


def _cluster_sessions(
    sessions: list[tuple[int, str, bytes]],
    threshold: float,
) -> list[list[int]]:
    """Cluster sessions by task-embedding cosine similarity.

    Delegates to ``consolidation.greedy_cluster`` (running-mean centroids).
    Returns clusters as lists of indices into ``sessions``; None-embedding
    sessions are excluded from all clusters.
    """
    import numpy as np  # noqa: PLC0415
    from ..memory.consolidation import greedy_cluster  # noqa: PLC0415

    valid_indices: list[int] = []
    valid_vecs: list[np.ndarray] = []
    for idx, (_tid, _task, emb_bytes) in enumerate(sessions):
        if emb_bytes is None:
            continue
        try:
            valid_vecs.append(np.frombuffer(emb_bytes, dtype=np.float32).copy())
            valid_indices.append(idx)
        except Exception:
            pass

    if not valid_vecs:
        return []

    raw = greedy_cluster(valid_vecs, threshold)
    return [[valid_indices[i] for i in cluster] for cluster in raw]


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def run_skill_designer(hook: "NanoHermesHook") -> list[str]:
    """Propose new skills for task categories with no existing skill coverage.

    Returns list of skill names successfully proposed this cycle.
    """
    from ..utils.error_classifier import EvolutionAbortError, classify_llm_response  # noqa: PLC0415
    from .propose_tool import ProposeSkillTool  # noqa: PLC0415

    cfg = hook.config.skill_stats
    if not getattr(cfg, "skill_designer_enabled", False):
        return []

    lookback_days: int = getattr(cfg, "skill_designer_lookback_days", 30)
    max_candidates: int = getattr(cfg, "skill_designer_max_candidates", 50)
    min_cluster_size: int = getattr(cfg, "skill_designer_min_cluster_size", 3)
    cosine_threshold: float = 0.75

    cutoff = time.time() - lookback_days * 86400

    # 1. Query no-skill failures with first-chunk embedding via scalar subquery.
    # chunks_vec is a sqlite_vec virtual table; rowid is not selectable as a
    # column alias in a JOIN, so we use a correlated scalar subquery instead.
    rows = hook.db.execute(
        """
        SELECT t.id, t.task,
            (SELECT embedding FROM chunks_vec
             WHERE chunk_id = (
                 SELECT c.id FROM chunks c
                 WHERE c.session_id = t.session_id
                 ORDER BY c.turn_index ASC
                 LIMIT 1
             )
            ) AS embedding
        FROM trajectories t
        WHERE t.outcome IN ('fail', 'partial')
          AND (t.skills_used IS NULL
               OR t.skills_used = '[]'
               OR json_array_length(t.skills_used) = 0)
          AND t.created_at > ?
        ORDER BY t.created_at DESC
        LIMIT ?
        """,
        (cutoff, max_candidates),
    ).fetchall()

    if not rows:
        log.debug("skill_designer: no no-skill failures in window, skipping")
        return []

    # 2. Cluster by cosine similarity on embeddings
    sessions = [(r[0], r[1], r[2]) for r in rows]
    clusters = _cluster_sessions(sessions, cosine_threshold)

    # Filter to clusters large enough
    eligible = [c for c in clusters if len(c) >= min_cluster_size]
    if not eligible:
        log.debug(
            "skill_designer: %d session(s) found but no cluster >= %d, skipping",
            len(sessions), min_cluster_size,
        )
        return []

    proposed: list[str] = []
    tool = ProposeSkillTool(hook=hook)

    for cluster in eligible:
        task_texts = [sessions[i][1] for i in cluster]
        examples = "\n---\n".join(
            f"{i + 1}. {t[:300]}" for i, t in enumerate(task_texts[:5])
        )
        prompt = _DESIGNER_PROMPT.format(task_examples=examples)

        try:
            resp = await hook._loop.provider.chat_with_retry(
                messages=[{"role": "user", "content": prompt}],
                model=getattr(hook._loop, "model", None),
                max_tokens=1024,
            )
            err = classify_llm_response(resp)
            if err is not None:
                log.warning("skill_designer: LLM error — %s", err.reason.value)
                if err.should_abort:
                    raise EvolutionAbortError(err)
                continue
            body = (resp.content or "").strip()
        except EvolutionAbortError:
            raise
        except Exception:
            log.exception("skill_designer: LLM call failed for cluster of %d", len(cluster))
            continue

        if not body:
            log.warning("skill_designer: empty LLM response for cluster of %d", len(cluster))
            continue

        # Parse skill name from SKILL.md header
        m = _NAME_RE.search(body)
        if not m:
            log.warning("skill_designer: could not parse skill name from response")
            continue
        skill_name = m.group(1)

        # Security scan
        scan_err = scan_skill_content(body)
        if scan_err:
            log.warning("skill_designer: security scan blocked '%s' — %s", skill_name, scan_err)
            continue

        # Write-approval gate
        from ..governance import write_approval as wa  # noqa: PLC0415
        if wa.is_gated(hook, "skills"):
            wa.stage_skill_write(
                hook,
                skill_name=skill_name,
                description="",
                body=body,
                reason=f"skill_designer: coverage gap ({len(cluster)} no-skill failures)",
                origin="skill_designer",
            )
            log.info("skill_designer: staged '%s' for approval (gate=approve)", skill_name)
            proposed.append(skill_name)
            continue

        result = await tool._create(
            skill_name=skill_name,
            description="",
            body=body,
            files=[],
            redaction_note=" [designed by SkillDesigner]",
        )
        if result.startswith("Error"):
            log.warning("skill_designer: create failed for '%s' — %s", skill_name, result)
        else:
            log.info(
                "skill_designer: proposed '%s' (cluster size=%d)", skill_name, len(cluster)
            )
            proposed.append(skill_name)

    return proposed
