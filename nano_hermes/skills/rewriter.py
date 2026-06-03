"""Failure-driven skill rewriter (SkillForge pattern).

Identifies active skills whose failure rate exceeds the configured
threshold, gathers context from failed sessions, and asks the LLM to
produce an improved version.  The rewritten body is safety-scanned and
saved as a draft (via the existing propose_skill 'edit' pipeline); the
original text is preserved in ``skill_versions`` for diff history.

Safety invariant: the judge prompt is immutable and separate from the
skill text being optimised — this prevents the metric-gaming failure
mode described in the DGM paper.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .guard import scan_skill_content

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# Judge prompt is a module-level constant so it cannot be overwritten by
# skill content.
_JUDGE_PROMPT = """\
You are auditing an AI agent skill that has been failing too often.

SKILL NAME: {skill_name}

CURRENT SKILL.md CONTENT:
---
{current_body}
---

CONTEXT FROM FAILED SESSIONS (most recent first):
---
{failure_context}
---
{localized_critique_block}
Task: analyse why this skill fails on these tasks, then produce a
rewritten SKILL.md that fixes the issues.  Keep the same markdown
structure (# name, description block, ## Usage, ## Steps sections).
Output ONLY the new SKILL.md content — no preamble, no explanation.
"""


_CRITIC_SYSTEM = """\
You are a strict AI skill-quality auditor. Your job is to judge whether a \
rewritten skill is safe to deploy. Answer EXACTLY three lines, each "YES" or "NO":

1. Does the rewrite still cover the original skill's intended use case?
2. Does the rewrite address at least one of the cited failure modes?
3. Is the rewrite free from obvious overfitting to the provided failure examples \
   (i.e. it would still be useful on unseen tasks)?

Output format — three lines, nothing else:
YES or NO
YES or NO
YES or NO"""

_CRITIC_PROMPT = """\
ORIGINAL SKILL ({skill_name}):
---
{original_body}
---

REWRITTEN SKILL:
---
{new_body}
---

FAILURE EXAMPLES USED IN REWRITE:
---
{failure_context}
---"""


@dataclass
class RewriteCandidate:
    skill_name: str
    use_count: int
    success_count: int

    @property
    def failure_rate(self) -> float:
        return (self.use_count - self.success_count) / self.use_count


def get_rewrite_candidates(
    db: sqlite3.Connection,
    *,
    failure_threshold: float,
    min_uses: int,
) -> list[RewriteCandidate]:
    """Return active skills whose failure rate exceeds *failure_threshold*.

    Only ``origin='agent'`` skills (created via propose_skill) and not
    user-pinned are eligible — builtin/external/hand-authored skills are never
    auto-rewritten. This is the single candidate source for both the rewriter
    and GEPA (``gepa.get_rewrite_candidates``)."""
    rows = db.execute(
        """
        SELECT name, use_count, success_count
        FROM skill_stats
        WHERE status = 'active'
          AND origin = 'agent'
          AND pinned = 0
          AND use_count >= ?
          AND CAST(use_count - success_count AS REAL) / use_count > ?
        ORDER BY CAST(use_count - success_count AS REAL) / use_count DESC
        """,
        (min_uses, failure_threshold),
    ).fetchall()
    return [
        RewriteCandidate(skill_name=r[0], use_count=r[1], success_count=r[2])
        for r in rows
    ]


def gather_failure_context(
    db: sqlite3.Connection, skill_name: str, limit: int = 5
) -> list[str]:
    """Return chunk contents from failed/partial sessions that used *skill_name*.

    Joins trajectories (outcome in fail/partial, skills_used mentions the
    skill) with chunks to surface what the agent was doing when it failed.
    """
    rows = db.execute(
        """
        SELECT c.content
        FROM chunks c
        JOIN trajectories t ON t.session_id = c.session_id,
             json_each(t.skills_used) j
        WHERE t.outcome IN ('fail', 'partial')
          AND j.value = ?
        ORDER BY t.created_at DESC, c.turn_index ASC
        LIMIT ?
        """,
        (skill_name, limit),
    ).fetchall()
    return [r[0] for r in rows]


def save_skill_version(
    db: sqlite3.Connection, skill_name: str, body: str, reason: str
) -> None:
    """Persist a snapshot of *body* into ``skill_versions`` before rewriting."""
    db.execute(
        "INSERT INTO skill_versions (skill_name, body, reason, created_at) "
        "VALUES (?, ?, ?, ?)",
        (skill_name, body, reason, time.time()),
    )
    db.commit()


def _parse_critic_response(text: str) -> bool:
    """Return True only if all three critic lines are YES."""
    lines = [ln.strip().upper() for ln in text.strip().splitlines() if ln.strip()]
    if len(lines) < 3:
        return False
    return all(ln.startswith("YES") for ln in lines[:3])


async def _run_critic(
    hook: "NanoHermesHook",
    *,
    skill_name: str,
    original_body: str,
    new_body: str,
    failure_context: str,
) -> bool:
    """Ask an independent LLM call to approve the rewrite.

    Returns True if the critic approves (all three rubric questions answered YES).
    Retries once on network/parse failure; rejects on second failure.
    """
    prompt = _CRITIC_PROMPT.format(
        skill_name=skill_name,
        original_body=original_body,
        new_body=new_body,
        failure_context=failure_context,
    )
    last_raw: str | None = None
    for attempt in range(2):
        try:
            resp = await hook._loop.provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": _CRITIC_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=getattr(hook._loop, "model", None),
                max_tokens=20,
            )
            last_raw = (resp.content or "").strip()
            approved = _parse_critic_response(last_raw)
            if approved:
                return True
            # Unambiguous rejection — no point retrying.
            log.info(
                "rewriter: critic rejected %s — %r", skill_name, last_raw
            )
            return False
        except Exception:
            log.warning(
                "rewriter: critic LLM call failed for %s (attempt %d/2)",
                skill_name,
                attempt + 1,
                exc_info=True,
            )
    log.warning("rewriter: critic gave up on %s after 2 failures — rejecting", skill_name)
    return False


async def rewrite_skill(
    hook: "NanoHermesHook",
    candidate: RewriteCandidate,
) -> str | None:
    """Run the full rewrite pipeline for *candidate*.

    Returns the new skill body on success, or ``None`` if the pipeline
    failed (reasons logged at WARNING level).
    """
    cfg = hook.config.skill_stats
    skill_path = hook.workspace / "skills" / candidate.skill_name / "SKILL.md"
    if not skill_path.exists():
        log.warning("rewriter: %s — SKILL.md not found, skipping", candidate.skill_name)
        return None

    current_body = skill_path.read_text()
    failure_contexts = gather_failure_context(
        hook.db, candidate.skill_name, limit=cfg.rewrite_context_chunks
    )

    if not failure_contexts:
        log.info(
            "rewriter: %s — no failed-session chunks found, skipping",
            candidate.skill_name,
        )
        return None

    context_text = "\n---\n".join(failure_contexts)

    # AgentPRM-lite: localize the failing step before rewriting so the
    # rewrite prompt knows what to target. Fail-soft — empty critique
    # falls back to the legacy "no localization" flow.
    localized_critique_block = ""
    if cfg.rewrite_step_localization_enabled:
        from .step_localize import localize_failure_step  # noqa: PLC0415
        critique = await localize_failure_step(
            hook,
            skill_name=candidate.skill_name,
            current_body=current_body,
            failure_context=context_text,
        )
        if critique:
            log.info(
                "rewriter: %s — localized defect: %s",
                candidate.skill_name, critique,
            )
            localized_critique_block = (
                f"\nLOCALIZED DEFECT (from prior step-judge pass):\n{critique}\n"
            )

    prompt = _JUDGE_PROMPT.format(
        skill_name=candidate.skill_name,
        current_body=current_body,
        failure_context=context_text,
        localized_critique_block=localized_critique_block,
    )

    try:
        response = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(hook._loop, "model", None),
            max_tokens=2048,
        )
        new_body = (response.content or "").strip()
    except Exception:
        log.exception("rewriter: LLM call failed for skill %s", candidate.skill_name)
        return None

    if not new_body:
        log.warning("rewriter: empty response for skill %s", candidate.skill_name)
        return None

    # Safety gate — same check as propose_skill create/edit.
    err = scan_skill_content(new_body)
    if err:
        log.warning("rewriter: security scan blocked rewrite of %s — %s", candidate.skill_name, err)
        return None

    # Critic gate — independent LLM call with immutable system prompt.
    if cfg.rewrite_critic_enabled:
        approved = await _run_critic(
            hook,
            skill_name=candidate.skill_name,
            original_body=current_body,
            new_body=new_body,
            failure_context=context_text,
        )
        if not approved:
            log.info("rewriter: critic blocked rewrite of %s", candidate.skill_name)
            return None

    # ASG-SI replay gate — counterfactually replay failing trajectories
    # against the candidate body. Blocks promotion on any WORSE verdict.
    if cfg.rewrite_replay_gate_enabled:
        from .replay_gate import replay_passes_gate  # noqa: PLC0415
        passed = await replay_passes_gate(
            hook,
            skill_name=candidate.skill_name,
            original_body=current_body,
            new_body=new_body,
            min_pass_rate=cfg.rewrite_replay_min_pass_rate,
            max_trajectories=cfg.rewrite_replay_max_trajectories,
        )
        if not passed:
            log.info("rewriter: replay gate blocked rewrite of %s", candidate.skill_name)
            return None

    # Preserve the old version before overwriting.
    save_skill_version(
        hook.db,
        candidate.skill_name,
        current_body,
        reason=f"auto-rewrite trigger: failure_rate={candidate.failure_rate:.0%}",
    )

    return new_body


async def run_rewriter(
    hook: "NanoHermesHook",
    skip: frozenset[str] = frozenset(),
) -> list[str]:
    """Identify failing skills, rewrite them, and submit via propose_skill edit.

    Returns list of skill names that were successfully rewritten.
    Designed to be called from the dream/cron cycle, not per-turn.

    *skip* names skills already evolved by GEPA this cycle so they are not
    double-rewritten in the same pass.
    """
    cfg = hook.config.skill_stats
    candidates = get_rewrite_candidates(
        hook.db,
        failure_threshold=cfg.rewrite_failure_threshold,
        min_uses=cfg.rewrite_min_uses,
    )
    if skip:
        candidates = [c for c in candidates if c.skill_name not in skip]
    if not candidates:
        return []

    rewritten: list[str] = []
    for candidate in candidates:
        log.info(
            "rewriter: %s — %.0f%% failure rate (%d/%d uses), attempting rewrite",
            candidate.skill_name,
            candidate.failure_rate * 100,
            candidate.use_count - candidate.success_count,
            candidate.use_count,
        )
        new_body = await rewrite_skill(hook, candidate)
        if new_body is None:
            continue

        # Pass description="" — _edit only uses it in the success message;
        # SkillIndexer re-extracts description from the file on next search.
        # Use ProposeSkillTool's internal _edit path — import lazily to
        # avoid circular imports at module load time.
        from .propose_tool import ProposeSkillTool  # noqa: PLC0415
        tool = ProposeSkillTool(hook=hook)
        result = await tool._edit(
            skill_name=candidate.skill_name,
            description="",
            body=new_body,
            files=[],
            delete_files=[],
            redaction_note=" [auto-rewritten by SkillRewriter]",
        )
        if result.startswith("Error"):
            log.warning("rewriter: edit failed for %s — %s", candidate.skill_name, result)
        else:
            log.info("rewriter: successfully rewrote %s", candidate.skill_name)
            rewritten.append(candidate.skill_name)

    return rewritten


