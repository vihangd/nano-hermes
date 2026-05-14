"""GEPA-inspired skill text evolution (Genetic-Pareto Prompt Evolution).

Iteratively mutates active skill text using LLM calls, tracks a Pareto
frontier over (estimated_improvements, token_count), and promotes the
best mutation via the existing propose_skill edit pipeline.

Reference: GEPA (arXiv:2507.19457, ICLR 2026 Oral) — adapted to:
  - inference-only (no gradient updates, no GPU)
  - nano-hermes skill format (SKILL.md)
  - Pareto-dominant selection over 2 objectives:
      maximize  estimated_improvements  (LLM reflection judge)
      minimize  token_count             (Pi 3B+ context budget)

Safety invariant: both prompts are module-level constants — they cannot
be overwritten by the skill text being evolved.

Relation to the SkillForge rewriter (rewriter.py):
  - Rewriter: one-shot, severe failures (default ≥60% failure rate)
  - GEPA:     iterative, moderate failures (default ≥40% failure rate)
  - GEPA runs first; a skill that GEPA cannot improve falls through to
    the rewriter on the next cycle.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..memory.budgets import _count_tokens
from .guard import scan_skill_content
from .rewriter import gather_failure_context, get_rewrite_candidates, save_skill_version

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Immutable prompts — never derived from skill content
# ---------------------------------------------------------------------------

_MUTATION_PROMPT = """\
You are improving an AI agent skill that has been failing.

SKILL NAME: {skill_name}
MUTATION ROUND: {round_n}/{max_rounds}

CURRENT SKILL TEXT:
---
{current_body}
---

EXAMPLES OF FAILURES (sessions where this skill was used but the task failed):
---
{failure_context}
---

Produce an improved SKILL.md that would handle these failures better.
Keep the same markdown structure (# name, description block, ## sections).
Output ONLY the new SKILL.md content — no preamble, no explanation.
"""

_EVALUATION_PROMPT = """\
You are evaluating whether an improved agent skill would have prevented past failures.

SKILL NAME: {skill_name}

ORIGINAL SKILL:
---
{original_body}
---

IMPROVED SKILL:
---
{new_body}
---

For each failure below, answer Y if the improved skill would likely have prevented
it, or N if not. Respond with exactly {n} lines, each containing only Y or N.

FAILURES:
{numbered_failures}
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class GepaMutation:
    body: str
    estimated_improvements: int  # count of Y judgements from evaluation prompt
    token_count: int              # tiktoken count — our proxy for context cost


@dataclass
class GepaCandidate:
    skill_name: str
    use_count: int
    success_count: int

    @property
    def failure_rate(self) -> float:
        return (self.use_count - self.success_count) / self.use_count


# ---------------------------------------------------------------------------
# Pareto helpers
# ---------------------------------------------------------------------------


def _pareto_dominates(a: GepaMutation, b: GepaMutation) -> bool:
    """Return True if *a* lexicographically dominates *b*.

    Primary objective: maximize estimated_improvements (more repairs is always
    better — a mutation that fixes one more failure wins regardless of size).
    Secondary objective: minimize token_count (Pi 3B+ context budget).

    This differs from strict bi-objective Pareto: improvements takes absolute
    priority. Token count only breaks ties between mutations with equal improvements.
    """
    if a.estimated_improvements > b.estimated_improvements:
        return True
    if a.estimated_improvements == b.estimated_improvements and a.token_count < b.token_count:
        return True
    return False


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------


def _parse_yn_response(text: str, expected_n: int) -> int:
    """Count Y answers in an LLM response of Y/N lines.

    Robust to extra whitespace, mixed case, and trailing punctuation.
    Returns a count in [0, expected_n].
    """
    lines = [ln.strip().upper() for ln in text.strip().splitlines() if ln.strip()]
    yn_lines = [ln for ln in lines if re.match(r"^[YN][.:]?$", ln)]
    # If the model added prose, take only the first expected_n clean lines.
    count = sum(1 for ln in yn_lines[:expected_n] if ln.startswith("Y"))
    return count


# ---------------------------------------------------------------------------
# Candidate discovery
# ---------------------------------------------------------------------------


def get_gepa_candidates(
    db: sqlite3.Connection,
    *,
    failure_threshold: float,
    min_uses: int,
) -> list[GepaCandidate]:
    """Return active skills in the GEPA target range.

    Intentionally uses the same SQL as get_rewrite_candidates — the
    threshold values differ (caller passes the GEPA-specific ones).
    """
    rows = get_rewrite_candidates(db, failure_threshold=failure_threshold, min_uses=min_uses)
    return [
        GepaCandidate(
            skill_name=r.skill_name,
            use_count=r.use_count,
            success_count=r.success_count,
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Core GEPA loop for a single skill
# ---------------------------------------------------------------------------


async def evolve_skill(
    hook: "NanoHermesHook",
    candidate: GepaCandidate,
    *,
    max_rounds: int,
    minibatch_size: int,
) -> GepaMutation | None:
    """Run GEPA evolution for *candidate*.

    Returns the Pareto-best mutation found, or ``None`` if evolution
    produced no improvement over the original (or failed entirely).
    """
    skill_path = hook.workspace / "skills" / candidate.skill_name / "SKILL.md"
    if not skill_path.exists():
        log.warning("gepa: %s — SKILL.md not found", candidate.skill_name)
        return None

    original_body = skill_path.read_text()
    failure_contexts = gather_failure_context(
        hook.db, candidate.skill_name, limit=minibatch_size
    )
    if not failure_contexts:
        log.info("gepa: %s — no failed-session chunks, skipping", candidate.skill_name)
        return None

    numbered_failures = "\n".join(
        f"{i + 1}. {ctx[:400]}" for i, ctx in enumerate(failure_contexts)
    )
    failure_text = "\n---\n".join(failure_contexts)

    # Seed the Pareto frontier with the original body (0 improvements baseline).
    baseline = GepaMutation(
        body=original_body,
        estimated_improvements=0,
        token_count=_count_tokens(original_body),
    )
    best = baseline
    current_body = original_body  # start each round from the Pareto-best body

    for round_n in range(1, max_rounds + 1):
        log.debug("gepa: %s — round %d/%d", candidate.skill_name, round_n, max_rounds)

        # --- Step 1: mutate ---
        mutation_prompt = _MUTATION_PROMPT.format(
            skill_name=candidate.skill_name,
            round_n=round_n,
            max_rounds=max_rounds,
            current_body=current_body,
            failure_context=failure_text,
        )
        try:
            resp = await hook._loop.provider.chat_with_retry(
                messages=[{"role": "user", "content": mutation_prompt}],
                max_tokens=2048,
            )
            new_body = (resp.content or "").strip()
        except Exception:
            log.exception("gepa: mutation LLM call failed — %s round %d", candidate.skill_name, round_n)
            continue

        if not new_body or new_body == current_body:
            log.debug("gepa: %s round %d — empty or unchanged mutation", candidate.skill_name, round_n)
            continue

        err = scan_skill_content(new_body)
        if err:
            log.warning("gepa: security scan blocked mutation of %s — %s", candidate.skill_name, err)
            continue

        # --- Step 2: evaluate ---
        eval_prompt = _EVALUATION_PROMPT.format(
            skill_name=candidate.skill_name,
            original_body=original_body,
            new_body=new_body,
            n=len(failure_contexts),
            numbered_failures=numbered_failures,
        )
        try:
            eval_resp = await hook._loop.provider.chat_with_retry(
                messages=[{"role": "user", "content": eval_prompt}],
                max_tokens=len(failure_contexts) * 4 + 20,
            )
            improvements = _parse_yn_response(
                eval_resp.content or "", len(failure_contexts)
            )
        except Exception:
            log.exception("gepa: evaluation LLM call failed — %s round %d", candidate.skill_name, round_n)
            continue

        candidate_mutation = GepaMutation(
            body=new_body,
            estimated_improvements=improvements,
            token_count=_count_tokens(new_body),
        )

        log.info(
            "gepa: %s round %d — improvements=%d/%d, tokens=%d→%d",
            candidate.skill_name,
            round_n,
            improvements,
            len(failure_contexts),
            baseline.token_count,
            candidate_mutation.token_count,
        )

        # --- Step 3: Pareto update ---
        if _pareto_dominates(candidate_mutation, best):
            best = candidate_mutation
            current_body = new_body  # next round starts from this body
            log.info(
                "gepa: %s round %d — new Pareto-best (improvements=%d, tokens=%d)",
                candidate.skill_name, round_n, best.estimated_improvements, best.token_count,
            )

    # Only return if we genuinely improved (improvements > 0 AND body changed).
    # The Pareto check can select a shorter body with 0 improvements as "best"
    # during intermediate rounds — guard against promoting those.
    if best.estimated_improvements > 0 and best.body != original_body:
        return best
    return None


# ---------------------------------------------------------------------------
# Top-level runner — called from dream/cron cycle
# ---------------------------------------------------------------------------


async def run_gepa(hook: "NanoHermesHook") -> list[str]:
    """Run GEPA evolution over all eligible skills.

    Returns the list of skill names whose text was updated.
    Designed to be called from the dream/cron cycle, not per-turn.
    """
    cfg = hook.config.skill_stats
    if not getattr(cfg, "gepa_enabled", False):
        return []

    failure_threshold = getattr(cfg, "gepa_failure_threshold", 0.4)
    min_uses = getattr(cfg, "gepa_min_uses", 5)
    max_rounds = getattr(cfg, "gepa_max_mutations", 3)
    minibatch_size = getattr(cfg, "gepa_minibatch_size", 3)

    candidates = get_gepa_candidates(
        hook.db, failure_threshold=failure_threshold, min_uses=min_uses
    )
    if not candidates:
        return []

    evolved: list[str] = []
    for candidate in candidates:
        log.info(
            "gepa: %s — %.0f%% failure rate (%d/%d uses), starting evolution",
            candidate.skill_name,
            candidate.failure_rate * 100,
            candidate.use_count - candidate.success_count,
            candidate.use_count,
        )
        best = await evolve_skill(
            hook, candidate, max_rounds=max_rounds, minibatch_size=minibatch_size
        )
        if best is None:
            log.info("gepa: %s — no Pareto improvement found", candidate.skill_name)
            continue

        # Preserve original before overwriting.
        skill_path = hook.workspace / "skills" / candidate.skill_name / "SKILL.md"
        save_skill_version(
            hook.db,
            candidate.skill_name,
            skill_path.read_text(),
            reason=(
                f"gepa: Pareto-best mutation "
                f"(improvements={best.estimated_improvements}, tokens={best.token_count}, "
                f"failure_rate={candidate.failure_rate:.0%})"
            ),
        )

        from .propose_tool import ProposeSkillTool  # noqa: PLC0415
        tool = ProposeSkillTool(hook=hook)
        result = await tool._edit(
            skill_name=candidate.skill_name,
            description="",
            body=best.body,
            files=[],
            delete_files=[],
            redaction_note=" [evolved by GEPA]",
        )
        if result.startswith("Error"):
            log.warning("gepa: edit failed for %s — %s", candidate.skill_name, result)
        else:
            log.info(
                "gepa: successfully evolved %s (improvements=%d, tokens=%d)",
                candidate.skill_name,
                best.estimated_improvements,
                best.token_count,
            )
            evolved.append(candidate.skill_name)

    return evolved
