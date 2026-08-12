"""OPRO prompt meta-optimization (arXiv:2309.03409).

Periodically treats nano-hermes's internal evolution prompts as optimizable
objects. A meta-LLM call sees the history of (prompt, score) pairs (sorted
worst→best) and generates improved candidates. The best-scoring candidate
is promoted as the active prompt for subsequent GEPA rounds.

Score metric for 'gepa_mutation':
  score = gepa_improvements / gepa_total_rounds (over a rolling window)
  where improvements = skill_versions rows with reason LIKE 'gepa:%'
  and total_rounds is estimated from improvement events + cycle tracking.

Safety:
- Candidate prompts must retain all required template variables; variants
  that drop a variable are rejected before insertion.
- OPRO fires at most once per opro_cycle_interval evolution cycles.
- Default off (opro_enabled = False). No LLM calls when disabled.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# Required template variables for each prompt name — missing any → reject.
_REQUIRED_VARS: dict[str, list[str]] = {
    "gepa_mutation": [
        "{skill_name}", "{round_n}", "{max_rounds}",
        "{lens}", "{current_body}", "{failure_context}",
    ],
}

_META_PROMPT_TEMPLATE = """\
You are optimizing the mutation prompt used in an AI skill evolution system.

Here are previous mutation prompt versions with their performance scores (higher = better),
sorted worst to best:
{history_block}

Score measures: what percentage of GEPA evolution rounds produced a Pareto-dominant
skill mutation (higher = more rounds succeeded). Range 0–100.

Generate a NEW mutation prompt that would achieve a HIGHER score.
The prompt MUST include these exact template variables (copy them verbatim):
  {{skill_name}}, {{round_n}}, {{max_rounds}}, {{lens}}, {{current_body}}, {{failure_context}}

{{lens}} is substituted with a per-round inspection lens that varies by round —
place it where it will frame how the skill is examined.

Output ONLY the new prompt text — no explanation, no preamble.
"""


# ---- DB helpers ------------------------------------------------------------


def get_active_prompt(
    db: sqlite3.Connection,
    prompt_name: str,
    fallback: str,
) -> str:
    """Return the active OPRO-optimized prompt, or *fallback* if none exists.

    The active prompt is re-validated on read, not just at promotion time: when
    a new required variable is introduced, prompts promoted under the older
    contract are still sitting in the table marked active. Serving one would
    silently drop whatever that variable controls (no error — ``str.format``
    ignores unused kwargs), so a stale prompt falls back to the built-in.
    """
    row = db.execute(
        "SELECT body FROM prompt_versions WHERE prompt_name = ? AND active = 1 "
        "ORDER BY scored_at DESC LIMIT 1",
        (prompt_name,),
    ).fetchone()
    if not row:
        return fallback
    if not _validate_prompt(row[0], prompt_name):
        log.warning(
            "active %s prompt is missing required variables — using built-in "
            "fallback until OPRO promotes a conforming one",
            prompt_name,
        )
        return fallback
    return row[0]


def _score_prompt(db: sqlite3.Connection, window_days: int = 30) -> float | None:
    """Estimate GEPA mutation success rate over the last *window_days*.

    Uses skill_versions rows with ``reason LIKE 'gepa:%'`` as the proxy for
    rounds that produced a Pareto-dominant mutation.  Returns ``None`` when
    fewer than 5 improvement events exist (insufficient evidence).
    """
    cutoff = time.time() - window_days * 86400
    improvements: int = db.execute(
        "SELECT COUNT(*) FROM skill_versions "
        "WHERE reason LIKE 'gepa:%' AND created_at >= ?",
        (cutoff,),
    ).fetchone()[0]
    if improvements < 5:
        return None
    # Approximate total rounds from cycle counter stored in meta table.
    row = db.execute(
        "SELECT value FROM meta WHERE key = 'opro_total_gepa_rounds'"
    ).fetchone()
    total_rounds = int(row[0]) if row else improvements  # fallback: assume 100% rate
    if total_rounds <= 0:
        return None
    return min(100.0, improvements / total_rounds * 100)


def _increment_gepa_rounds(db: sqlite3.Connection, n: int) -> None:
    """Add *n* to the running gepa-rounds counter in the meta table."""
    db.execute(
        "INSERT INTO meta (key, value) VALUES ('opro_total_gepa_rounds', ?) "
        "ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + ?",
        (str(n), n),
    )
    db.commit()


def record_gepa_rounds(db: sqlite3.Connection, rounds_run: int) -> None:
    """Called by run_gepa after each candidate to track total rounds for OPRO score."""
    if rounds_run > 0:
        _increment_gepa_rounds(db, rounds_run)


def _validate_prompt(body: str, prompt_name: str) -> bool:
    """Return True if *body* contains all required template variables."""
    required = _REQUIRED_VARS.get(prompt_name, [])
    return all(var in body for var in required)


def _build_history_block(history: list[tuple[str, float]]) -> str:
    """Format (body, score) pairs as a worst→best block for the meta-prompt."""
    if not history:
        return "(no prior versions scored yet)"
    lines = []
    for body, score in history:
        preview = body[:200].replace("\n", " ")
        lines.append(f"score={score:.1f} | prompt preview: \"{preview}...\"")
    return "\n".join(lines)


# ---- Main OPRO runner ------------------------------------------------------


async def run_opro(hook: "NanoHermesHook") -> bool:
    """Run one OPRO round if conditions are met.

    Returns True if new candidates were generated, False otherwise.
    """
    cfg = hook.config.skill_stats
    if not getattr(cfg, "opro_enabled", False):
        return False

    cycle_interval: int = getattr(cfg, "opro_cycle_interval", 20)
    candidates_per_round: int = getattr(cfg, "opro_candidates_per_round", 8)
    prompt_name = "gepa_mutation"
    db = hook.db

    # Gate: only fire every N evolution cycles.
    count = hook._evolution_cycle_count
    if count == 0 or count % cycle_interval != 0:
        return False

    # Score the current active prompt (or module-level fallback).
    current_score = _score_prompt(db)
    if current_score is None:
        log.info("opro: insufficient GEPA data to score — skipping this round")
        return False

    # Score any previously generated but unscored candidates.
    unscored = db.execute(
        "SELECT id, body FROM prompt_versions "
        "WHERE prompt_name = ? AND score IS NULL",
        (prompt_name,),
    ).fetchall()
    for pid, body in unscored:
        # Use the same window-based score function as a proxy for all candidates
        # (we can't replay each individually without replaying GEPA, so we
        # approximate: the currently observed rate applies uniformly while the
        # candidate was active between OPRO rounds).
        db.execute(
            "UPDATE prompt_versions SET score = ?, scored_at = ? WHERE id = ?",
            (current_score, time.time(), pid),
        )
    if unscored:
        db.commit()

    # Build history: last 8 scored variants, sorted worst→best.
    history_rows = db.execute(
        "SELECT body, score FROM prompt_versions "
        "WHERE prompt_name = ? AND score IS NOT NULL "
        "ORDER BY score ASC LIMIT 8",
        (prompt_name,),
    ).fetchall()
    history = [(r[0], r[1]) for r in history_rows]

    # Add current active score as the latest entry if not already in DB.
    history_block = _build_history_block(history + [(
        "(current active prompt)", current_score
    )])

    meta_prompt = _META_PROMPT_TEMPLATE.format(history_block=history_block)

    from ..utils.error_classifier import EvolutionAbortError, classify_llm_response  # noqa: PLC0415

    generated = 0
    for i in range(candidates_per_round):
        try:
            resp = await hook._loop.provider.chat_with_retry(
                messages=[{"role": "user", "content": meta_prompt}],
                model=getattr(hook._loop, "model", None),
                max_tokens=1024,
            )
            err = classify_llm_response(resp)
            if err is not None:
                log.warning("opro: LLM call error (candidate %d) — %s", i + 1, err.reason.value)
                if err.should_abort:
                    raise EvolutionAbortError(err)
                continue
            new_body = (resp.content or "").strip()
        except EvolutionAbortError:
            raise
        except Exception:
            log.warning("opro: LLM call failed for candidate %d", i + 1, exc_info=True)
            continue

        if not new_body:
            continue
        if not _validate_prompt(new_body, prompt_name):
            log.warning(
                "opro: candidate %d missing required template variables — rejected", i + 1
            )
            continue

        db.execute(
            "INSERT INTO prompt_versions (prompt_name, body, active, created_at) "
            "VALUES (?, ?, 0, ?)",
            (prompt_name, new_body, time.time()),
        )
        generated += 1

    if generated:
        db.commit()
        log.info("opro: generated %d new %s candidates", generated, prompt_name)

    # Promote the best-scored candidate if it beats the current rate.
    best_row = db.execute(
        "SELECT id, body, score FROM prompt_versions "
        "WHERE prompt_name = ? AND score IS NOT NULL "
        "ORDER BY score DESC LIMIT 1",
        (prompt_name,),
    ).fetchone()
    if best_row and best_row[2] is not None and best_row[2] > current_score:
        # Deactivate all, activate winner.
        db.execute(
            "UPDATE prompt_versions SET active = 0 WHERE prompt_name = ?",
            (prompt_name,),
        )
        db.execute(
            "UPDATE prompt_versions SET active = 1 WHERE id = ?",
            (best_row[0],),
        )
        db.commit()
        log.info(
            "opro: promoted new active %s (score %.1f > current %.1f)",
            prompt_name, best_row[2], current_score,
        )

    return generated > 0
