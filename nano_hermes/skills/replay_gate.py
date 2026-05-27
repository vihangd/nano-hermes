"""ASG-SI replay gate (Phase 8).

Before promoting a rewritten skill draft to active, replay the last N
failing trajectories against the candidate body using an LLM judge:
"given the new SKILL.md, would the agent have succeeded on this
trajectory?". The candidate is accepted only if a majority of the
replays judge an improvement — a counterfactual "monotone improvement"
check.

True execution replay is infeasible on Pi 3B+ (no local LLM), so this
is a counterfactual judge in the spirit of ASG-SI (arXiv 2512.23760).
Fails open on any error — never blocks a rewrite for infrastructure
reasons.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


_REPLAY_SYSTEM = """\
You are evaluating whether a rewritten AI agent skill would have changed \
the outcome of a previously failed task. Answer with EXACTLY one word on \
one line: IMPROVED, SAME, or WORSE.

- IMPROVED: the rewritten skill would likely have led the agent to succeed.
- SAME: same outcome — the failure was caused by something outside this skill.
- WORSE: the rewrite removes guidance the agent needed; it would fail harder.
"""

_REPLAY_PROMPT = """\
SKILL NAME: {skill_name}

ORIGINAL SKILL (under which this trajectory failed):
---
{original_body}
---

REWRITTEN SKILL CANDIDATE:
---
{new_body}
---

FAILURE TRAJECTORY (chunks from the failing session, in order):
---
{trajectory_text}
---

One word: IMPROVED, SAME, or WORSE."""


_VALID_VERDICTS = {"IMPROVED", "SAME", "WORSE"}


def gather_failure_trajectories(
    db: sqlite3.Connection,
    skill_name: str,
    *,
    max_trajectories: int = 3,
    chunks_per_trajectory: int = 6,
) -> list[str]:
    """Return up to *max_trajectories* failure traces (joined chunk text per session).

    Each trace is the most recent ``chunks_per_trajectory`` chunks from a
    fail/partial session that listed this skill in its skills_used.
    Sessions are ordered most-recent-first.
    """
    sessions = db.execute(
        """
        SELECT DISTINCT t.session_id
        FROM trajectories t,
             json_each(t.skills_used) j
        WHERE t.outcome IN ('fail', 'partial')
          AND j.value = ?
          AND t.session_id IS NOT NULL
        ORDER BY t.created_at DESC
        LIMIT ?
        """,
        (skill_name, max_trajectories),
    ).fetchall()
    if not sessions:
        return []
    traces: list[str] = []
    for (sid,) in sessions:
        chunks = db.execute(
            """
            SELECT role, content FROM chunks
            WHERE session_id = ?
            ORDER BY turn_index DESC
            LIMIT ?
            """,
            (sid, chunks_per_trajectory),
        ).fetchall()
        # Reverse to chronological order for human readability.
        ordered = list(reversed(chunks))
        trace = "\n".join(
            f"[{role}] {content[:400]}{'…' if len(content) > 400 else ''}"
            for role, content in ordered
        )
        if trace:
            traces.append(trace)
    return traces


def _parse_verdict(text: str) -> str | None:
    """Return one of IMPROVED/SAME/WORSE, or None if the response is unparseable."""
    line = (text or "").strip().splitlines()[0:1]
    if not line:
        return None
    token = line[0].strip().upper().rstrip(".,!?:")
    return token if token in _VALID_VERDICTS else None


async def replay_passes_gate(
    hook: "NanoHermesHook",
    *,
    skill_name: str,
    original_body: str,
    new_body: str,
    min_pass_rate: float = 0.6,
    max_trajectories: int = 3,
) -> bool:
    """Counterfactually replay failing trajectories; require majority improvement.

    Returns True if at least ``min_pass_rate`` of the replays judge
    IMPROVED. Returns True (fails open) when no failure trajectories are
    available — there's nothing to regress against. Returns True on any
    infrastructure failure so the rewrite path isn't blocked by LLM
    flakiness.
    """
    provider = getattr(hook._loop, "provider", None)
    if provider is None:
        return True
    model = getattr(hook._loop, "model", None)
    if model is None:
        return True
    try:
        traces = gather_failure_trajectories(
            hook.db, skill_name, max_trajectories=max_trajectories
        )
    except Exception:
        log.debug("replay_gate: trajectory gather failed", exc_info=True)
        return True
    if not traces:
        log.debug("replay_gate: no failure trajectories for %s — pass", skill_name)
        return True

    improved = 0
    worse = 0
    judged = 0
    for trace in traces:
        prompt = _REPLAY_PROMPT.format(
            skill_name=skill_name,
            original_body=original_body,
            new_body=new_body,
            trajectory_text=trace,
        )
        try:
            resp = await provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": _REPLAY_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=model,
                max_tokens=10,
            )
        except Exception:
            log.debug("replay_gate: provider call failed", exc_info=True)
            continue
        verdict = _parse_verdict(resp.content or "")
        if verdict is None:
            continue
        judged += 1
        if verdict == "IMPROVED":
            improved += 1
        elif verdict == "WORSE":
            worse += 1

    if judged == 0:
        log.debug("replay_gate: no parseable verdicts for %s — fail open", skill_name)
        return True
    # WORSE veto: even one WORSE verdict blocks promotion. ASG-SI is
    # "non-regressing" — refuse any visible regression.
    if worse > 0:
        log.info(
            "replay_gate: %s blocked — %d WORSE verdict(s) out of %d",
            skill_name, worse, judged,
        )
        return False
    pass_rate = improved / judged
    log.info(
        "replay_gate: %s — %d/%d IMPROVED (%.0f%% ≥ %.0f%% required)",
        skill_name, improved, judged, pass_rate * 100, min_pass_rate * 100,
    )
    return pass_rate >= min_pass_rate
