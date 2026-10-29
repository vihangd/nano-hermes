"""Background judge for session outcome labels (config ``outcome_judge``).

Runs after a session ends — never on the reply path — when the rule label
(``session/outcome.py``) isn't already a hard failure. Asks one question of
the ended session: did the assistant accomplish what the user asked?

- **jev** (primary): TypeSafe's System One decision model. Two yes/no
  ("noul") questions — accomplished, and its negation — return calibrated-ish
  probabilities. A verdict needs one side at or above ``min_confidence`` *and*
  the other at or below ``1 - min_confidence``, so the pair must agree (an
  independent audit found a noul and its negation summing anywhere from 0.71
  to 1.42 — a disagreeing pair is noise, not a verdict).
- **llm** (fallback): the configured background model answers
  SUCCESS / FAIL / UNCLEAR.

In ``shadow`` mode the verdict is only recorded in ``outcome_signals``; in
``apply`` mode it becomes the stored outcome (``outcome_source='judge:…'``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import TYPE_CHECKING, Any

import httpx

from ..redact import redact
from ..session.archiver import _extract_text
from ..utils.error_classifier import classify_llm_response
from .budgets import _count_tokens

if TYPE_CHECKING:
    from ..config import OutcomeJudgeConfig
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})
_RETRIES = 2

_ACCOMPLISHED = {
    "type": "noul",
    "instructions": "The assistant accomplished what the user asked for in this session.",
    "criteria": {
        "true": "The request was carried out and the user's last reaction, if any, accepts it.",
        "false": "The request was not carried out, was carried out wrongly, or was abandoned.",
    },
}
_FAILED = {
    "type": "noul",
    "instructions": "The assistant failed or left the user's request unresolved in this session.",
    "criteria": {
        "true": "The request was not carried out, was carried out wrongly, or was abandoned.",
        "false": "The request was carried out and the user's last reaction, if any, accepts it.",
    },
}

_LLM_PROMPT = """\
Below is the end of a conversation between a user and an AI assistant.
Did the assistant accomplish what the user asked for?

Answer with exactly one word:
SUCCESS — the request was carried out and nothing indicates the user was unhappy
FAIL — the request was not carried out, was done wrongly, or the user pushed back unresolved
UNCLEAR — you cannot tell with confidence (the default when unsure)

Task: {task}

Conversation (oldest first, may be truncated):
{transcript}"""


def _fit_transcript(messages: list[dict[str, Any]], max_tokens: int) -> tuple[str, str]:
    """(task, transcript tail) redacted and fitted under *max_tokens*.

    Drops the oldest messages first: the outcome lives at the end.
    """
    lines = []
    for m in messages:
        text = _extract_text(m).strip()
        if text and m.get("role") in ("user", "assistant"):
            lines.append(f"{m['role'].upper()}: {redact(text).text}")
    task = next(
        (redact(_extract_text(m)).text[:500] for m in messages if m.get("role") == "user"), ""
    )
    budget = max_tokens - _count_tokens(task) - 200  # questions + JSON overhead
    kept: list[str] = []
    used = 0
    for line in reversed(lines):
        cost = _count_tokens(line)
        if used + cost > budget:
            if not kept:  # a single huge last message: keep its tail
                kept.append(line[-budget * 3:])
            break
        kept.append(line)
        used += cost
    return task, "\n\n".join(reversed(kept))


async def _post_jev(cfg: "OutcomeJudgeConfig", key: str, body: dict) -> dict | None:
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    async with httpx.AsyncClient(timeout=cfg.timeout_s) as client:
        for attempt in range(_RETRIES + 1):
            try:
                resp = await client.post(cfg.jev_url, headers=headers, json=body)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                log.debug("outcome judge: jev transport error %s", e)
            else:
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:
                        return None
                    return data if isinstance(data, dict) else None
                if resp.status_code not in _RETRY_STATUS:
                    log.info("outcome judge: jev HTTP %d: %s", resp.status_code, resp.text[:200])
                    return None
                retry_after = resp.headers.get("Retry-After")
                if retry_after and retry_after.isdigit() and attempt < _RETRIES:
                    await asyncio.sleep(min(float(retry_after), 30.0))
                    continue
            if attempt < _RETRIES:
                await asyncio.sleep(0.5 * 2 ** attempt)
    return None


def _jev_verdict(answers: dict, min_conf: float) -> tuple[str | None, dict]:
    try:
        p_ok = float(answers["accomplished"]["noul"])
        p_fail = float(answers["failed"]["noul"])
    except (KeyError, TypeError, ValueError):
        return None, {}
    detail = {"p_accomplished": p_ok, "p_failed": p_fail}
    if p_ok >= min_conf and p_fail <= 1.0 - min_conf:
        return "ok", detail
    if p_fail >= min_conf and p_ok <= 1.0 - min_conf:
        return "fail", detail
    return None, detail


async def _judge_jev(cfg: "OutcomeJudgeConfig", task: str, transcript: str) -> dict | None:
    """Jev verdict dict, or None when the backend is unavailable / failed."""
    key = os.environ.get(cfg.jev_key_env, "").strip()
    if not key:
        log.info("outcome judge: %s not set — jev backend unavailable", cfg.jev_key_env)
        return None
    body = {
        "model": cfg.jev_model,
        "state": {"task": task, "transcript_tail": transcript},
        "questions": {"accomplished": _ACCOMPLISHED, "failed": _FAILED},
    }
    data = await _post_jev(cfg, key, body)
    if not data:
        return None
    verdict, detail = _jev_verdict(data.get("answers") or {}, cfg.min_confidence)
    return {"backend": "jev", "model": data.get("model", cfg.jev_model), "verdict": verdict, **detail}


async def _judge_llm(hook: "NanoHermesHook", task: str, transcript: str) -> dict | None:
    try:
        resp = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": _LLM_PROMPT.format(task=task, transcript=transcript)}],
            model=hook.background_model,
            max_tokens=5,
        )
    except Exception:
        log.debug("outcome judge: llm call failed", exc_info=True)
        return None
    if classify_llm_response(resp) is not None:
        return None
    word = (resp.content or "").strip().split()[:1]
    answer = word[0].strip(".:").upper() if word else ""
    verdict = {"SUCCESS": "ok", "FAIL": "fail"}.get(answer)
    return {"backend": "llm", "model": hook.background_model, "verdict": verdict}


async def judge(hook: "NanoHermesHook", messages: list[dict[str, Any]]) -> dict | None:
    """Run the configured backend (then the fallback); None if neither answered."""
    cfg = hook.config.outcome_judge
    task, transcript = _fit_transcript(messages, cfg.max_state_tokens)
    if not transcript:
        return None
    result = await _judge_jev(cfg, task, transcript) if cfg.backend == "jev" else None
    if result is None and (cfg.backend == "llm" or cfg.fallback == "llm"):
        result = await _judge_llm(hook, task, transcript)
    return result


def combine(rule: str, verdict: str | None) -> str:
    """A confident verdict decides; otherwise the rule label stands.

    Hard rule failures never reach the judge (``is_hard_failure``).
    """
    return verdict or rule


def record(
    hook: "NanoHermesHook", session_id: int, rule: str, result: dict
) -> str:
    """Store *result* on the session's trajectory; return the effective outcome."""
    cfg = hook.config.outcome_judge
    row = hook.db.execute(
        "SELECT id, outcome_signals FROM trajectories WHERE session_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    if row is None:
        return rule
    tid, raw = row
    signals = json.loads(raw) if raw else {}
    signals["judge"] = result
    outcome = combine(rule, result.get("verdict")) if cfg.mode == "apply" else rule
    if outcome != rule:
        hook.db.execute(
            "UPDATE trajectories SET outcome = ?, outcome_source = ?, outcome_signals = ? "
            "WHERE id = ?",
            (outcome, f"judge:{result['backend']}", json.dumps(signals), tid),
        )
    else:
        hook.db.execute(
            "UPDATE trajectories SET outcome_signals = ? WHERE id = ?",
            (json.dumps(signals), tid),
        )
    hook.db.commit()
    return outcome
