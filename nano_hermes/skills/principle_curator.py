"""ACE principle curator — automatic delta evolution of the principles playbook.

Session-boundary, one hosted LLM call per run (gated + cooldown). Turns recent
session failures into add / update / prune *deltas* over the `principles` store,
applied by a deterministic, non-LLM, by-id merge. The LLM never re-emits the
whole playbook — that's exactly the ACE context-collapse failure mode
(arXiv 2510.04618, where an end-to-end rewrite dropped 18,282 tokens to 122 and
accuracy below the no-adapt baseline). Embedding dedup keeps adds from piling
up; counter-driven pruning bounds growth.
"""
from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING

from .principle_index import upsert_principle

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_META_LAST_RUN = "principle_curator.last_run_at"

# Immutable, module-level — never built from stored content, so a malicious
# principle/trajectory can't inject instructions into the curator prompt.
_CURATOR_PROMPT = """You maintain a small "playbook" of operating principles for an AI agent.
Each principle is: WHEN <condition> THEN <action> (SO THAT <expected_outcome>).

Recent FAILED or PARTIAL sessions:
{failures}

Current playbook (id | helpful-harmful | condition -> action):
{principles}

Propose a SMALL set of delta operations that would help avoid the failures.
Rules:
- Prefer editing/adding a few high-value principles over churn.
- "prune" only principles that are wrong or chronically unhelpful, by id.
- Keep condition/action terse and general (not task-specific trivia).

Respond with ONLY a JSON object, no prose:
{{"ops": [
  {{"op": "add", "condition": "...", "action": "...", "expected_outcome": "..."}},
  {{"op": "update", "id": 12, "condition": "...", "action": "...", "expected_outcome": "..."}},
  {{"op": "prune", "id": 7}}
]}}"""


def _should_run(db, cooldown_hours: int, *, now: float | None = None) -> bool:
    if cooldown_hours <= 0:
        return True
    now_ts = now if now is not None else time.time()
    row = db.execute("SELECT value FROM meta WHERE key = ?", (_META_LAST_RUN,)).fetchone()
    if not row:
        return True
    try:
        return (now_ts - float(row[0])) >= cooldown_hours * 3600
    except ValueError:
        return True


def _mark_run(db, *, now: float | None = None) -> None:
    db.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (_META_LAST_RUN, str(now if now is not None else time.time())),
    )
    db.commit()


def _recent_failures(db, limit: int) -> list[str]:
    rows = db.execute(
        "SELECT task, reflection FROM trajectories "
        "WHERE outcome IN ('fail', 'partial') ORDER BY created_at DESC LIMIT ?",
        (limit,),
    ).fetchall()
    out = []
    for task, reflection in rows:
        line = f"- {task}"
        if reflection:
            line += f"\n  (reflection: {reflection})"
        out.append(line)
    return out


def _current_principles(db, limit: int = 50) -> list[tuple]:
    return db.execute(
        "SELECT id, condition, action, success_count, harmful_count FROM principles "
        "ORDER BY (success_count - harmful_count) DESC LIMIT ?",
        (limit,),
    ).fetchall()


def parse_ops(text: str) -> list[dict]:
    """Tolerantly extract the ops list from an LLM response. [] on any failure
    (a malformed response must be a safe no-op, never a crash)."""
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 2)[1] if s.count("```") >= 2 else s.strip("`")
        if s.startswith("json"):
            s = s[4:]
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return []
    try:
        data = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    ops = data.get("ops") if isinstance(data, dict) else None
    if not isinstance(ops, list):
        return []
    return [o for o in ops if isinstance(o, dict) and o.get("op") in ("add", "update", "prune")]


async def apply_ops(hook: "NanoHermesHook", ops: list[dict], cfg) -> dict[str, int]:
    """Deterministic by-id merge of curator deltas. Never rewrites the table."""
    counts = {"added": 0, "merged": 0, "updated": 0, "pruned": 0, "skipped": 0}
    for op in ops[: cfg.max_ops_per_run]:
        kind = op["op"]
        try:
            if kind == "add":
                cond, act = op.get("condition"), op.get("action")
                if not cond or not act:
                    continue
                _pid, outcome = await upsert_principle(
                    hook,
                    condition=cond,
                    action=act,
                    expected_outcome=op.get("expected_outcome") or None,
                    origin="curator",
                    dedup_threshold=cfg.dedup_threshold,
                    protect_manual=True,
                )
                counts[{"inserted": "added", "inserted_no_vec": "added",
                        "merged": "merged", "skipped": "skipped"}[outcome]] += 1
            elif kind == "update":
                pid = op.get("id")
                # Only curator-authored, unpinned rows — never overwrite a human.
                cur = hook.db.execute(
                    "UPDATE principles SET action = COALESCE(?, action), "
                    "condition = COALESCE(?, condition), "
                    "expected_outcome = COALESCE(?, expected_outcome), updated_at = ? "
                    "WHERE id = ? AND origin = 'curator' AND pinned = 0",
                    (op.get("action"), op.get("condition"),
                     op.get("expected_outcome"), time.time(), pid),
                )
                hook.db.commit()
                counts["updated"] += cur.rowcount
            elif kind == "prune":
                cur = hook.db.execute(
                    "DELETE FROM principles WHERE id = ? AND origin = 'curator' AND pinned = 0",
                    (op.get("id"),),
                )
                hook.db.commit()
                counts["pruned"] += cur.rowcount
        except Exception:
            log.exception("principle curator: op failed: %s", op)
    return counts


def _prune_over_budget(db, max_principles: int) -> int:
    """Drop lowest-value curator-authored principles when over the cap."""
    total = db.execute("SELECT COUNT(*) FROM principles").fetchone()[0]
    over = total - max_principles
    if over <= 0:
        return 0
    cur = db.execute(
        "DELETE FROM principles WHERE id IN ("
        "  SELECT id FROM principles WHERE origin = 'curator' AND pinned = 0 "
        "  ORDER BY (success_count - harmful_count) ASC, updated_at ASC LIMIT ?)",
        (over,),
    )
    db.commit()
    return cur.rowcount


async def run_principle_curator(hook: "NanoHermesHook") -> dict[str, int]:
    """One curation pass. Returns a counts dict (empty when skipped)."""
    cfg = hook.config.principles
    if not cfg.enabled:
        return {}
    if not _should_run(hook.db, cfg.cooldown_hours):
        log.debug("principle curator: cooldown — skipping")
        return {}

    failures = _recent_failures(hook.db, limit=8)
    if not failures:
        _mark_run(hook.db)
        return {}

    principles = _current_principles(hook.db)
    principle_lines = "\n".join(
        f"{r[0]} | {r[3] - r[4]:+d} | {r[1]} -> {r[2]}" for r in principles
    ) or "(empty)"
    prompt = _CURATOR_PROMPT.format(
        failures="\n".join(failures), principles=principle_lines
    )

    try:
        resp = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(hook._loop, "model", None),
            max_tokens=1024,
        )
        text = (resp.content or "").strip()
    except Exception:
        log.exception("principle curator: LLM call failed")
        return {}

    ops = parse_ops(text)

    # Write-approval gate: under "approve" stage the op list for review instead
    # of applying it. Replay re-runs apply_ops against the live table, so its
    # deterministic dedup/prune re-merge correctly — no stale-base check needed.
    from ..governance import write_approval as wa  # noqa: PLC0415
    if ops and wa.is_gated(hook, "principles"):
        wa.stage_principle_ops(hook, ops=ops, reason=f"{len(ops)} curator op(s)")
        _mark_run(hook.db)
        log.info("principle curator: staged %d op(s) for approval (gate=approve)", len(ops))
        return {"staged": len(ops)}

    counts = await apply_ops(hook, ops, cfg)
    counts["pruned"] += _prune_over_budget(hook.db, cfg.max_principles)
    _mark_run(hook.db)
    if any(counts.values()):
        log.info("principle curator: %s", counts)
    return counts
