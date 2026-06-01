"""Bi-temporal fact invalidation — Zep/Graphiti cheap variant (arXiv 2501.13956).

A newly distilled semantic fact can *supersede* an existing one: "user moved
from Berlin to Lisbon", "the deploy command changed to `make ship`", "the API
key rotated". Importance and decay can't express supersession — they only
control how long a fact survives, not that a specific later fact made an
earlier one false.

When a new fact is written, we look at its nearest already-stored facts (the
embedding for the new fact is already in ``semantic_facts_vec`` after A-MEM
linking). For neighbours above a *supersession* threshold — tighter than the
A-MEM link threshold, since near-duplicates are the only plausible
contradictions — one LLM call decides which (if any) the new fact supersedes.
Those facts get ``invalid_at`` stamped; they're kept on disk for history but
filtered out of live views (``WHERE invalid_at IS NULL``).

Cost profile (Pi-friendly): zero added cost at retrieval (just a WHERE
filter); at most one LLM call per write, and only when a near-duplicate
neighbour exists — most writes make no call at all.
"""
from __future__ import annotations

import json as _json
import logging
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_DEFAULT_SUPERSEDE_THRESHOLD = 0.86
_MAX_CANDIDATES = 4

_SUPERSEDE_PROMPT = """\
A new long-term fact was just learned. Decide which (if any) of the existing \
facts it makes OUTDATED — i.e. the new fact contradicts or replaces them \
(a changed value, a moved location, a rotated key, a renamed thing). Do NOT \
flag facts that are merely related, more specific, or complementary; only \
ones the new fact makes FALSE or STALE.

NEW FACT:
{new_fact}

EXISTING FACTS:
{candidates}

Reply with ONLY a JSON array of the numbers of the existing facts the new \
fact makes outdated, e.g. [1, 3]. Reply [] if none are outdated.
"""


async def invalidate_superseded_facts(
    hook: "NanoHermesHook",
    new_fact_id: int,
    new_fact_text: str,
    *,
    enabled: bool = True,
    sim_threshold: float = _DEFAULT_SUPERSEDE_THRESHOLD,
) -> list[int]:
    """Stamp ``invalid_at`` on facts the new fact supersedes.

    Returns the list of invalidated fact ids (empty when nothing was
    superseded, the feature is disabled, no near-duplicate exists, or the
    LLM is unavailable). Never raises — supersession is best-effort and
    must not break distillation.
    """
    if not enabled:
        return []

    db = hook.db
    row = db.execute(
        "SELECT embedding FROM semantic_facts_vec WHERE fact_id = ?",
        (new_fact_id,),
    ).fetchone()
    if row is None:
        return []  # A-MEM linking didn't store a vector (embedder was down)
    vec_bytes = bytes(row[0])

    # Nearest prior facts by cosine distance, excluding the new fact itself.
    knn = db.execute(
        "SELECT fact_id, distance FROM semantic_facts_vec "
        "WHERE embedding MATCH ? AND fact_id != ? AND k = ? "
        "ORDER BY distance",
        (vec_bytes, new_fact_id, _MAX_CANDIDATES + 4),
    ).fetchall()

    candidate_ids = [
        fid for fid, dist in knn if (1.0 - float(dist)) >= sim_threshold
    ][:_MAX_CANDIDATES]
    if not candidate_ids:
        return []

    placeholders = ",".join("?" * len(candidate_ids))
    rows = db.execute(
        f"SELECT id, content FROM semantic_facts "
        f"WHERE id IN ({placeholders}) AND invalid_at IS NULL",
        candidate_ids,
    ).fetchall()
    if not rows:
        return []

    # Stable numbering: 1-based index -> fact id, in the order shown to the LLM.
    numbered = list(enumerate(rows, start=1))
    superseded = await _ask_superseded(hook, new_fact_text, numbered)
    if not superseded:
        return []

    now = time.time()
    with db:
        for fid in superseded:
            db.execute(
                "UPDATE semantic_facts SET invalid_at = ? "
                "WHERE id = ? AND invalid_at IS NULL",
                (now, fid),
            )
    log.info(
        "bitemporal: fact %d superseded %d prior fact(s): %s",
        new_fact_id,
        len(superseded),
        superseded,
    )
    return superseded


async def _ask_superseded(
    hook: "NanoHermesHook",
    new_fact_text: str,
    numbered: list[tuple[int, tuple]],
) -> list[int]:
    """One LLM call: which numbered candidates does the new fact supersede?

    Returns the list of fact ids (not display numbers). Empty on any failure.
    """
    provider = getattr(hook._loop, "provider", None)  # noqa: SLF001
    model = getattr(hook._loop, "model", None)  # noqa: SLF001
    if provider is None or model is None:
        return []

    candidates_block = "\n".join(
        f"{n}. {content}" for n, (_fid, content) in numbered
    )
    prompt = _SUPERSEDE_PROMPT.format(
        new_fact=new_fact_text, candidates=candidates_block
    )
    try:
        resp = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=60,
        )
        content = (resp.content or "").strip()
    except Exception:
        return []
    if not content:
        return []

    nums = _parse_index_list(content)
    num_to_id = {n: fid for n, (fid, _content) in numbered}
    return [num_to_id[n] for n in nums if n in num_to_id]


def _parse_index_list(text: str) -> list[int]:
    """Extract a JSON array of ints from an LLM reply, tolerating fences/prose."""
    payload = text
    if "[" in payload and "]" in payload:
        payload = payload[payload.index("[") : payload.rindex("]") + 1]
    try:
        data = _json.loads(payload)
    except (ValueError, TypeError):
        return []
    if not isinstance(data, list):
        return []
    out: list[int] = []
    for x in data:
        try:
            out.append(int(x))
        except (TypeError, ValueError):
            continue
    return out
