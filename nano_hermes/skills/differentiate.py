"""Skill++ description differentiation (arXiv 2611.00544).

As the library grows, similar-but-distinct skills steal each other's routing.
For each agent-authored skill whose nearest neighbour (cosine over the stored
skill vectors) is close, one LLM call rewrites its description to state what
its body supports, how it differs from that neighbour, and a closing
"Not for <case>." the skip gate (indexer.py) already reads. Skill++ reports
FC@5 36.4 → 76.3 at 731 skills; differentiation is its largest ablation.

Background only (evolution cycle); default off. No renames: a rename would
orphan the skill's stats and directory.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from ..governance import write_approval as wa
from ..utils.error_classifier import (
    EvolutionAbortError,
    FailoverReason,
    classify_llm_response,
)
from .external import _FRONTMATTER_RE
from .indexer import _split_skip, description_digest

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# Immutable, module-level — never built from stored skill content.
_PROMPT = """\
Two agent skills are easily confused. Rewrite the FIRST skill's description so a
router picks the right one.

FIRST SKILL: {name}
Current description: {description}
Body:
---
{body}
---

NEIGHBOUR SKILL: {neighbour}
Its description: {neighbour_description}

Write ONE line, under 300 characters: what the first skill does, judged by its
body; how it differs from {neighbour}; then end with the sentence
"Not for <the cases {neighbour} covers>." Keep any exclusion the current
description already has. Use no colons and no quotes. Output only the line.
"""

def _valid(description: str) -> bool:
    """One line, no double quotes, backslashes or control characters, no
    edge apostrophe: render_skill quotes it when YAML would misread it, and
    the indexer's regex read (which strips edge quotes and never unescapes)
    then gives back the same text."""
    return (
        bool(description) and len(description) <= 400
        and description.isprintable()
        and not any(c in description for c in "\"\\")
        and "'" not in (description[0], description[-1])
    )


def _candidates(hook: "NanoHermesHook") -> tuple[list[tuple], list[tuple]]:
    rows = hook.db.execute(
        "SELECT ss.name, ss.content_hash, ss.differentiated_hash, ss.origin, ss.pinned, "
        "ss.status, sv.embedding FROM skill_stats ss "
        "JOIN skill_vec sv ON sv.skill_id = ss.id WHERE ss.status != 'deprecated'"
    ).fetchall()
    # A skill with an open or declined differentiation is left alone.
    held = {
        r[0] for r in hook.db.execute(
            "SELECT skill_name FROM pending_writes WHERE origin = 'differentiate' "
            "AND status IN ('pending', 'rejected')"
        )
    }
    todo = [
        r for r in rows
        if r[3] == "agent" and not r[4] and r[1] is not None
        and r[2] != r[1] and r[0] not in held
    ]
    return todo, rows


def _read(hook: "NanoHermesHook", name: str) -> tuple[str, str] | None:
    """``(description, body)`` of a workspace skill, or None."""
    path = hook.workspace / "skills" / name / "SKILL.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return None
    # The indexer's read: handles folded/block scalars the regex parse misses.
    description = hook.skill_indexer._description_for({"name": name, "path": str(path)})
    return description, text[m.end():].strip()


def _mark(hook: "NanoHermesHook", name: str, digest: str) -> None:
    hook.db.execute(
        "UPDATE skill_stats SET differentiated_hash = ? WHERE name = ?", (digest, name)
    )
    hook.db.commit()


async def run_differentiation(hook: "NanoHermesHook") -> list[str]:
    """Differentiate up to ``differentiation_max_per_cycle`` skills. Returns
    the names rewritten (or staged for approval)."""
    cfg = hook.config.skill_stats
    if not cfg.differentiation_enabled:
        return []
    todo, rows = _candidates(hook)
    if not todo or len(rows) < 2:
        return []

    names = [r[0] for r in rows]
    vecs = np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
    index = {n: i for i, n in enumerate(names)}
    descriptions = hook.skill_indexer._descriptions(hook.skill_indexer._list_entries())

    from .propose_tool import ProposeSkillTool  # noqa: PLC0415
    tool = ProposeSkillTool(hook=hook)
    done: list[str] = []
    calls = 0
    for name, content_hash, *_ in todo:
        if calls >= cfg.differentiation_max_per_cycle:
            break
        i = index[name]
        sims = vecs @ vecs[i]
        sims[i] = -1.0
        j = int(np.argmax(sims))
        if sims[j] < cfg.differentiation_min_sim:
            _mark(hook, name, content_hash)  # nothing close enough to confuse
            continue
        current = _read(hook, name)
        if current is None:
            continue
        description, body = current
        neighbour = names[j]
        calls += 1
        try:
            resp = await hook._loop.provider.chat_with_retry(
                messages=[{"role": "user", "content": _PROMPT.format(
                    name=name, description=description, body=body[:2000],
                    neighbour=neighbour,
                    neighbour_description=descriptions.get(neighbour, ""),
                )}],
                model=hook.background_model,
                max_tokens=160,
            )
        except Exception:
            # Provider down: retry next cycle, don't spend the rest now.
            log.warning("differentiate: LLM call failed for %s", name, exc_info=True)
            break
        err = classify_llm_response(resp)
        if err is not None:
            if err.should_abort:
                raise EvolutionAbortError(err)
            if err.reason not in (FailoverReason.content_policy, FailoverReason.context_overflow):
                # Outage or rate limit: retry next cycle, unmarked.
                log.warning("differentiate: %s for %s", err.reason.value, name)
                break
        new = None if err else (resp.content or "").strip()
        if not new or not _valid(new) or not _split_skip(new)[1]:
            # Marked done, else the same few skills take every cycle's calls.
            log.info("differentiate: rejected reply for %s: %r", name, (new or "")[:120])
            _mark(hook, name, content_hash)
            continue
        if new == description:
            _mark(hook, name, content_hash)
            continue
        if wa.is_gated(hook, "skills"):
            wa.stage_skill_write(
                hook, skill_name=name, description=new, body=body,
                reason=f"differentiate from {neighbour}", origin="differentiate",
            )
            # Approved and re-indexed, the skill's hash becomes this digest.
            _mark(hook, name, description_digest(name, new))
            done.append(name)
            continue
        result = await tool._edit(
            skill_name=name, description=new, body=body, files=[], delete_files=[],
            redaction_note=" [differentiated]",
        )
        if result.startswith("Error"):
            log.warning("differentiate: edit failed for %s — %s", name, result)
            _mark(hook, name, content_hash)
            continue
        # The indexer stores this digest once it re-embeds the file as written.
        written = _read(hook, name)
        _mark(hook, name, description_digest(name, written[0] if written else new))
        log.info("differentiate: %s now distinct from %s", name, neighbour)
        done.append(name)
    return done
