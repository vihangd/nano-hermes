"""Skill umbrella consolidation — merge sibling skills into one broader skill.

The library accumulates narrow, near-duplicate skills (one-session-one-skill);
search-time diversity dedup hides the symptom but the on-disk corpus is never
consolidated, so near-synonyms split the relevance mass at query time. This
fixes the corpus: cluster active, agent-authored, unpinned skills by cosine
over their stored vectors, and merge each cluster into a single umbrella skill
via one LLM call, deprecating the absorbed siblings (recoverable — files and
rows are kept, marked 'deprecated' with an "absorbed_into" audit reason).

Runs inside the evolution cycle, which already snapshots DB + skills/ for
rollback. Borrowed from hermes-agent's curator umbrella playbook. Off by
default; conservative by design (agent-origin + unpinned only, never deletes).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from .._atomic import atomic_write_text
from .curator import transition_skill
from .propose_tool import _NAME_RE

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

# Immutable, module-level — never built from stored skill content.
_MERGE_PROMPT = """You are consolidating an AI agent's skill library. The skills below are
near-duplicates that should become ONE broader "umbrella" skill covering the
whole class, with the specifics preserved as labeled subsections.

Skills to merge:
{bodies}

Produce a single merged skill. Respond with ONLY a JSON object, no prose:
{{"name": "kebab-case-umbrella-name",
  "description": "one line describing the broader class",
  "body": "the merged SKILL.md body (markdown, no frontmatter), with a
           subsection per absorbed specific",
  "absorbed": ["name1", "name2"]}}

Rules: "absorbed" must be a subset of the input names. Keep the body terse but
complete. The name may reuse one input name or be new."""


def _candidate_vectors(db) -> list[tuple[int, str, np.ndarray]]:
    rows = db.execute(
        "SELECT ss.id, ss.name, sv.embedding FROM skill_stats ss "
        "JOIN skill_vec sv ON sv.skill_id = ss.id "
        "WHERE ss.status = 'active' AND ss.origin = 'agent' AND ss.pinned = 0"
    ).fetchall()
    return [(r[0], r[1], np.frombuffer(r[2], dtype=np.float32)) for r in rows]


def find_merge_clusters(
    db, *, sim_threshold: float, min_cluster: int, max_cluster: int
) -> list[list[str]]:
    """Greedy single-pass clustering of near-duplicate sibling skills.

    Vectors are L2-normalised, so ``np.dot`` is cosine similarity. Each skill
    joins at most one cluster (first anchor wins), bounding the work.
    """
    cands = _candidate_vectors(db)
    clustered: set[str] = set()
    clusters: list[list[str]] = []
    for i, (_id_i, name_i, vec_i) in enumerate(cands):
        if name_i in clustered:
            continue
        group = [name_i]
        for _id_j, name_j, vec_j in cands[i + 1 :]:
            if name_j in clustered:
                continue
            if float(np.dot(vec_i, vec_j)) >= sim_threshold:
                group.append(name_j)
                if len(group) >= max_cluster:
                    break
        if len(group) >= min_cluster:
            clusters.append(group)
            clustered.update(group)
    return clusters


def _read_body(hook: "NanoHermesHook", name: str) -> str | None:
    path = hook.workspace / "skills" / name / "SKILL.md"
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


def _write_umbrella(hook: "NanoHermesHook", name: str, description: str, body: str) -> None:
    path = hook.workspace / "skills" / name / "SKILL.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body.strip()}\n"
    atomic_write_text(path, content)


def _parse_merge(text: str) -> dict | None:
    s = text.strip()
    start, end = s.find("{"), s.rfind("}")
    if start == -1 or end == -1:
        return None
    try:
        data = json.loads(s[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


async def _ask_merge(hook: "NanoHermesHook", bodies: dict[str, str]) -> dict | None:
    listing = "\n\n".join(f"### {n}\n{b}" for n, b in bodies.items())
    prompt = _MERGE_PROMPT.format(bodies=listing)
    try:
        resp = await hook._loop.provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=getattr(hook._loop, "model", None),
            max_tokens=2048,
        )
    except Exception:
        log.exception("umbrella: LLM call failed")
        return None
    return _parse_merge((resp.content or "").strip())


async def run_umbrella_merge(hook: "NanoHermesHook") -> list[str]:
    """Merge near-duplicate skill clusters into umbrellas. Returns umbrella
    names created. Gated by config; safe to call from the evolution cycle."""
    cfg = hook.config.skill_stats
    if not cfg.umbrella_merge_enabled:
        return []

    clusters = find_merge_clusters(
        hook.db,
        sim_threshold=cfg.umbrella_sim_threshold,
        min_cluster=cfg.umbrella_min_cluster,
        max_cluster=cfg.umbrella_max_cluster,
    )
    merged: list[str] = []
    for cluster in clusters[: cfg.umbrella_max_merges_per_run]:
        bodies = {n: b for n in cluster if (b := _read_body(hook, n))}
        if len(bodies) < cfg.umbrella_min_cluster:
            continue
        result = await _ask_merge(hook, bodies)
        if not result:
            continue
        name = str(result.get("name", "")).strip()
        description = str(result.get("description", "")).strip()
        body = str(result.get("body", "")).strip()
        absorbed = [s for s in result.get("absorbed", []) if s in bodies]
        if not (_NAME_RE.match(name) and description and body) or len(absorbed) < 2:
            log.warning("umbrella: rejecting malformed/empty merge for %s", cluster)
            continue

        # Refuse to hijack a skill outside this cluster: if the chosen name
        # already exists and isn't one of the siblings being merged, writing it
        # would overwrite that skill's file and reset its origin/pin. Only an
        # in-cluster sibling name (name in bodies) is a legitimate reuse.
        if name not in bodies:
            clash = hook.db.execute(
                "SELECT 1 FROM skill_stats WHERE name = ?", (name,)
            ).fetchone()
            if clash is not None:
                log.warning(
                    "umbrella: name %r collides with an out-of-cluster skill — skipping",
                    name,
                )
                continue

        # Write-approval gate: under "approve" stage the whole merge (umbrella
        # body + absorbed siblings) for review; don't write or deprecate yet,
        # and don't report it as merged.
        from ..governance import write_approval as wa  # noqa: PLC0415
        if wa.is_gated(hook, "skills"):
            wa.stage_umbrella_write(
                hook, name=name, description=description, body=body,
                absorbed=absorbed,
                reason=f"umbrella merge of {absorbed}",
            )
            log.info("umbrella: staged merge %s -> %s for approval (gate=approve)", absorbed, name)
            continue

        _write_umbrella(hook, name, description, body)
        # The umbrella is agent-authored so it stays eligible for future
        # evolution; the indexer embeds it on the next refresh.
        hook.db.execute(
            "INSERT INTO skill_stats (name, status, origin) VALUES (?, 'active', 'agent') "
            "ON CONFLICT(name) DO UPDATE SET status = 'active', origin = 'agent'",
            (name,),
        )
        hook.db.commit()
        for sib in absorbed:
            if sib == name:
                continue  # LLM reused a sibling name as the umbrella — keep it
            transition_skill(
                hook.db,
                sib,
                new_status="deprecated",
                reason=f"absorbed_into: {name}",
                current_body=bodies.get(sib),
            )
        merged.append(name)
        log.info("umbrella: merged %s -> %s", absorbed, name)
    return merged
