"""Paired skill utility (SkillApt arXiv 2609.26863; CompSkill 2611.01115).

``success_count / use_count`` only sees sessions that used a skill, so a skill
loaded on easy tasks looks great. SkillApt pairs runs with and without the
skill on the same state; we pair observationally instead: each session that
loaded the skill against its nearest similar sessions (task embedding) that
didn't. Weaker than replay — the agent chose when to load — so it ships off.

Background only (evolution cycle), no LLM calls: numpy over at most
``_MAX_ROWS`` trajectory vectors (2000 x 512 float32 ≈ 4 MB).
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_MAX_ROWS = 2000
_CONTROLS = 5  # nearest non-loading sessions per loading session


def update_paired_utility(hook: "NanoHermesHook") -> int:
    """Recompute ``skill_stats.paired_utility``; returns how many are set."""
    cfg = hook.config.skill_stats
    if not cfg.paired_utility_enabled:
        return 0
    db = hook.db
    # Only ok/fail say anything; rows before skills_loaded existed can't tell
    # a loaded skill from a search candidate.
    rows = db.execute(
        "SELECT t.skills_loaded, t.outcome, v.embedding FROM trajectories t "
        "JOIN trajectories_vec v ON v.trajectory_id = t.id "
        "WHERE t.skills_loaded IS NOT NULL AND t.outcome IN ('ok', 'fail') "
        "ORDER BY t.id DESC LIMIT ?",
        (_MAX_ROWS,),
    ).fetchall()
    skills = [r[0] for r in db.execute("SELECT name FROM skill_stats WHERE status != 'deprecated'")]
    values: dict[str, float | None] = dict.fromkeys(skills)
    if rows:
        loaded = [set(json.loads(r[0])) for r in rows]
        ok = np.array([r[1] == "ok" for r in rows], dtype=np.float32)
        vecs = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
        vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)
        for name in skills:
            values[name] = _paired(name, loaded, ok, vecs, cfg.paired_min_sim, cfg.paired_min_pairs)
    db.executemany(
        "UPDATE skill_stats SET paired_utility = ? WHERE name = ?",
        [(v, n) for n, v in values.items()],
    )
    db.commit()
    return sum(v is not None for v in values.values())


def _paired(
    name: str, loaded: list[set[str]], ok: np.ndarray, vecs: np.ndarray,
    min_sim: float, min_pairs: int,
) -> float | None:
    """Laplace success rate with the skill minus the similarity-weighted
    Laplace rate of matched sessions without it; None below *min_pairs*."""
    with_mask = np.array([name in s for s in loaded])
    if with_mask.sum() < min_pairs or with_mask.all():
        return None
    sims = vecs[with_mask] @ vecs.T
    sims[:, with_mask] = -1.0
    k = min(_CONTROLS, int((~with_mask).sum()))
    top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
    top_sims = np.take_along_axis(sims, top, axis=1)
    keep = top_sims >= min_sim
    paired_rows = keep.any(axis=1)
    if paired_rows.sum() < min_pairs:
        return None
    w = np.where(keep, top_sims, 0.0)[paired_rows]
    c_ok = ok[top[paired_rows]]
    control = (float((w * c_ok).sum()) + 1.0) / (float(w.sum()) + 2.0)
    treated = ok[with_mask][paired_rows]
    return (float(treated.sum()) + 1.0) / (len(treated) + 2.0) - control
