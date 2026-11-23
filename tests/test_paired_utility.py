"""Paired skill utility (skills/paired_utility.py)."""
from __future__ import annotations

import json

import numpy as np
import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.session.db import _apply_migrations
from nano_hermes.skills.paired_utility import update_paired_utility


def _hook(tmp_path, **cfg):
    return nano_hermes.install(
        _make_loop(tmp_path),
        config={"skill_stats": {"paired_utility_enabled": True,
                                "skill_search_dedup_threshold": 1.0, **cfg}},
    )


def _vec(hook, topic: int, jitter: float = 0.0) -> bytes:
    v = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
    v[topic] = 1.0
    v[topic + 1] = jitter
    return v.tobytes()


def _traj(hook, *, loaded, outcome, topic=0, jitter=0.0, vec=None):
    cur = hook.db.execute(
        "INSERT INTO trajectories (task, outcome, skills_loaded, created_at) VALUES (?, ?, ?, 0)",
        ("t", outcome, None if loaded is None else json.dumps(loaded)),
    )
    hook.db.execute(
        "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
        (cur.lastrowid, vec if vec is not None else _vec(hook, topic, jitter)),
    )


def _skill(hook, name, use_count=0, success_count=0):
    cur = hook.db.execute(
        "INSERT INTO skill_stats (name, status, use_count, success_count) VALUES (?, 'active', ?, ?)",
        (name, use_count, success_count),
    )
    hook.db.commit()
    return cur.lastrowid


def _paired(hook, name):
    return hook.db.execute(
        "SELECT paired_utility FROM skill_stats WHERE name = ?", (name,)
    ).fetchone()[0]


def test_a_skill_loaded_on_easy_tasks_earns_nothing(tmp_path):
    """Raw rate 1.0, but similar sessions without it succeed just as often."""
    hook = _hook(tmp_path)
    _skill(hook, "easy", 6, 6)
    for i in range(6):
        _traj(hook, loaded=["easy"], outcome="ok", jitter=i * 0.01)
        _traj(hook, loaded=[], outcome="ok", jitter=i * 0.01)
    hook.db.commit()
    assert update_paired_utility(hook) == 1
    assert abs(_paired(hook, "easy")) < 0.15


def test_a_skill_that_lifts_matched_sessions_is_positive(tmp_path):
    hook = _hook(tmp_path)
    _skill(hook, "helps")
    for i in range(6):
        _traj(hook, loaded=["helps"], outcome="ok", jitter=i * 0.01)
    _traj(hook, loaded=[], outcome="fail")
    # A merely related task (cosine 0.5 < paired_min_sim) where everything
    # succeeds must not be a control.
    related = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
    related[0], related[10] = 0.5, 0.866
    for _ in range(6):
        _traj(hook, loaded=[], outcome="ok", vec=related.tobytes())
    hook.db.commit()
    update_paired_utility(hook)
    assert _paired(hook, "helps") > 0.5


def test_too_few_pairs_is_null(tmp_path):
    hook = _hook(tmp_path, paired_min_pairs=5)
    _skill(hook, "rare")
    for i in range(4):
        _traj(hook, loaded=["rare"], outcome="ok", jitter=i * 0.01)
        _traj(hook, loaded=[], outcome="fail", jitter=i * 0.01)
    hook.db.commit()
    assert update_paired_utility(hook) == 0
    assert _paired(hook, "rare") is None


@pytest.mark.parametrize("loaded, outcome", [
    (None, "ok"),            # before skills_loaded was recorded
    (["s"], "unknown"),      # no evidence either way
    (["s"], "partial"),
])
def test_uninformative_rows_are_ignored(tmp_path, loaded, outcome):
    hook = _hook(tmp_path, paired_min_pairs=1)
    _skill(hook, "s")
    _traj(hook, loaded=loaded, outcome=outcome)
    _traj(hook, loaded=[], outcome="fail")
    hook.db.commit()
    update_paired_utility(hook)
    assert _paired(hook, "s") is None


def test_disabled_writes_nothing(tmp_path):
    hook = _hook(tmp_path, paired_utility_enabled=False)
    _skill(hook, "s")
    assert update_paired_utility(hook) == 0


def _rank(hook, enabled: bool) -> list[str]:
    hook.config.skill_stats.paired_utility_enabled = enabled
    hook.skill_indexer._stats_config = hook.config.skill_stats
    q = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
    q[0] = 1.0
    hits = hook.skill_indexer._vec_query(
        query_vec=q, k=2, description_by_name={"a": "a", "b": "b"}, location_by_name={},
    )
    return [h.name for h in hits]


@pytest.mark.parametrize("enabled, first", [(False, "a"), (True, "b")])
def test_ranking_uses_paired_utility_only_when_enabled(tmp_path, enabled, first):
    hook = _hook(tmp_path)
    q = np.zeros(hook.config.embedding.target_dims, dtype=np.float32)
    q[0] = 1.0
    for name, uses, wins, paired in (("a", 20, 20, -0.4), ("b", 20, 10, 0.6)):
        sid = _skill(hook, name, uses, wins)
        hook.db.execute("UPDATE skill_stats SET paired_utility = ? WHERE id = ?", (paired, sid))
        hook.db.execute(
            "INSERT INTO skill_vec (skill_id, embedding) VALUES (?, ?)", (sid, q.tobytes())
        )
    hook.db.commit()
    assert _rank(hook, enabled)[0] == first


def test_finalize_records_the_loaded_skills(tmp_path):
    hook = _hook(tmp_path)
    sid = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES ('k', 0)"
    ).lastrowid
    hook.db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', 'plan the trip', 0)", (sid,),
    )
    hook.db.commit()
    hook._session_coord.finalize(sid, {"cand", "used"}, "ok", skills_loaded={"used"})
    row = hook.db.execute("SELECT skills_used, skills_loaded FROM trajectories").fetchone()
    assert sorted(json.loads(row[0])) == ["cand", "used"] and json.loads(row[1]) == ["used"]


def test_migration_is_idempotent(tmp_path):
    hook = nano_hermes.install(_make_loop(tmp_path))
    hook.db.execute("PRAGMA user_version = 0")
    _apply_migrations(hook.db)
    _apply_migrations(hook.db)
    assert "paired_utility" in {r[1] for r in hook.db.execute("PRAGMA table_info(skill_stats)")}
    assert "skills_loaded" in {r[1] for r in hook.db.execute("PRAGMA table_info(trajectories)")}
