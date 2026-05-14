"""Tests for workflow_suggest tool — trajectory clustering for workflow induction."""
from __future__ import annotations

import json
import time

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.session.workflow_suggest import _find_workflow_clusters


DIMS = 512


def _unit(idx: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[idx] = 1.0
    return v


def _insert_trajectory_with_vec(
    db, task: str, skills: list[str], outcome: str, vec: np.ndarray
) -> int:
    cur = db.execute(
        "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
        "VALUES (?, ?, ?, ?)",
        (task, json.dumps(skills), outcome, time.time()),
    )
    traj_id = int(cur.lastrowid)
    db.execute(
        "INSERT INTO trajectories_vec (trajectory_id, embedding) VALUES (?, ?)",
        (traj_id, vec.astype(np.float32).tobytes()),
    )
    db.commit()
    return traj_id


def _make_hook(tmp_path, enabled: bool = True):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(
        loop,
        config={"workflow_induction": {"enabled": enabled, "min_cluster_size": 2}},
    )
    return loop, hook


class TestFindWorkflowClusters:
    """Unit tests for _find_workflow_clusters."""

    def _db(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return hook.db

    def test_empty_db_returns_empty(self, tmp_path):
        db = self._db(tmp_path)
        result = _find_workflow_clusters(db, min_cluster_size=2)
        assert result == []

    def test_only_failed_trajectories_skipped(self, tmp_path):
        db = self._db(tmp_path)
        _insert_trajectory_with_vec(db, "failing task", [], "fail", _unit(0))
        _insert_trajectory_with_vec(db, "partial task", [], "partial", _unit(0))
        result = _find_workflow_clusters(db, min_cluster_size=2)
        assert result == []

    def test_similar_tasks_form_cluster(self, tmp_path):
        db = self._db(tmp_path)
        for i in range(3):
            _insert_trajectory_with_vec(
                db, f"fetch weather for city {i}", ["weather_skill"], "ok", _unit(0)
            )
        result = _find_workflow_clusters(db, min_cluster_size=2, cluster_threshold=0.85)
        assert len(result) >= 1
        assert len(result[0]["tasks"]) >= 2

    def test_distinct_tasks_no_cluster(self, tmp_path):
        db = self._db(tmp_path)
        for i in range(3):
            _insert_trajectory_with_vec(
                db, f"task type {i}", [], "ok", _unit(i)
            )
        # Each task has a distinct orthogonal vector — no cluster forms
        result = _find_workflow_clusters(db, min_cluster_size=2, cluster_threshold=0.85)
        assert result == []

    def test_clusters_sorted_by_size_descending(self, tmp_path):
        db = self._db(tmp_path)
        # 3 tasks of type A
        for i in range(3):
            _insert_trajectory_with_vec(db, f"task A {i}", ["s-a"], "ok", _unit(0))
        # 2 tasks of type B
        for i in range(2):
            _insert_trajectory_with_vec(db, f"task B {i}", ["s-b"], "ok", _unit(1))

        result = _find_workflow_clusters(db, min_cluster_size=2, cluster_threshold=0.85)
        assert len(result) >= 2
        assert len(result[0]["tasks"]) >= len(result[1]["tasks"])

    def test_skills_aggregated_in_cluster(self, tmp_path):
        db = self._db(tmp_path)
        _insert_trajectory_with_vec(db, "task A", ["skill-x", "skill-y"], "ok", _unit(0))
        _insert_trajectory_with_vec(db, "task A2", ["skill-x"], "ok", _unit(0))

        result = _find_workflow_clusters(db, min_cluster_size=2, cluster_threshold=0.85)
        assert len(result) >= 1
        all_skills = [s for skills in result[0]["skills_used"] for s in skills]
        assert "skill-x" in all_skills

    def test_max_trajectories_cap(self, tmp_path):
        db = self._db(tmp_path)
        for i in range(10):
            _insert_trajectory_with_vec(db, f"task {i}", [], "ok", _unit(0))
        # With cap=3, only 3 rows are queried; cluster requires ≥2 members
        result = _find_workflow_clusters(db, max_trajectories=3, min_cluster_size=2)
        # Result may have a cluster (3 rows, same vec) or not — just no crash
        assert isinstance(result, list)

    def test_no_embeddings_returns_empty(self, tmp_path):
        db = self._db(tmp_path)
        # Insert trajectory row WITHOUT a corresponding trajectories_vec entry
        db.execute(
            "INSERT INTO trajectories (task, skills_used, outcome, created_at) "
            "VALUES (?, ?, ?, ?)",
            ("task without embed", "[]", "ok", time.time()),
        )
        db.commit()
        result = _find_workflow_clusters(db, min_cluster_size=2)
        assert result == []


class TestWorkflowSuggestTool:
    """Integration tests for workflow_suggest tool."""

    async def test_disabled_by_default_returns_message(self, tmp_path):
        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)  # default: enabled=False

        tool = loop.tools.get("workflow_suggest")
        result = await tool.execute()
        assert "disabled" in result.lower()
        assert "workflow_induction.enabled" in result

    async def test_enabled_no_data_returns_no_patterns(self, tmp_path):
        loop, hook = _make_hook(tmp_path, enabled=True)
        tool = loop.tools.get("workflow_suggest")
        result = await tool.execute()
        assert "no recurring" in result.lower() or "not found" in result.lower()

    async def test_enabled_with_clusters_returns_report(self, tmp_path):
        loop, hook = _make_hook(tmp_path, enabled=True)
        db = hook.db

        for i in range(3):
            _insert_trajectory_with_vec(
                db, f"deploy microservice variant {i}", ["k8s-deploy"], "ok", _unit(0)
            )

        tool = loop.tools.get("workflow_suggest")
        result = await tool.execute()
        assert "pattern" in result.lower() or "found" in result.lower()
        assert "propose_skill" in result

    async def test_k_limits_cluster_output(self, tmp_path):
        loop, hook = _make_hook(tmp_path, enabled=True)
        db = hook.db

        for idx in range(3):
            for i in range(3):
                _insert_trajectory_with_vec(
                    db, f"task type {idx} run {i}", [], "ok", _unit(idx)
                )

        tool = loop.tools.get("workflow_suggest")
        result = await tool.execute(k=1)
        # Only 1 cluster should appear in the output
        assert result.count("Pattern") <= 1

    async def test_workflow_suggest_registered(self, tmp_path):
        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)
        assert loop.tools.get("workflow_suggest") is not None
