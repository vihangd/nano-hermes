"""Tests for MemSkill designer loop (skill_designer.py)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.skill_designer import _cluster_sessions, run_skill_designer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hook(tmp_path, extra_skill_stats=None):
    cfg = {"skill_stats": {"skill_designer_enabled": True, **(extra_skill_stats or {})}}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _vec(seed: int = 1) -> bytes:
    """512-dim unit vector seeded by first component value."""
    v = np.zeros(512, dtype=np.float32)
    v[seed % 512] = 1.0
    return v.tobytes()


def _seed_trajectory(hook, *, task="do something", outcome="fail", skills_used="[]",
                     days_ago=1.0, embedding: bytes | None = None):
    """Insert a trajectory row and optionally a chunk + chunk embedding."""
    ts = time.time() - days_ago * 86400
    cur = hook.db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (f"s:{ts}", ts),
    )
    sid = cur.lastrowid
    hook.db.execute(
        "INSERT INTO trajectories (session_id, task, skills_used, outcome, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (sid, task, skills_used, outcome, ts),
    )
    if embedding is not None:
        cur2 = hook.db.execute(
            "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
            "VALUES (?, 0, 'user', ?, ?)",
            (sid, task, ts),
        )
        cid = cur2.lastrowid
        hook.db.execute(
            "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
            (cid, embedding),
        )
    hook.db.commit()
    return sid


def _make_resp(content: str):
    r = MagicMock()
    r.finish_reason = "stop"
    r.content = content
    return r


VALID_SKILL_MD = """\
---
name: new_coverage_skill
description: handles coverage gaps
---

## When to Use
When the agent has no skill for this task type.

## Steps
1. Identify the task
2. Execute it
"""


# ---------------------------------------------------------------------------
# Unit: cosine + clustering
# ---------------------------------------------------------------------------

class TestGreedyCluster:
    def test_similar_sessions_grouped(self):
        emb = _vec(0)
        sessions = [
            (1, "task a", emb),
            (2, "task b", emb),
            (3, "task c", emb),
        ]
        clusters = _cluster_sessions(sessions, threshold=0.75)
        assert len(clusters) == 1
        assert len(clusters[0]) == 3

    def test_orthogonal_sessions_separated(self):
        e1 = _vec(0)
        e2 = _vec(1)
        sessions = [(1, "t1", e1), (2, "t2", e2)]
        clusters = _cluster_sessions(sessions, threshold=0.75)
        assert len(clusters) == 2

    def test_none_embedding_skipped(self):
        e = _vec(0)
        sessions = [(1, "t1", None), (2, "t2", e)]
        clusters = _cluster_sessions(sessions, threshold=0.75)
        # Only session 2 has embedding — forms solo cluster
        assert any(len(c) == 1 for c in clusters)
        # Session 0 (None emb) never starts a cluster
        assert all(0 not in c for c in clusters)


# ---------------------------------------------------------------------------
# Integration: run_skill_designer
# ---------------------------------------------------------------------------

class TestRunSkillDesignerDisabled:
    def test_disabled_by_default(self, tmp_path):
        hook = nano_hermes.install(_make_loop(tmp_path), config={})
        result = asyncio.run(run_skill_designer(hook))
        assert result == []

    def test_enabled_false_explicit(self, tmp_path):
        hook = nano_hermes.install(
            _make_loop(tmp_path),
            config={"skill_stats": {"skill_designer_enabled": False}},
        )
        result = asyncio.run(run_skill_designer(hook))
        assert result == []


class TestRunSkillDesignerNoFailures:
    def test_returns_empty_when_no_failures(self, tmp_path):
        hook = _hook(tmp_path)
        result = asyncio.run(run_skill_designer(hook))
        assert result == []

    def test_returns_empty_when_only_success(self, tmp_path):
        hook = _hook(tmp_path)
        _seed_trajectory(hook, outcome="success", skills_used="[]",
                         embedding=_vec(0))
        result = asyncio.run(run_skill_designer(hook))
        assert result == []

    def test_returns_empty_when_skills_were_used(self, tmp_path):
        hook = _hook(tmp_path)
        _seed_trajectory(hook, outcome="fail", skills_used='["some_skill"]',
                         embedding=_vec(0))
        result = asyncio.run(run_skill_designer(hook))
        assert result == []


class TestRunSkillDesignerClusterTooSmall:
    def test_cluster_below_min_size_skipped(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 3})
        e = _vec(0)
        _seed_trajectory(hook, task="task A", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task B", outcome="fail", embedding=e)
        # Only 2 sessions — below min_cluster_size=3
        result = asyncio.run(run_skill_designer(hook))
        assert result == []


class TestRunSkillDesignerProposal:
    def test_proposes_skill_for_eligible_cluster(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 2})
        e = _vec(0)
        _seed_trajectory(hook, task="task A", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task B", outcome="fail", embedding=e)

        mock_resp = _make_resp(VALID_SKILL_MD)
        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ):
            result = asyncio.run(run_skill_designer(hook))

        assert "new_coverage_skill" in result
        # Skill directory should exist
        assert (hook.workspace / "skills" / "new_coverage_skill" / "SKILL.md").exists()

    def test_skips_on_unparseable_name(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 2})
        e = _vec(0)
        _seed_trajectory(hook, task="task A", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task B", outcome="fail", embedding=e)

        bad_body = "No name header here at all."
        mock_resp = _make_resp(bad_body)
        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ):
            result = asyncio.run(run_skill_designer(hook))

        assert result == []

    def test_security_scan_blocks_malicious_content(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 2})
        e = _vec(0)
        _seed_trajectory(hook, task="task A", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task B", outcome="fail", embedding=e)

        mock_resp = _make_resp(VALID_SKILL_MD)
        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ), patch(
            "nano_hermes.skills.skill_designer.scan_skill_content",
            return_value="blocked: malicious",
        ):
            result = asyncio.run(run_skill_designer(hook))

        assert result == []

    def test_no_embedding_sessions_excluded_from_clustering(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 3})
        # 2 sessions with embeddings, 2 without — cluster of 2 < min 3
        e = _vec(0)
        _seed_trajectory(hook, task="task A", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task B", outcome="fail", embedding=e)
        _seed_trajectory(hook, task="task C", outcome="fail", embedding=None)
        _seed_trajectory(hook, task="task D", outcome="fail", embedding=None)

        result = asyncio.run(run_skill_designer(hook))
        assert result == []

    def test_partial_outcome_included(self, tmp_path):
        hook = _hook(tmp_path, {"skill_designer_min_cluster_size": 2})
        e = _vec(0)
        _seed_trajectory(hook, task="partial task A", outcome="partial", embedding=e)
        _seed_trajectory(hook, task="partial task B", outcome="partial", embedding=e)

        mock_resp = _make_resp(VALID_SKILL_MD)
        with patch.object(
            hook._loop.provider, "chat_with_retry", AsyncMock(return_value=mock_resp)
        ):
            result = asyncio.run(run_skill_designer(hook))

        assert "new_coverage_skill" in result
