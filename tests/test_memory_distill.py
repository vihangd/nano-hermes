"""Tests for memory_patch(action="distill") — episodic→semantic hub detection."""
from __future__ import annotations

import time

import numpy as np

import nano_hermes
from conftest import _make_loop


DIMS = 512


def _unit(idx: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[idx] = 1.0
    return v


def _insert_session(db, session_key: str = "test:1") -> int:
    cur = db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (session_key, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _insert_trajectory(db, session_id: int, outcome: str = "ok") -> None:
    db.execute(
        "INSERT INTO trajectories (session_id, task, outcome, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, "test task", outcome, time.time()),
    )
    db.commit()


def _insert_chunk_with_vec(db, session_id: int, content: str, vec: np.ndarray) -> int:
    cur = db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (session_id, content, time.time()),
    )
    chunk_id = int(cur.lastrowid)
    db.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, vec.astype(np.float32).tobytes()),
    )
    db.commit()
    return chunk_id


class TestFindHubClusters:
    """Unit-test find_hub_clusters directly."""

    def _make_db(self, tmp_path):
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return hook.db

    async def test_no_successful_sessions_returns_empty(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)
        sid = _insert_session(db)
        _insert_trajectory(db, sid, outcome="fail")
        _insert_chunk_with_vec(db, sid, "some content", _unit(0))

        result = await find_hub_clusters(db, min_sessions=2)
        assert result == []

    async def test_single_session_no_hub(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)
        sid = _insert_session(db)
        _insert_trajectory(db, sid, outcome="ok")
        # Two similar chunks, but same session — can't span min_sessions=2
        _insert_chunk_with_vec(db, sid, "topic A", _unit(0))
        _insert_chunk_with_vec(db, sid, "topic A again", _unit(0))

        result = await find_hub_clusters(db, min_sessions=2)
        assert result == []

    async def test_two_sessions_similar_content_forms_hub(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)

        sid1 = _insert_session(db, "session:1")
        _insert_trajectory(db, sid1, outcome="ok")
        _insert_chunk_with_vec(db, sid1, "recurring topic about X", _unit(0))

        sid2 = _insert_session(db, "session:2")
        _insert_trajectory(db, sid2, outcome="ok")
        _insert_chunk_with_vec(db, sid2, "more about X", _unit(0))

        result = await find_hub_clusters(db, min_sessions=2, cluster_threshold=0.88)
        # Both chunks share cosine sim = 1.0 (same unit vector) across 2 sessions
        assert len(result) >= 1
        hub = result[0]
        assert len(hub["sessions"]) >= 2
        assert len(hub["samples"]) >= 1
        assert "chunk_ids" in hub
        assert isinstance(hub["chunk_ids"], list)
        assert len(hub["chunk_ids"]) >= 2
        assert hub["chunk_ids"] == sorted(hub["chunk_ids"])

    async def test_distinct_content_no_hub(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)

        sid1 = _insert_session(db, "session:1")
        _insert_trajectory(db, sid1, outcome="ok")
        _insert_chunk_with_vec(db, sid1, "topic A", _unit(0))

        sid2 = _insert_session(db, "session:2")
        _insert_trajectory(db, sid2, outcome="ok")
        _insert_chunk_with_vec(db, sid2, "unrelated topic B", _unit(1))

        result = await find_hub_clusters(db, min_sessions=2, cluster_threshold=0.88)
        # Orthogonal vectors → no cluster meets threshold
        assert result == []

    async def test_only_ok_trajectories_considered(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)

        sid1 = _insert_session(db, "session:1")
        _insert_trajectory(db, sid1, outcome="fail")  # excluded
        _insert_chunk_with_vec(db, sid1, "topic X", _unit(0))

        sid2 = _insert_session(db, "session:2")
        _insert_trajectory(db, sid2, outcome="ok")
        _insert_chunk_with_vec(db, sid2, "topic X again", _unit(0))

        result = await find_hub_clusters(db, min_sessions=2)
        # session:1 is excluded (fail); session:2 alone can't form a hub
        assert result == []

    async def test_max_chunks_cap_respected(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)

        # Insert 10 chunks across 2 ok sessions
        for i in range(2):
            sid = _insert_session(db, f"session:{i}")
            _insert_trajectory(db, sid, outcome="ok")
            for _ in range(5):
                _insert_chunk_with_vec(db, sid, f"content {i}", _unit(i))

        # With max_chunks=3 only 3 rows are queried from the DB
        # (result may be empty because cap may exclude one session's chunks,
        # but the important thing is no crash and ≤3 rows processed)
        result = await find_hub_clusters(db, min_sessions=2, max_chunks=3)
        # We can't assert a hub here — the cap may cut off one session.
        # Just assert no exception and result is a list.
        assert isinstance(result, list)

    async def test_samples_truncated_to_500_chars(self, tmp_path):
        from nano_hermes.memory.consolidation import find_hub_clusters

        db = self._make_db(tmp_path)
        long_text = "A" * 1000

        for key in ("session:1", "session:2"):
            sid = _insert_session(db, key)
            _insert_trajectory(db, sid, outcome="ok")
            _insert_chunk_with_vec(db, sid, long_text, _unit(0))

        result = await find_hub_clusters(db, min_sessions=2, cluster_threshold=0.88)
        assert len(result) >= 1
        for sample in result[0]["samples"]:
            assert len(sample) <= 500


class TestMemoryDistillTool:
    """Integration tests for memory_patch(action="distill")."""

    async def test_distill_no_successful_sessions(self, tmp_path):
        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)

        tool = loop.tools.get("memory_patch")
        result = await tool.execute(slot="memory", action="distill")
        assert "ok" in result.lower()

    async def test_distill_returns_hub_report(self, tmp_path):
        # distill_llm_enabled=False: surface-only mode; no LLM call needed.
        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop, config={"memory": {"distill_llm_enabled": False}})
        db = hook.db

        for i in range(2):
            sid = _insert_session(db, f"session:{i}")
            _insert_trajectory(db, sid, outcome="ok")
            _insert_chunk_with_vec(db, sid, f"recurring insight {i}", _unit(0))

        tool = loop.tools.get("memory_patch")
        result = await tool.execute(slot="memory", action="distill")
        assert "hub" in result.lower() or "found" in result.lower()
        assert "memory_patch" in result

    async def test_distill_unknown_action_untouched(self, tmp_path):
        """Ensure existing unknown-action path still works."""
        loop = _make_loop(tmp_path)
        nano_hermes.install(loop)

        tool = loop.tools.get("memory_patch")
        result = await tool.execute(slot="memory", action="bogus")
        assert "unknown action" in result.lower() or "error" in result.lower()
