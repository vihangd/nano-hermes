"""Tests for Phase 4.3: LLM-enabled episodic→semantic distillation.

Tests 1-3 (find_hub_clusters chunk_ids, empty corpus, single-session)
are covered by test_memory_distill.py (augmented). This file covers
the LLM-path tests:
  4. _distill with LLM enabled → writes to semantic_facts + returns provenance
  5. _distill with LLM failure → no DB write, returns "no usable facts"
  6. _distill with distill_llm_enabled=False → no LLM call, no DB write
  7. second run with same corpus → writes new rows, no crash
"""
from __future__ import annotations

import json
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import nano_hermes
from conftest import _make_loop
from nano_hermes.memory.tool import MemoryPatchTool

DIMS = 512


def _unit(idx: int) -> np.ndarray:
    v = np.zeros(DIMS, dtype=np.float32)
    v[idx] = 1.0
    return v


def _insert_session(db, session_key: str) -> int:
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


def _make_hook(tmp_path, config_overrides=None):
    loop = _make_loop(tmp_path)
    hook = nano_hermes.install(loop, config=config_overrides or {})
    hook._loop.provider = MagicMock()
    return hook


def _mock_response(text: str) -> MagicMock:
    resp = MagicMock()
    resp.content = text
    return resp


def _seed_two_session_hub(db):
    """Insert two ok sessions with identical unit vectors — forms a hub."""
    chunk_ids = []
    for i in range(2):
        sid = _insert_session(db, f"session:{i}")
        _insert_trajectory(db, sid, outcome="ok")
        cid = _insert_chunk_with_vec(db, sid, f"recurring topic about X ({i})", _unit(0))
        chunk_ids.append(cid)
    return chunk_ids


# ---------------------------------------------------------------------------
# Test 4: LLM enabled → writes to semantic_facts, returns provenance
# ---------------------------------------------------------------------------

class TestDistillLLMEnabled:
    async def test_writes_to_semantic_facts_and_returns_provenance(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("A recurring fact about X.")
        )

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        rows = hook.db.execute(
            "SELECT content, source_chunk_ids FROM semantic_facts"
        ).fetchall()
        assert len(rows) == 1
        content, source_ids_json = rows[0]
        assert "recurring fact about X" in content
        ids = json.loads(source_ids_json)
        assert isinstance(ids, list)
        assert len(ids) >= 2

        assert "recurring fact about X" in result
        assert "chunk_ids" in result
        assert "memory_patch" in result

    async def test_llm_called_with_hub_samples_in_prompt(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("A fact.")
        )

        tool = MemoryPatchTool(hook=hook)
        await tool.execute(action="distill")

        assert hook._loop.provider.chat_with_retry.call_count >= 1
        call_kwargs = hook._loop.provider.chat_with_retry.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        prompt_text = messages[0]["content"]
        assert "recurring topic about X" in prompt_text

    async def test_source_chunk_ids_sorted_ascending(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("Some fact.")
        )

        tool = MemoryPatchTool(hook=hook)
        await tool.execute(action="distill")

        row = hook.db.execute(
            "SELECT source_chunk_ids FROM semantic_facts"
        ).fetchone()
        ids = json.loads(row[0])
        assert ids == sorted(ids)


# ---------------------------------------------------------------------------
# Test 5: LLM failure → no write, returns "no usable facts" message
# ---------------------------------------------------------------------------

class TestDistillLLMFailure:
    async def test_no_write_on_llm_exception(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            side_effect=RuntimeError("network timeout")
        )

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        rows = hook.db.execute("SELECT id FROM semantic_facts").fetchall()
        assert rows == []
        assert "no usable facts" in result.lower() or "ok" in result.lower()

    async def test_no_write_on_empty_llm_response(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("")
        )

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        rows = hook.db.execute("SELECT id FROM semantic_facts").fetchall()
        assert rows == []
        assert "no usable facts" in result.lower() or "ok" in result.lower()


# ---------------------------------------------------------------------------
# Test 6: distill_llm_enabled=False → no LLM call, no DB write, surfaces hubs
# ---------------------------------------------------------------------------

class TestDistillLLMDisabled:
    async def test_no_llm_call_no_db_write(self, tmp_path):
        hook = _make_hook(tmp_path, {"memory": {"distill_llm_enabled": False}})
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock()

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        hook._loop.provider.chat_with_retry.assert_not_called()
        rows = hook.db.execute("SELECT id FROM semantic_facts").fetchall()
        assert rows == []

        assert "hub" in result.lower() or "found" in result.lower()
        assert "chunk_ids" in result
        assert "memory_patch" in result

    async def test_no_hubs_returns_ok(self, tmp_path):
        hook = _make_hook(tmp_path, {"memory": {"distill_llm_enabled": False}})
        hook._loop.provider.chat_with_retry = AsyncMock()

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        hook._loop.provider.chat_with_retry.assert_not_called()
        assert "no recurring" in result.lower() or "ok" in result.lower()


# ---------------------------------------------------------------------------
# Test 7: second run → writes new rows (no dedup required), no crash
# ---------------------------------------------------------------------------

class TestDistillIdempotent:
    async def test_second_run_writes_new_rows(self, tmp_path):
        hook = _make_hook(tmp_path)
        _seed_two_session_hub(hook.db)
        hook._loop.provider.chat_with_retry = AsyncMock(
            return_value=_mock_response("A fact about X.")
        )

        tool = MemoryPatchTool(hook=hook)
        await tool.execute(action="distill")
        rows_after_first = hook.db.execute("SELECT id FROM semantic_facts").fetchall()
        assert len(rows_after_first) == 1

        await tool.execute(action="distill")
        rows_after_second = hook.db.execute("SELECT id FROM semantic_facts").fetchall()
        assert len(rows_after_second) == 2

    async def test_partial_hub_failure_preserves_successful_facts(self, tmp_path):
        """Hub 1 succeeds, hub 2 fails — hub 1's fact must be durably committed."""
        hook = _make_hook(tmp_path)
        db = hook.db

        for i in range(2):
            sid = _insert_session(db, f"sessionA:{i}")
            _insert_trajectory(db, sid, outcome="ok")
            _insert_chunk_with_vec(db, sid, f"hub A topic ({i})", _unit(0))

        for i in range(2):
            sid = _insert_session(db, f"sessionB:{i}")
            _insert_trajectory(db, sid, outcome="ok")
            _insert_chunk_with_vec(db, sid, f"hub B topic ({i})", _unit(1))

        call_count = 0

        async def _flaky_chat(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                resp = MagicMock()
                resp.content = "Hub A fact."
                return resp
            raise RuntimeError("hub B failed")

        hook._loop.provider.chat_with_retry = _flaky_chat

        tool = MemoryPatchTool(hook=hook)
        result = await tool.execute(action="distill")

        rows = hook.db.execute("SELECT content FROM semantic_facts").fetchall()
        assert len(rows) >= 1
        assert any("Hub A fact" in r[0] for r in rows)
