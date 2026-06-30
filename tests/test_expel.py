"""Tests for ExpeL contrastive insight extraction (memory/expel.py)."""
from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import nano_hermes
from conftest import _make_loop
from nano_hermes.memory.expel import (
    _outcome_class,
    _opposite_outcomes,
    _store_insight,
    extract_contrastive_insight,
    find_contrasting_session,
)


def _hook(tmp_path, config=None):
    cfg = config or {}
    return nano_hermes.install(_make_loop(tmp_path), config=cfg)


def _add_trajectory(db, session_id, outcome, created_at=None):
    db.execute(
        "INSERT INTO trajectories (session_id, task, outcome, created_at) "
        "VALUES (?, ?, ?, ?)",
        (session_id, f"task for session {session_id}", outcome, created_at or time.time()),
    )
    db.commit()


def _add_session(db, session_key="s"):
    cur = db.execute(
        "INSERT INTO sessions (session_key, started_at) VALUES (?, ?)",
        (session_key, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _add_chunk(db, session_id, content="do the thing"):
    cur = db.execute(
        "INSERT INTO chunks (session_id, turn_index, role, content, created_at) "
        "VALUES (?, 0, 'user', ?, ?)",
        (session_id, content, time.time()),
    )
    db.commit()
    return int(cur.lastrowid)


def _add_chunk_emb(db, session_id, vec, content="do the thing"):
    import numpy as np
    chunk_id = _add_chunk(db, session_id, content)
    db.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?, ?)",
        (chunk_id, np.asarray(vec, dtype=np.float32).tobytes()),
    )
    db.commit()
    return chunk_id


class TestOutcomeHelpers:
    def test_outcome_class_success(self):
        assert _outcome_class("success") == "success"

    def test_outcome_class_fail(self):
        assert _outcome_class("fail") == "failure"

    def test_outcome_class_partial(self):
        assert _outcome_class("partial") == "failure"

    def test_opposite_of_success(self):
        assert set(_opposite_outcomes("success")) == {"fail", "partial"}

    def test_opposite_of_failure(self):
        assert set(_opposite_outcomes("failure")) == {"success"}


class TestFindContrastingSession:
    def test_no_trajectories_returns_none(self, tmp_path):
        hook = _hook(tmp_path)
        sid = _add_session(hook.db)
        _add_chunk(hook.db, sid)
        assert find_contrasting_session(hook.db, sid, "success") is None

    def test_no_opposite_outcome_returns_none(self, tmp_path):
        hook = _hook(tmp_path)
        sid1 = _add_session(hook.db, "s1")
        sid2 = _add_session(hook.db, "s2")
        _add_chunk(hook.db, sid1)
        _add_chunk(hook.db, sid2)
        _add_trajectory(hook.db, sid2, "success")  # same class — no contrast
        assert find_contrasting_session(hook.db, sid1, "success") is None

    def test_no_chunk_embedding_returns_none(self, tmp_path):
        hook = _hook(tmp_path)
        sid1 = _add_session(hook.db, "s1")
        sid2 = _add_session(hook.db, "s2")
        _add_chunk(hook.db, sid1)  # no embedding
        _add_chunk(hook.db, sid2)
        _add_trajectory(hook.db, sid2, "fail")
        assert find_contrasting_session(hook.db, sid1, "success") is None

    def test_finds_similar_opposite_outcome(self, tmp_path):
        # Regression guard: candidate embedding must be looked up by chunk_id,
        # not rowid — exercises the cosine-pairing loop end to end.
        dims = 512
        near = [1.0] + [0.0] * (dims - 1)
        far = [0.0, 1.0] + [0.0] * (dims - 2)
        hook = _hook(tmp_path)
        cur = _add_session(hook.db, "cur")
        match = _add_session(hook.db, "match")
        other = _add_session(hook.db, "other")
        _add_chunk_emb(hook.db, cur, near)
        _add_chunk_emb(hook.db, match, near)   # cosine 1.0 with cur
        _add_chunk_emb(hook.db, other, far)    # cosine 0.0 with cur
        _add_trajectory(hook.db, match, "fail")
        _add_trajectory(hook.db, other, "fail")
        result = find_contrasting_session(hook.db, cur, "success", threshold=0.5)
        assert result is not None
        assert result[0] == match
        assert result[1] == "fail"

    def test_below_threshold_returns_none(self, tmp_path):
        dims = 512
        near = [1.0] + [0.0] * (dims - 1)
        far = [0.0, 1.0] + [0.0] * (dims - 2)
        hook = _hook(tmp_path)
        cur = _add_session(hook.db, "cur")
        cand = _add_session(hook.db, "cand")
        _add_chunk_emb(hook.db, cur, near)
        _add_chunk_emb(hook.db, cand, far)     # cosine 0.0 < threshold
        _add_trajectory(hook.db, cand, "fail")
        assert find_contrasting_session(hook.db, cur, "success", threshold=0.5) is None


class TestStoreInsight:
    def test_stores_with_correct_fact_type(self, tmp_path):
        hook = _hook(tmp_path)
        fact_id = _store_insight(hook.db, "Check inputs before calling API.", "api task")
        row = hook.db.execute(
            "SELECT fact_type, task_category, content FROM semantic_facts WHERE id = ?",
            (fact_id,),
        ).fetchone()
        assert row[0] == "expel"
        assert row[1] == "api task"
        assert "API" in row[2]

    def test_store_increments_id(self, tmp_path):
        hook = _hook(tmp_path)
        id1 = _store_insight(hook.db, "Lesson one is important.", "cat1")
        id2 = _store_insight(hook.db, "Lesson two is different.", "cat2")
        assert id2 > id1


class TestExtractContrastiveInsight:
    def _make_resp(self, content="Always verify inputs first."):
        resp = MagicMock()
        resp.finish_reason = "stop"
        resp.content = content
        return resp

    def test_disabled_returns_none(self, tmp_path):
        hook = _hook(tmp_path, config={})
        hook.config.expel_enabled = False
        result = asyncio.run(
            extract_contrastive_insight(hook, session_id=1, outcome="success", messages=[])
        )
        assert result is None

    def test_no_contrast_returns_none(self, tmp_path):
        hook = _hook(tmp_path)
        sid = _add_session(hook.db)
        _add_chunk(hook.db, sid)
        result = asyncio.run(
            extract_contrastive_insight(hook, session_id=sid, outcome="success", messages=[])
        )
        assert result is None

    def test_skip_response_not_stored(self, tmp_path):
        hook = _hook(tmp_path)
        mock_provider = AsyncMock()
        mock_provider.chat_with_retry = AsyncMock(return_value=self._make_resp("SKIP"))
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(99, "fail"),
        ), patch("nano_hermes.memory.expel._chunk_text", return_value="some task text"):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=1, outcome="success", messages=[])
            )
        assert result is None
        count = hook.db.execute(
            "SELECT COUNT(*) FROM semantic_facts WHERE fact_type = 'expel'"
        ).fetchone()[0]
        assert count == 0

    def test_short_response_not_stored(self, tmp_path):
        hook = _hook(tmp_path)
        mock_provider = AsyncMock()
        mock_provider.chat_with_retry = AsyncMock(return_value=self._make_resp("ok"))
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(99, "fail"),
        ), patch("nano_hermes.memory.expel._chunk_text", return_value="some meaningful task"):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=1, outcome="success", messages=[])
            )
        assert result is None

    def test_valid_insight_stored(self, tmp_path):
        hook = _hook(tmp_path)
        insight_text = "Always verify API tokens before making calls to prevent auth errors."
        mock_provider = AsyncMock()
        mock_provider.chat_with_retry = AsyncMock(
            return_value=self._make_resp(insight_text)
        )
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(99, "fail"),
        ), patch(
            "nano_hermes.memory.expel._chunk_text",
            return_value="investigate the failing service endpoint",
        ):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=1, outcome="success", messages=[])
            )

        assert result == insight_text
        row = hook.db.execute(
            "SELECT content, fact_type FROM semantic_facts WHERE fact_type = 'expel'"
        ).fetchone()
        assert row is not None
        assert row[0] == insight_text
        assert row[1] == "expel"

    def test_llm_error_returns_none(self, tmp_path):
        hook = _hook(tmp_path)
        err_resp = MagicMock()
        err_resp.finish_reason = "error"
        err_resp.error_status_code = 429
        err_resp.error_type = "rate_limit"
        err_resp.error_code = ""
        err_resp.content = ""
        err_resp.error_kind = ""
        mock_provider = AsyncMock()
        mock_provider.chat_with_retry = AsyncMock(return_value=err_resp)
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(99, "fail"),
        ), patch("nano_hermes.memory.expel._chunk_text", return_value="some task text"):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=1, outcome="success", messages=[])
            )
        assert result is None

    def test_success_to_fail_contrast(self, tmp_path):
        hook = _hook(tmp_path)
        mock_provider = AsyncMock()
        insight = "Confirm the environment variable is set before running the task."
        mock_provider.chat_with_retry = AsyncMock(return_value=self._make_resp(insight))
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(55, "partial"),
        ), patch("nano_hermes.memory.expel._chunk_text", return_value="run deployment pipeline"):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=10, outcome="success", messages=[])
            )
        assert result == insight

    def test_fail_to_success_contrast(self, tmp_path):
        hook = _hook(tmp_path)
        mock_provider = AsyncMock()
        insight = "The key difference was using the correct retry backoff strategy."
        mock_provider.chat_with_retry = AsyncMock(return_value=self._make_resp(insight))
        hook._loop.provider = mock_provider

        with patch(
            "nano_hermes.memory.expel.find_contrasting_session",
            return_value=(77, "success"),
        ), patch("nano_hermes.memory.expel._chunk_text", return_value="debug the webhook handler"):
            result = asyncio.run(
                extract_contrastive_insight(hook, session_id=20, outcome="fail", messages=[])
            )
        assert result == insight
