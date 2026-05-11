"""Tests for contextual preamble injection (embedding/contextual.py + archiver)."""
from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from nano_hermes.embedding.contextual import add_context_preamble
from nano_hermes.session.archiver import SessionArchiver
from nano_hermes.session.db import open_db


# ---------------------------------------------------------------------------
# Unit tests for add_context_preamble
# ---------------------------------------------------------------------------

class TestAddContextPreamble:
    def test_no_task_returns_text_unchanged(self):
        assert add_context_preamble("hello", None) == "hello"

    def test_empty_task_returns_text_unchanged(self):
        assert add_context_preamble("hello", "") == "hello"

    def test_whitespace_only_task_returns_text_unchanged(self):
        assert add_context_preamble("hello", "   ") == "hello"

    def test_preamble_prepended(self):
        result = add_context_preamble("the chunk text", "write a poem")
        assert result.startswith("Task: write a poem")
        assert "the chunk text" in result

    def test_long_task_truncated_to_200_chars(self):
        long_task = "x" * 300
        result = add_context_preamble("body", long_task)
        assert len(result) < 300 + len("Task: \n\nbody") + 10
        assert "x" * 200 in result
        assert "x" * 201 not in result

    def test_preamble_separated_by_double_newline(self):
        result = add_context_preamble("body", "task")
        assert "\n\nbody" in result


# ---------------------------------------------------------------------------
# Integration: archiver embeds with preamble, stores without
# ---------------------------------------------------------------------------

def _make_archiver(db: sqlite3.Connection) -> tuple[SessionArchiver, list[list[str]]]:
    """Return an archiver plus a list that captures texts passed to embed()."""
    captured: list[list[str]] = []

    async def fake_embed(texts: list[str]) -> list[np.ndarray]:
        captured.append(list(texts))
        return [np.zeros(4, dtype=np.float32) for _ in texts]

    chain = MagicMock()
    chain.__aenter__ = AsyncMock(return_value=chain)
    chain.__aexit__ = AsyncMock(return_value=False)
    chain.embed = AsyncMock(side_effect=fake_embed)

    archiver = SessionArchiver(
        db=db,
        embedder_factory=lambda: chain,
        target_dims=4,
        redact_secrets=False,
    )
    return archiver, captured


@pytest.fixture()
def db(tmp_path):
    conn = open_db(tmp_path, target_dims=4)
    yield conn
    conn.close()


class TestArchiverContextualEmbedding:
    def test_preamble_in_embed_not_in_db(self, db):
        archiver, captured = _make_archiver(db)
        msgs = [
            {"role": "user", "content": "fix the login bug"},
            {"role": "assistant", "content": "I'll look into it"},
        ]

        async def run():
            _, task = archiver.archive_and_embed(msgs)
            if task:
                await task

        asyncio.run(run())

        # Embedding input must contain preamble
        assert captured, "embed() was never called"
        all_texts = [t for batch in captured for t in batch]
        assert any("Task: fix the login bug" in t for t in all_texts), (
            f"preamble not found in embed texts: {all_texts}"
        )

        # Stored content must NOT contain preamble
        rows = db.execute("SELECT content FROM chunks").fetchall()
        stored = [r[0] for r in rows]
        assert all("Task:" not in s for s in stored), (
            f"preamble leaked into stored content: {stored}"
        )

    def test_no_preamble_when_no_user_message(self, db):
        archiver, captured = _make_archiver(db)
        msgs = [
            {"role": "assistant", "content": "Starting up"},
        ]

        async def run():
            _, task = archiver.archive_and_embed(msgs)
            if task:
                await task

        asyncio.run(run())

        all_texts = [t for batch in captured for t in batch]
        # No user message → no preamble
        assert all("Task:" not in t for t in all_texts), (
            f"unexpected preamble: {all_texts}"
        )

    def test_task_context_consistent_across_batches(self, db):
        archiver, captured = _make_archiver(db)
        msgs: list[dict] = [{"role": "user", "content": "deploy the service"}]

        async def run_batch1():
            _, task = archiver.archive_and_embed(msgs)
            if task:
                await task

        asyncio.run(run_batch1())

        msgs.append({"role": "assistant", "content": "Deploying now"})

        async def run_batch2():
            _, task = archiver.archive_and_embed(msgs)
            if task:
                await task

        asyncio.run(run_batch2())

        # Both batches should have preamble with the same task
        all_texts = [t for batch in captured for t in batch]
        preamble_texts = [t for t in all_texts if "Task:" in t]
        assert preamble_texts, "no preambled texts found"
        # All preambled texts reference the same task
        assert all("deploy the service" in t for t in preamble_texts)
