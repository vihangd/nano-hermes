"""Tests for embedding-based memory consolidation (memory/consolidation.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from nano_hermes.memory.consolidation import (
    consolidate_entries,
    greedy_cluster,
    split_entries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _norm(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_embedder(vectors: list[np.ndarray]):
    """Return an embedder_factory whose chain.embed() returns *vectors*."""
    chain = MagicMock()
    chain.embed = AsyncMock(return_value=vectors)
    chain.__aenter__ = AsyncMock(return_value=chain)
    chain.__aexit__ = AsyncMock(return_value=False)
    factory = MagicMock(return_value=chain)
    return factory


# ---------------------------------------------------------------------------
# split_entries
# ---------------------------------------------------------------------------

class TestSplitEntries:
    def test_paragraph_split(self):
        text = "Entry A\n\nEntry B\n\nEntry C"
        assert split_entries(text) == ["Entry A", "Entry B", "Entry C"]

    def test_line_split_fallback(self):
        text = "- Item one\n- Item two\n- Item three"
        result = split_entries(text)
        assert result == ["- Item one", "- Item two", "- Item three"]

    def test_filters_empty_lines(self):
        text = "\n\nEntry A\n\n\nEntry B\n\n"
        result = split_entries(text)
        assert result == ["Entry A", "Entry B"]

    def test_single_entry(self):
        text = "Just one entry"
        assert split_entries(text) == ["Just one entry"]

    def test_empty_string(self):
        assert split_entries("") == []
        assert split_entries("   \n\n  ") == []


# ---------------------------------------------------------------------------
# greedy_cluster
# ---------------------------------------------------------------------------

class TestGreedyCluster:
    def test_near_identical_vectors_merge(self):
        base = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        near = _norm(np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32))
        clusters = greedy_cluster([base, near], threshold=0.92)
        assert len(clusters) == 1
        assert set(clusters[0]) == {0, 1}

    def test_orthogonal_vectors_separate(self):
        a = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        b = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        clusters = greedy_cluster([a, b], threshold=0.92)
        assert len(clusters) == 2

    def test_three_entries_two_clusters(self):
        a = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        a2 = _norm(np.array([0.98, 0.1, 0.0, 0.0], dtype=np.float32))
        b = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        clusters = greedy_cluster([a, a2, b], threshold=0.92)
        assert len(clusters) == 2
        # a and a2 should be in the same cluster
        flat = {frozenset(c) for c in clusters}
        assert frozenset({0, 1}) in flat

    def test_single_vector_one_cluster(self):
        v = _norm(np.array([1.0, 0.0], dtype=np.float32))
        clusters = greedy_cluster([v], threshold=0.92)
        assert clusters == [[0]]

    def test_threshold_boundary(self):
        # Construct two vectors with exactly dot product == 0.92
        a = _norm(np.array([1.0, 0.0], dtype=np.float32))
        cos_val = 0.92
        sin_val = float(np.sqrt(1 - cos_val ** 2))
        b = _norm(np.array([cos_val, sin_val], dtype=np.float32))
        clusters = greedy_cluster([a, b], threshold=0.92)
        # At exactly threshold they should merge (>=)
        assert len(clusters) == 1

    def test_empty_input(self):
        assert greedy_cluster([]) == []


# ---------------------------------------------------------------------------
# consolidate_entries
# ---------------------------------------------------------------------------

class TestConsolidateEntries:
    def test_merges_near_duplicates(self):
        entries = ["Short duplicate", "Longer entry that wins because it is longer"]
        base = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        near = _norm(np.array([0.99, 0.1, 0.0, 0.0], dtype=np.float32))
        factory = _make_embedder([base, near])

        surviving, n_removed = asyncio.run(consolidate_entries(entries, factory, threshold=0.92))

        assert n_removed == 1
        assert len(surviving) == 1
        assert surviving[0] == "Longer entry that wins because it is longer"

    def test_preserves_distinct_entries(self):
        entries = ["Entry about Python", "Entry about databases"]
        a = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        b = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        factory = _make_embedder([a, b])

        surviving, n_removed = asyncio.run(consolidate_entries(entries, factory, threshold=0.92))

        assert n_removed == 0
        assert len(surviving) == 2

    def test_empty_list_no_op(self):
        factory = _make_embedder([])
        surviving, n_removed = asyncio.run(consolidate_entries([], factory))
        assert surviving == []
        assert n_removed == 0
        factory.assert_not_called()

    def test_single_entry_no_op(self):
        factory = _make_embedder([])
        surviving, n_removed = asyncio.run(consolidate_entries(["only one"], factory))
        assert surviving == ["only one"]
        assert n_removed == 0
        factory.assert_not_called()

    def test_longest_wins_in_cluster(self):
        entries = ["a", "bb", "ccc"]
        # All three are near-identical
        v = _norm(np.array([1.0, 0.0], dtype=np.float32))
        factory = _make_embedder([v, v, v])

        surviving, n_removed = asyncio.run(consolidate_entries(entries, factory, threshold=0.92))

        assert n_removed == 2
        assert surviving == ["ccc"]

    def test_three_entries_partial_merge(self):
        entries = ["dup A", "dup B longer", "completely different"]
        dup_vec = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        diff_vec = _norm(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
        factory = _make_embedder([dup_vec, dup_vec, diff_vec])

        surviving, n_removed = asyncio.run(consolidate_entries(entries, factory, threshold=0.92))

        assert n_removed == 1
        assert len(surviving) == 2
        assert "dup B longer" in surviving
        assert "completely different" in surviving


# ---------------------------------------------------------------------------
# Integration: MemoryPatchTool consolidate action
# ---------------------------------------------------------------------------

class TestMemoryPatchToolConsolidate:
    def _make_tool(self, tmp_path):
        import nano_hermes
        from conftest import _make_loop
        from nano_hermes.memory.tool import MemoryPatchTool

        loop = _make_loop(tmp_path)
        hook = nano_hermes.install(loop)
        return MemoryPatchTool(hook=hook), hook

    def test_consolidate_empty_slot(self, tmp_path):
        tool, hook = self._make_tool(tmp_path)
        # memory slot is empty by default
        result = asyncio.run(tool.execute(slot="memory", action="consolidate"))
        assert "empty" in result or "nothing" in result

    def test_consolidate_single_entry(self, tmp_path):
        tool, hook = self._make_tool(tmp_path)
        hook.budgeted_memory.add("memory", "- Only one entry here")
        result = asyncio.run(tool.execute(slot="memory", action="consolidate"))
        assert "nothing" in result

    def test_consolidate_merges_near_duplicates(self, tmp_path):
        tool, hook = self._make_tool(tmp_path)

        # Write two entries (paragraph-separated)
        hook.budgeted_memory.write("memory", "Short dup\n\nLonger duplicate entry wins")

        # Patch the hook's embedder to return near-identical vectors
        dup_vec = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        mock_chain = MagicMock()
        mock_chain.embed = AsyncMock(return_value=[dup_vec, dup_vec])
        mock_chain.__aenter__ = AsyncMock(return_value=mock_chain)
        mock_chain.__aexit__ = AsyncMock(return_value=False)
        hook.embedder = MagicMock(return_value=mock_chain)

        result = asyncio.run(tool.execute(slot="memory", action="consolidate"))

        assert "merged" in result or "consolidated" in result
        remaining = hook.budgeted_memory.read("memory")
        assert "Longer duplicate entry wins" in remaining
        assert "Short dup" not in remaining

    def test_consolidate_no_op_on_distinct_entries(self, tmp_path):
        tool, hook = self._make_tool(tmp_path)

        hook.budgeted_memory.write("memory", "Entry about Python\n\nEntry about databases")

        a = _norm(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
        b = _norm(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
        mock_chain = MagicMock()
        mock_chain.embed = AsyncMock(return_value=[a, b])
        mock_chain.__aenter__ = AsyncMock(return_value=mock_chain)
        mock_chain.__aexit__ = AsyncMock(return_value=False)
        hook.embedder = MagicMock(return_value=mock_chain)

        result = asyncio.run(tool.execute(slot="memory", action="consolidate"))

        assert "no near-duplicates" in result
        # Both entries preserved
        remaining = hook.budgeted_memory.read("memory")
        assert "Entry about Python" in remaining
        assert "Entry about databases" in remaining
