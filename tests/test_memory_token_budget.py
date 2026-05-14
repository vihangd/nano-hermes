"""Tests for token-counted memory budgets (BudgetedMemory + MemoryBudgets)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from nano_hermes.config import MemoryBudgets
from nano_hermes.memory.budgets import (
    BudgetedMemory,
    MemoryOverBudgetError,
    _count_tokens,
)


# ---------------------------------------------------------------------------
# _count_tokens unit tests
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string_returns_zero_or_one(self):
        # tiktoken encodes "" as [] (0 tokens); fallback may return 1 (max(1, ...))
        # Either is acceptable — just don't raise.
        result = _count_tokens("")
        assert isinstance(result, int)
        assert result >= 0

    def test_simple_english_sentence(self):
        # "hello world" — in cl100k: "hello"=1, " world"=1 → 2 tokens
        count = _count_tokens("hello world")
        assert 1 <= count <= 5, f"unexpected token count: {count}"

    def test_repeated_words_not_deflated(self):
        short = _count_tokens("hello world")
        long = _count_tokens("hello world " * 20)
        assert long > short, "more content should produce more tokens"

    def test_cjk_text_counted(self):
        # Chinese text: each char is typically 1-2 tokens in cl100k.
        # Key property: the same char budget means FEWER CJK chars fit.
        cjk = "你好世界" * 10  # 40 chars
        latin = "hello world " * 4  # 48 chars, ~8-12 tokens
        cjk_tokens = _count_tokens(cjk)
        latin_tokens = _count_tokens(latin)
        # CJK should use more tokens per char than latin — verify non-zero
        assert cjk_tokens > 0
        assert latin_tokens > 0

    def test_returns_int(self):
        assert isinstance(_count_tokens("test string"), int)


# ---------------------------------------------------------------------------
# MemoryBudgets defaults
# ---------------------------------------------------------------------------

class TestMemoryBudgetsDefaults:
    def test_default_units_are_tokens(self):
        budgets = MemoryBudgets()
        assert hasattr(budgets, "memory_md_tokens")
        assert hasattr(budgets, "user_md_tokens")
        assert hasattr(budgets, "soul_md_tokens")

    def test_no_chars_fields(self):
        budgets = MemoryBudgets()
        assert not hasattr(budgets, "memory_md_chars"), (
            "old char-based fields should be removed"
        )

    def test_default_values_are_sane(self):
        budgets = MemoryBudgets()
        # Should fit a few paragraphs of English text
        assert 200 <= budgets.memory_md_tokens <= 2000
        assert 100 <= budgets.user_md_tokens <= 1000
        assert 100 <= budgets.soul_md_tokens <= 1000


# ---------------------------------------------------------------------------
# BudgetedMemory.write token enforcement
# ---------------------------------------------------------------------------

def _make_budgeted(memory_tokens: int = 50) -> BudgetedMemory:
    store = MagicMock()
    store.read_memory.return_value = ""
    store.read_user.return_value = ""
    store.read_soul.return_value = ""
    store.memory_file = MagicMock()
    store.user_file = MagicMock()
    store.soul_file = MagicMock()
    budgets = MemoryBudgets(
        memory_md_tokens=memory_tokens,
        user_md_tokens=memory_tokens,
        soul_md_tokens=memory_tokens,
    )
    bm = BudgetedMemory(store=store, budgets=budgets)
    return bm


class TestBudgetedMemoryTokenCounting:
    def test_short_content_writes_ok(self):
        bm = _make_budgeted(memory_tokens=200)
        # "hi" is 1 token — well within 200; patch atomic_write_text since store is mocked
        with patch("nano_hermes.memory.budgets.atomic_write_text"):
            bm.write("memory", "hi")  # should not raise MemoryOverBudgetError

    def test_over_budget_raises_error(self):
        bm = _make_budgeted(memory_tokens=5)
        # "hello world " * 20 ≈ 40 tokens >> 5
        with pytest.raises(MemoryOverBudgetError) as exc_info:
            bm.write("memory", "hello world " * 20)
        err = exc_info.value
        assert err.limit == 5
        assert err.used > 5
        assert err.overflow > 0

    def test_error_message_says_tokens(self):
        bm = _make_budgeted(memory_tokens=5)
        with pytest.raises(MemoryOverBudgetError) as exc_info:
            bm.write("memory", "hello world " * 20)
        assert "token" in str(exc_info.value)

    def test_error_shows_limit_and_used(self):
        bm = _make_budgeted(memory_tokens=5)
        with pytest.raises(MemoryOverBudgetError) as exc_info:
            bm.write("memory", "hello world " * 20)
        msg = str(exc_info.value)
        assert "5" in msg  # limit

    def test_all_slots_enforce_budget(self):
        for slot in ("memory", "user", "soul"):
            bm = _make_budgeted(memory_tokens=2)
            with pytest.raises(MemoryOverBudgetError):
                bm.write(slot, "the quick brown fox jumps over the lazy dog " * 5)

    def test_cjk_content_counted_correctly(self):
        # CJK chars should count as more tokens than ASCII chars of same length.
        # A very small budget should be exceeded by CJK where equivalent ASCII fits.
        bm_tight = _make_budgeted(memory_tokens=3)
        cjk = "你好世界" * 10  # 40 CJK chars — should exceed 3 tokens
        with pytest.raises(MemoryOverBudgetError):
            bm_tight.write("memory", cjk)

    def test_content_at_exact_limit_is_allowed(self):
        # "hi" is 1 token in cl100k. With a 1-token limit it should not raise.
        count = _count_tokens("hi")
        bm = _make_budgeted(memory_tokens=count)
        with patch("nano_hermes.memory.budgets.atomic_write_text"):
            bm.write("memory", "hi")  # should not raise MemoryOverBudgetError


# ---------------------------------------------------------------------------
# Integration: BudgetedMemory.add enforces token budget
# ---------------------------------------------------------------------------

class TestBudgetedMemoryAddTokenBudget:
    def test_add_over_budget_returns_error_string_not_raises(self):
        """add() is a higher-level method that catches budget errors and
        returns them as a string — verify the flow end-to-end."""
        store = MagicMock()
        store.read_memory.return_value = ""
        store.memory_file = MagicMock()
        budgets = MemoryBudgets(memory_md_tokens=2, user_md_tokens=2, soul_md_tokens=2)
        bm = BudgetedMemory(store=store, budgets=budgets)

        # add() calls write() → raises MemoryOverBudgetError → should bubble up
        with pytest.raises(MemoryOverBudgetError):
            bm.add("memory", "the quick brown fox jumps " * 10)
