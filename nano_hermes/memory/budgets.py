"""Budget-enforcing facade over ``nanobot.agent.memory.MemoryStore``.

Nanobot writes MEMORY.md / USER.md / SOUL.md without size checks. We wrap
the underlying store so the agent's ``memory_patch`` tool gets precise,
actionable errors when a write would overflow. The Dream / Consolidator
path writes directly through ``loop.context.memory`` — it skips budgets on
purpose (background consolidation gets latitude).
"""
from __future__ import annotations

from typing import Literal

from nanobot.agent.memory import MemoryStore as NanobotMemoryStore

from ..config import MemoryBudgets

Slot = Literal["memory", "user", "soul"]


class MemoryOverBudgetError(Exception):
    def __init__(self, slot: Slot, used: int, limit: int) -> None:
        super().__init__(
            f"{slot} budget exceeded: {used}/{limit} chars "
            f"(need {used - limit} fewer — remove or replace lower-value entries first)"
        )
        self.slot = slot
        self.used = used
        self.limit = limit
        self.overflow = used - limit


class MemoryEntryNotFoundError(Exception):
    def __init__(self, slot: Slot, needle: str) -> None:
        super().__init__(f"no substring match in {slot} for: {needle[:80]!r}")
        self.slot = slot
        self.needle = needle


class BudgetedMemory:
    """Thin wrapper that enforces char limits and exposes add/replace/remove
    semantics on top of nanobot's per-slot read/write methods."""

    def __init__(self, store: NanobotMemoryStore, budgets: MemoryBudgets) -> None:
        self.store = store
        self.budgets = budgets

    def _budget(self, slot: Slot) -> int:
        return {
            "memory": self.budgets.memory_md_chars,
            "user": self.budgets.user_md_chars,
            "soul": self.budgets.soul_md_chars,
        }[slot]

    def read(self, slot: Slot) -> str:
        if slot == "memory":
            return self.store.read_memory()
        if slot == "user":
            return self.store.read_user()
        if slot == "soul":
            return self.store.read_soul()
        raise ValueError(f"unknown slot {slot!r}")

    def write(self, slot: Slot, content: str) -> None:
        limit = self._budget(slot)
        if len(content) > limit:
            raise MemoryOverBudgetError(slot, len(content), limit)
        if slot == "memory":
            self.store.write_memory(content)
        elif slot == "user":
            self.store.write_user(content)
        elif slot == "soul":
            self.store.write_soul(content)
        else:
            raise ValueError(f"unknown slot {slot!r}")

    def add(self, slot: Slot, entry: str) -> None:
        cur = self.read(slot)
        joined = (cur + "\n" + entry.strip()) if cur else entry.strip()
        self.write(slot, joined.strip())

    def replace(self, slot: Slot, needle: str, replacement: str) -> None:
        cur = self.read(slot)
        if needle not in cur:
            raise MemoryEntryNotFoundError(slot, needle)
        self.write(slot, cur.replace(needle, replacement, 1))

    def remove(self, slot: Slot, needle: str) -> None:
        cur = self.read(slot)
        if needle not in cur:
            raise MemoryEntryNotFoundError(slot, needle)
        new = cur.replace(needle, "", 1)
        while "\n\n\n" in new:
            new = new.replace("\n\n\n", "\n\n")
        self.write(slot, new.strip())
