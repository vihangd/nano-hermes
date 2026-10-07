"""Budget-enforcing facade over ``nanobot.agent.memory.MemoryStore``.

Nanobot writes MEMORY.md / USER.md / SOUL.md without size checks. We wrap
the underlying store so the agent's ``memory_patch`` tool gets precise,
actionable errors when a write would overflow. The Dream / Consolidator
path writes directly through ``loop.context.memory`` — it skips budgets on
purpose (background consolidation gets latitude).
"""
from __future__ import annotations

import logging
from typing import Literal

from nanobot.agent.memory import MemoryStore as NanobotMemoryStore

from .._atomic import atomic_write_text
from ..config import MemoryBudgets
from .guard import RAW_READ_ATTR, scan_memory_content

log = logging.getLogger(__name__)


# Cached encoder + a sentinel marking "tiktoken unavailable, use the byte
# heuristic". ``get_encoding`` loads a ~1.7MB BPE vocab (and may hit the
# network on first call) — doing that on every memory_patch budget check is
# pure waste on a Pi, so resolve it once per process.
_ENC: object | None = None


def _encoder() -> object | None:
    """Return a memoized cl100k_base encoder, or ``None`` if tiktoken can't
    be loaded (e.g. restricted-network Pi where the vocab can't be fetched)."""
    global _ENC
    if _ENC is None:
        try:
            import tiktoken
            _ENC = tiktoken.get_encoding("cl100k_base")
        except Exception:
            log.debug("tiktoken unavailable — falling back to byte estimate", exc_info=True)
            _ENC = False  # negative cache — don't retry the failed import
    return _ENC or None


def _count_tokens(text: str) -> int:
    """Count tokens using cl100k_base (GPT-4 / Claude shared BPE).

    Falls back to a UTF-8 byte approximation if tiktoken is unavailable
    (e.g., restricted-network Pi environment where the vocab file can't be
    downloaded). The fallback slightly over-counts CJK but stays conservative.
    """
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    # ~4 UTF-8 bytes per token for Latin text; ~3 bytes/char for CJK
    # which is ~1.5 bytes/token — dividing by 3 is a safe upper bound.
    return max(1, len(text.encode("utf-8")) // 3)

Slot = Literal["memory", "user", "soul"]


class MemoryOverBudgetError(Exception):
    def __init__(self, slot: Slot, used: int, limit: int) -> None:
        super().__init__(
            f"{slot} budget exceeded: {used}/{limit} tokens "
            f"(need {used - limit} fewer — remove or replace lower-value entries first)"
        )
        self.slot = slot
        self.used = used
        self.limit = limit
        self.overflow = used - limit


class MemoryEntryNotFoundError(Exception):
    def __init__(self, slot: Slot, needle: str, closest: list[str] | None = None) -> None:
        msg = f"no substring match in {slot} for: {needle[:80]!r}"
        if closest:
            # The model usually paraphrased or quoted a stale copy; naming the
            # nearest stored lines lets it retry with the real text.
            msg += "; closest stored lines: " + " | ".join(repr(c[:80]) for c in closest)
        super().__init__(msg)
        self.slot = slot
        self.needle = needle


# Typography a model re-emits differently from what it stored: curly vs
# straight quotes, en/em dashes vs hyphens. Folded only for the fallback match.
_FOLD = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"', "\u2033": '"',
    "\u2013": "-", "\u2014": "-", "\u2012": "-", "\u2212": "-",
})


def _folded(text: str) -> tuple[str, list[int]]:
    """*text* with typography folded and whitespace runs collapsed to one
    space, plus each folded character's index in *text*."""
    out: list[str] = []
    index: list[int] = []
    in_space = False
    for i, ch in enumerate(text.translate(_FOLD)):
        if ch.isspace():
            if in_space:
                continue
            in_space = True
            ch = " "
        else:
            in_space = False
        out.append(ch)
        index.append(i)
    return "".join(out), index


def _closest_lines(text: str, needle: str, n: int = 2) -> list[str]:
    words = {w for w in needle.lower().split() if any(c.isalnum() for c in w)}
    scored = [
        (len(words & set(line.lower().split())), line.strip())
        for line in text.splitlines()
        if line.strip()
    ]
    return [line for score, line in sorted(scored, key=lambda x: -x[0])[:n] if score]


def _unique_find(slot: Slot, text: str, needle: str) -> int | None:
    """Index of *needle*'s only occurrence, None if absent; raise if several.

    Overlapping occurrences count: ``str.count`` would see ``"abab"`` once in
    ``"ababab"``, where it names two places.
    """
    at = text.find(needle)
    if at == -1:
        return None
    if text.find(needle, at + 1) != -1:
        raise ValueError(
            f"needle matches more than one place in {slot}; include more surrounding text"
        )
    return at


def _locate(slot: Slot, text: str, needle: str) -> tuple[int, int]:
    """Span of the one place *needle* names in *text*.

    Mirrors hermes's memory-tool fixes: a needle with no letter or digit
    (``"."``, ``"\u2014"``) used to select whatever punctuation came first in
    the file; an ambiguous needle silently edited its first occurrence; and a
    correct needle missed when quotes or line wrapping had changed since it
    was stored. Exact matches win; the folded match is only a fallback, and
    both must be unique.
    """
    if not any(c.isalnum() for c in needle):
        raise ValueError(
            "needle must contain a letter or digit — quote part of the entry to change"
        )
    at = _unique_find(slot, text, needle)
    if at is not None:
        return at, at + len(needle)
    folded_text, index = _folded(text)
    folded_needle = _folded(needle.strip())[0]
    at = _unique_find(slot, folded_text, folded_needle)
    if at is not None:
        return index[at], index[at + len(folded_needle) - 1] + 1
    raise MemoryEntryNotFoundError(slot, needle, _closest_lines(text, needle))


class BudgetedMemory:
    """Thin wrapper that enforces char limits and exposes add/replace/remove
    semantics on top of nanobot's per-slot read/write methods."""

    def __init__(self, store: NanobotMemoryStore, budgets: MemoryBudgets) -> None:
        self.store = store
        self.budgets = budgets

    def _budget(self, slot: Slot) -> int:
        return {
            "memory": self.budgets.memory_md_tokens,
            "user": self.budgets.user_md_tokens,
            "soul": self.budgets.soul_md_tokens,
        }[slot]

    def read(self, slot: Slot) -> str:
        if slot == "memory":
            # add/replace/remove read-modify-write this value back to disk, so
            # it must be the raw file text. The load-time guard sanitises
            # store.read_memory (that copy feeds the system prompt) and leaves
            # the unwrapped reader here; editing the sanitised copy would
            # persist "[BLOCKED: …]" over the user's real notes.
            raw = getattr(self.store, RAW_READ_ATTR, None)
            return raw() if callable(raw) else self.store.read_memory()
        if slot == "user":
            return self.store.read_user()
        if slot == "soul":
            return self.store.read_soul()
        raise ValueError(f"unknown slot {slot!r}")

    def write(self, slot: Slot, content: str) -> None:
        limit = self._budget(slot)  # raises on unknown slot before any I/O
        used = _count_tokens(content)
        if used > limit:
            raise MemoryOverBudgetError(slot, used, limit)
        # Bypass MemoryStore.write_* (raw write_text) and write atomically
        # to the path the store would have targeted. A crash mid-write
        # never leaves a half-written MEMORY.md/USER.md/SOUL.md.
        target = {
            "memory": self.store.memory_file,
            "user":   self.store.user_file,
            "soul":   self.store.soul_file,
        }[slot]
        atomic_write_text(target, content)

    def add(self, slot: Slot, entry: str) -> str | None:
        """Add an entry to a memory slot.

        Returns ``None`` on success, ``"duplicate"`` if the entry already
        exists (so the caller can give a friendlier message without raising).
        Raises ``ValueError`` on empty content or guard rejection.
        Raises ``MemoryOverBudgetError`` on budget overflow.
        """
        stripped = entry.strip()
        if not stripped:
            raise ValueError("content must not be empty or whitespace-only")
        err = scan_memory_content(stripped)
        if err:
            raise ValueError(err)
        cur = self.read(slot)
        if stripped in cur:
            return "duplicate"
        joined = (cur + "\n" + stripped) if cur else stripped
        self.write(slot, joined.strip())
        return None

    def replace(self, slot: Slot, needle: str, replacement: str) -> None:
        if not replacement.strip():
            raise ValueError("replacement must not be empty or whitespace-only — use remove() instead")
        err = scan_memory_content(replacement)
        if err:
            raise ValueError(err)
        cur = self.read(slot)
        start, end = _locate(slot, cur, needle)
        self.write(slot, cur[:start] + replacement + cur[end:])

    def remove(self, slot: Slot, needle: str) -> None:
        cur = self.read(slot)
        start, end = _locate(slot, cur, needle)
        new = cur[:start] + cur[end:]
        while "\n\n\n" in new:
            new = new.replace("\n\n\n", "\n\n")
        self.write(slot, new.strip())
