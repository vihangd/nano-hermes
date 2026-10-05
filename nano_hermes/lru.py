"""Bounded per-conversation state.

One hook instance serves every chat the loop runs, so anything keyed by
conversation must be capped or a long-lived gateway grows it forever. All the
per-conversation dicts share one cap and one recency rule, so they cannot evict
different conversations from one another.
"""
from __future__ import annotations

from typing import Callable, TypeVar

K = TypeVar("K")
V = TypeVar("V")

MAX_TRACKED_CONVERSATIONS = 32


def touch(
    d: dict[K, V],
    key: K,
    default: Callable[[], V] | None = None,
    *,
    cap: int = MAX_TRACKED_CONVERSATIONS,
) -> list[K]:
    """Mark *key* most recently used and evict past *cap*; return evicted keys.

    Dicts are insertion-ordered, so re-inserting moves a key to the end. With
    *default*, a missing key is created; without it a missing key is left
    missing and only eviction runs.
    """
    if key in d:
        d[key] = d.pop(key)
    elif default is not None:
        d[key] = default()
    evicted: list[K] = []
    while len(d) > cap:
        oldest = next(iter(d))
        d.pop(oldest)
        evicted.append(oldest)
    return evicted
