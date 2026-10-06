"""Bounded per-conversation state.

One hook instance serves every chat the loop runs, so anything keyed by
conversation must be capped or a long-lived gateway grows it forever. The
per-conversation dicts share one cap and one recency rule, but each is touched
by its own owner at its own moments, so they can still drop different
conversations — whatever must survive an eviction (a session's ``ended_at``) is
written at eviction time rather than left to another dict's memory.
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
