"""Contextual preamble injection for session chunk embeddings.

Prepending a short task-context string to each chunk before embedding
(not to the stored content) lets the vector model encode *what the
conversation was about* alongside the chunk text.  The stored
``chunks.content`` column remains unchanged — only the vector changes.

This mirrors the technique described in Anthropic's contextual retrieval
post (+49% recall).  No LLM call is needed: the task is the first user
message of the session, already available in the archiver.
"""
from __future__ import annotations

_MAX_TASK_CHARS = 200  # keep preamble short to avoid drowning the chunk


def add_context_preamble(text: str, task: str | None) -> str:
    """Return ``text`` prefixed with a task preamble if *task* is available.

    The result is used *only* for embedding — never stored.
    """
    if not task:
        return text
    short_task = task[:_MAX_TASK_CHARS].strip()
    if not short_task:
        return text
    return f"Task: {short_task}\n\n{text}"
