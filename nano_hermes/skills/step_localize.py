"""AgentPRM-lite step localization (Phase 8).

Before rewriting a chronically failing skill, ask an LLM to read the
current SKILL.md and the failure-context chunks and produce a one-line
critique identifying *which step* fails and *why*. The critique is
spliced into the rewrite prompt so the rewriter focuses on the actual
bug rather than rewriting blindly from "this skill fails a lot".

Cheap variant of AgentPRM (arXiv 2511.08325): no trained reward model,
just one LLM judge call. Returns None on any failure — the rewrite path
falls through to the legacy "no localization" flow.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)


_LOCALIZE_PROMPT = """\
You are localizing a defect in an AI agent skill.

SKILL NAME: {skill_name}

CURRENT SKILL.md:
---
{current_body}
---

FAILURE TRACES (most recent first):
---
{failure_context}
---

In ONE sentence (≤ 35 words), identify WHICH step or instruction in the
skill is the proximate cause of the failures, and what specifically goes
wrong there. Do NOT propose a fix — only localize the defect.

Output: just the sentence, no preamble, no bullet points, no quotation marks.
"""


async def localize_failure_step(
    hook: "NanoHermesHook",
    *,
    skill_name: str,
    current_body: str,
    failure_context: str,
    max_tokens: int = 120,
) -> str | None:
    """Return a one-sentence critique of where the skill fails, or None.

    Best-effort: any provider or parse failure returns None so the
    rewrite path can continue without localization.
    """
    provider = getattr(hook._loop, "provider", None)
    if provider is None:
        return None
    model = getattr(hook._loop, "model", None)
    if model is None:
        return None
    prompt = _LOCALIZE_PROMPT.format(
        skill_name=skill_name,
        current_body=current_body,
        failure_context=failure_context,
    )
    try:
        resp = await provider.chat_with_retry(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=max_tokens,
        )
        text = (resp.content or "").strip()
    except Exception:
        log.debug("step_localize: provider call failed", exc_info=True)
        return None
    if not text:
        return None
    # Take only the first non-empty line so a chatty model doesn't sneak
    # multi-paragraph output into the rewrite prompt.
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    if not first:
        return None
    # Strip surrounding quote marks the model may have added despite
    # being asked not to.
    if first.startswith('"') and first.endswith('"') and len(first) > 2:
        first = first[1:-1]
    return first[:240]
