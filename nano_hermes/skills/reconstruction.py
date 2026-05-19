"""MIND-Skill-inspired reconstruction check for draft→active promotion gate.

Before a draft skill is promoted to active, an independent LLM call is made
to verify that the skill body actually implements what its description claims.
This prevents low-quality or misnamed skills from polluting the active index.

Safety: the audit prompt is a module-level constant, separate from the skill
content being evaluated.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..hook import NanoHermesHook

log = logging.getLogger(__name__)

_AUDIT_SYSTEM = """\
You are a strict AI skill-quality auditor. Your job is to check whether a \
skill's body actually implements what its description claims.

Answer with EXACTLY one word: YES (the body matches the description) or NO \
(the body is about something else, or is too vague to deliver the description).\
"""

_AUDIT_PROMPT = """\
SKILL DESCRIPTION (stated purpose):
{description}

SKILL BODY:
---
{body}
---

Does the skill body implement what the description claims? Answer YES or NO.\
"""

# Minimum body character length to run the check. Very short bodies are likely
# stubs and would produce unreliable YES/NO judgements.
_MIN_BODY_CHARS = 80


async def check_reconstruction(
    hook: "NanoHermesHook",
    *,
    skill_name: str,
    description: str,
    body: str,
) -> bool:
    """Return True if the LLM judges the body consistent with the description.

    Retries once on network failure; returns True (allow promotion) on both
    failures so that a transient LLM outage never permanently blocks a skill.
    """
    if len(body.strip()) < _MIN_BODY_CHARS:
        log.debug("reconstruction: %s — body too short, skipping check", skill_name)
        return True

    provider = getattr(hook._loop, "provider", None)
    if provider is None:
        log.debug("reconstruction: %s — no LLM provider, skipping check", skill_name)
        return True

    prompt = _AUDIT_PROMPT.format(description=description, body=body[:4000])
    for attempt in range(2):
        try:
            resp = await provider.chat_with_retry(
                messages=[
                    {"role": "system", "content": _AUDIT_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                model=getattr(hook._loop, "model", None),
                max_tokens=5,
            )
            answer = (resp.content or "").strip().upper()
            if answer.startswith("YES"):
                return True
            if answer.startswith("NO"):
                log.info(
                    "reconstruction: %s — body does not match description, blocking promotion",
                    skill_name,
                )
                return False
            # Ambiguous response — treat as pass (don't punish for model quirks).
            log.warning(
                "reconstruction: %s — ambiguous response %r, allowing promotion",
                skill_name,
                answer,
            )
            return True
        except Exception:
            log.warning(
                "reconstruction: LLM call failed for %s (attempt %d/2)",
                skill_name,
                attempt + 1,
                exc_info=True,
            )
    # Both attempts failed — fail open so transient outages don't block skills.
    log.warning("reconstruction: gave up on %s after 2 failures — allowing promotion", skill_name)
    return True
