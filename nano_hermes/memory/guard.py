"""Security scanning for memory content writes.

Guards against prompt-injection, exfiltration attempts, and invisible
unicode smuggling. Every ``add`` and ``replace`` call in ``BudgetedMemory``
passes content through ``scan_memory_content`` before writing.

Based on the threat patterns observed in NousResearch/hermes-agent's
``tools/memory_tool._scan_memory_content`` — adapted for nano-hermes.
"""
from __future__ import annotations

import re

# --- Prompt injection patterns -------------------------------------------

_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ignore instructions", re.compile(
        r"ignore\s+(previous|all|above)\s+instructions", re.IGNORECASE
    )),
    ("role hijack", re.compile(r"you\s+are\s+now\s+", re.IGNORECASE)),
    ("deception directive", re.compile(
        r"do\s+not\s+tell\s+the\s+user", re.IGNORECASE
    )),
    ("system prompt override", re.compile(
        r"system\s+prompt\s+override", re.IGNORECASE
    )),
    ("disregard rules", re.compile(
        r"disregard\s+your\s+(instructions|rules|guidelines)", re.IGNORECASE
    )),
    ("no restrictions", re.compile(
        r"act\s+as\s+if\s+you\s+have\s+no\s+(restrictions|limits)", re.IGNORECASE
    )),
]

# --- Exfiltration patterns ------------------------------------------------

_EXFIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("credential exfiltration via curl/wget", re.compile(
        r"(curl|wget).*\$(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        re.IGNORECASE | re.DOTALL,
    )),
    ("sensitive file read", re.compile(
        r"cat\s+.*(\.(env|netrc|pgpass|npmrc|pypirc)|credentials)",
        re.IGNORECASE,
    )),
    ("authorized_keys access", re.compile(r"authorized_keys", re.IGNORECASE)),
    ("ssh directory access", re.compile(r"[/~]\.ssh", re.IGNORECASE)),
]

# --- Invisible unicode detection ------------------------------------------
# These characters are used for prompt injection via hidden content.

_INVISIBLE_CODEPOINTS = frozenset([
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # BOM / zero-width no-break space
    "\u202a",  # left-to-right embedding
    "\u202b",  # right-to-left embedding
    "\u202c",  # pop directional formatting
    "\u202d",  # left-to-right override
    "\u202e",  # right-to-left override (classic injection vector)
])


def scan_memory_content(text: str) -> str | None:
    """Return ``None`` if *text* is safe to write, or an error string.

    Checks for:
    - Prompt injection phrases
    - Credential/file exfiltration commands
    - Invisible unicode characters used for hidden-prompt attacks
    """
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return (
                f"content rejected — prompt injection pattern detected: {label}. "
                "Do not include instruction-overriding phrases in memory."
            )

    for label, pattern in _EXFIL_PATTERNS:
        if pattern.search(text):
            return (
                f"content rejected — potential exfiltration pattern detected: {label}. "
                "Do not store commands that read credentials or sensitive files."
            )

    found = [cp for cp in _INVISIBLE_CODEPOINTS if cp in text]
    if found:
        names = ", ".join(f"U+{ord(c):04X}" for c in found)
        return (
            f"content rejected — invisible unicode detected: {names}. "
            "These characters are commonly used for hidden prompt injection."
        )

    return None
