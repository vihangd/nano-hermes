"""Security scanning for memory content writes.

Guards against prompt-injection, exfiltration attempts, and invisible
unicode smuggling. Every ``add`` and ``replace`` call in ``BudgetedMemory``
passes content through ``scan_memory_content`` before writing.

Based on the threat patterns observed in NousResearch/hermes-agent's
``tools/memory_tool._scan_memory_content`` — adapted for nano-hermes.
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger(__name__)

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


def threat_label(text: str) -> str | None:
    """Return a short threat label if *text* trips any guard, else ``None``.

    Shared by the write-time gate (``scan_memory_content``) and the load-time
    sanitiser (``sanitize_loaded_memory``).
    """
    for label, pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            return f"prompt injection: {label}"

    for label, pattern in _EXFIL_PATTERNS:
        if pattern.search(text):
            return f"exfiltration: {label}"

    found = [cp for cp in _INVISIBLE_CODEPOINTS if cp in text]
    if found:
        names = ", ".join(f"U+{ord(c):04X}" for c in found)
        return f"invisible unicode: {names}"

    return None


def scan_memory_content(text: str) -> str | None:
    """Return ``None`` if *text* is safe to write, or an error string.

    Checks for:
    - Prompt injection phrases
    - Credential/file exfiltration commands
    - Invisible unicode characters used for hidden-prompt attacks
    """
    label = threat_label(text)
    if label is None:
        return None
    if label.startswith("prompt injection"):
        return (
            f"content rejected — {label}. "
            "Do not include instruction-overriding phrases in memory."
        )
    if label.startswith("exfiltration"):
        return (
            f"content rejected — potential {label}. "
            "Do not store commands that read credentials or sensitive files."
        )
    return (
        f"content rejected — {label}. "
        "These characters are commonly used for hidden prompt injection."
    )


def sanitize_loaded_memory(text: str) -> tuple[str, list[str]]:
    """Sanitise memory text at prompt-load time.

    The write-time gate (``scan_memory_content``) only fires on the agent's
    own ``memory_patch`` writes. MEMORY.md is a plain file: a direct on-disk
    edit, a sync, or a restore from an older DB can reintroduce a poisoned
    entry that never passed the gate. This scans each line at the moment it
    would enter the system prompt and replaces any offending line with a
    ``[BLOCKED: …]`` placeholder, leaving the on-disk file untouched so the
    user can still inspect and remove it.

    The threat patterns use ``\\s+``, which spans newlines, so an attacker
    could split a phrase across lines to dodge the per-line pass. After it,
    we re-scan the joined result; if a cross-line pattern still fires, the
    whole block is blocked wholesale (safe failure mode).

    SOUL.md / USER.md are out of scope here: they're injected via nanobot's
    separate ``_load_bootstrap_files`` path (not through the store), are
    user-authored identity files changed rarely, and wrapping that private
    builder method would be version-fragile. The agent's own writes to those
    slots are still covered by the write-time gate.

    Returns ``(sanitised_text, reasons)`` where *reasons* lists the threat
    labels that were blocked (empty when nothing was found).
    """
    reasons: list[str] = []
    out: list[str] = []
    for line in text.splitlines():
        if not line.strip():
            out.append(line)
            continue
        label = threat_label(line)
        if label is None:
            out.append(line)
        else:
            reasons.append(label)
            out.append(f"[BLOCKED: {label}]")
    clean = "\n".join(out)
    # Backstop: a multi-line injection whose individual lines look benign
    # survives the per-line pass but still trips the newline-spanning blob
    # patterns. If so, block the entire block rather than emit it.
    residual = threat_label(clean)
    if residual is not None:
        reasons.append(residual)
        return f"[BLOCKED: {residual}]", reasons
    return clean, reasons


def install_loadtime_memory_scan(store: object) -> None:
    """Wrap a nanobot ``MemoryStore.get_memory_context`` so the MEMORY.md
    block is sanitised at the moment it enters the system prompt.

    Only the prompt-injection path is wrapped — the agent's own read/edit
    path (``read_memory``) is left untouched so ``memory_patch`` still sees
    and edits the real on-disk content. Idempotent: a second call is a no-op.
    """
    original = getattr(store, "get_memory_context", None)
    if original is None or getattr(original, "_nh_loadtime_scan", False):
        return

    def wrapped() -> str:
        text = original()
        if not text:
            return text
        clean, reasons = sanitize_loaded_memory(text)
        if reasons:
            log.warning(
                "nano-hermes: blocked %d poisoned memory line(s) at load: %s",
                len(reasons),
                "; ".join(reasons),
            )
        return clean

    wrapped._nh_loadtime_scan = True  # type: ignore[attr-defined]
    store.get_memory_context = wrapped  # type: ignore[attr-defined]
