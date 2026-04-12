"""Security scanning for agent-proposed skill content.

Guards against destructive shell commands, exfiltration payloads, persistence
mechanisms, obfuscated code execution, and prompt-injection phrases in skill
bodies. Every ``propose_skill`` write passes through ``scan_skill_content``
before the SKILL.md is written to disk.

Pattern categories match the threat model in ``memory/guard.py``, plus
skill-specific patterns for shell-level attacks that would be legitimate
in raw memory but dangerous in executable skill instructions.
"""
from __future__ import annotations

import re

from ..memory.guard import _INJECTION_PATTERNS, _INVISIBLE_CODEPOINTS

# --- Destructive shell commands -------------------------------------------
# Skills that include these would execute destructively when an LLM follows them.

_DESTRUCTIVE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("recursive delete", re.compile(r"\brm\s+-[a-z]*r[a-z]*f\b", re.IGNORECASE)),
    ("dangerous chmod", re.compile(r"\bchmod\s+777\b", re.IGNORECASE)),
    ("disk format", re.compile(r"\bmkfs\b", re.IGNORECASE)),
    ("raw disk write", re.compile(r"\bdd\s+if=", re.IGNORECASE)),
]

# --- Persistence mechanisms -----------------------------------------------

_PERSISTENCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("crontab modification", re.compile(r"\bcrontab\b", re.IGNORECASE)),
    ("systemd service enable", re.compile(r"\bsystemctl\s+enable\b", re.IGNORECASE)),
    ("launchctl load", re.compile(r"\blaunchctl\s+load\b", re.IGNORECASE)),
    ("ssh authorized_keys", re.compile(r"authorized_keys", re.IGNORECASE)),
]

# --- Exfiltration commands ------------------------------------------------

_EXFIL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("credential exfiltration via curl", re.compile(
        r"curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)",
        re.IGNORECASE,
    )),
    ("credential exfiltration via wget", re.compile(
        r"wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)",
        re.IGNORECASE,
    )),
    ("netcat reverse shell", re.compile(r"\bnc\s+-e\b", re.IGNORECASE)),
]

# --- Code obfuscation / dynamic execution ---------------------------------

_OBFUSCATION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("base64 pipe decode", re.compile(r"base64\s+-d\s*\|", re.IGNORECASE)),
    ("eval call", re.compile(r"\beval\s*\(", re.IGNORECASE)),
    ("exec call", re.compile(r"\bexec\s*\(", re.IGNORECASE)),
    ("dynamic import", re.compile(r"\b__import__\s*\(", re.IGNORECASE)),
]

_ALL_CATEGORIES: list[tuple[str, list[tuple[str, re.Pattern[str]]]]] = [
    ("prompt injection", _INJECTION_PATTERNS),
    ("destructive shell command", _DESTRUCTIVE_PATTERNS),
    ("persistence mechanism", _PERSISTENCE_PATTERNS),
    ("exfiltration command", _EXFIL_PATTERNS),
    ("obfuscated code execution", _OBFUSCATION_PATTERNS),
]


def scan_skill_content(body: str) -> str | None:
    """Return ``None`` if *body* is safe to write, or an error string.

    Checks for:
    - Prompt injection phrases (shared with memory guard)
    - Destructive shell commands
    - Persistence mechanisms (crontab, launchctl, etc.)
    - Credential exfiltration commands
    - Obfuscated code execution (eval, base64 pipe decode)
    - Invisible unicode characters
    """
    found_invisible = [cp for cp in _INVISIBLE_CODEPOINTS if cp in body]
    if found_invisible:
        names = ", ".join(f"U+{ord(c):04X}" for c in found_invisible)
        return (
            f"skill content rejected — invisible unicode detected: {names}. "
            "These characters are commonly used for hidden prompt injection."
        )

    for category, patterns in _ALL_CATEGORIES:
        for label, pattern in patterns:
            if pattern.search(body):
                return (
                    f"skill content rejected — {category} pattern detected: {label}. "
                    "Remove the flagged content and re-propose."
                )

    return None
