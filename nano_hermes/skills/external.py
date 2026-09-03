"""Discovery for read-only external skill directories.

External dirs are scanned on each indexer refresh. Scan verdicts are memoised
on a hash of the file's content (see ``_cached_scan``) because discovery sits on
the foreground ``skill_search`` path.

Everything discovered here is **untrusted input on a read path**. An external
dir is a shared folder, a synced drive, or a cloned skills repo — exactly the
supply-chain position where hostile content arrives without anyone writing it
through ``propose_skill``. The write paths (propose / GEPA / rewriter /
designer) all gate on ``scan_skill_content``; applying the same scanner here
closes the asymmetry, mirroring the load-time MEMORY.md scan. The source file
is never modified — these dirs are read-only by contract.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n(.+?)\n---\n", re.DOTALL)
_KEY_VAL_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
_EXCLUDED_DIR_NAMES = frozenset({".git", ".github", ".hub"})

# Scan verdicts keyed by content hash -> verdict. Discovery runs on every
# indexer refresh, which sits on the foreground skill_search path, where the
# regex sweep measured ~78% of discovery cost; re-scanning unchanged files each
# turn is pure waste on an SD card.
#
# Keyed on the CONTENT, not (mtime, size): this memoises a *security* verdict,
# and anyone able to write to a shared/synced external dir — precisely the
# threat model above — can preserve mtime (`touch -r`, `rsync --times`) and pad
# to an identical size, which would re-serve a stale "safe" verdict for
# poisoned content. The file text is already in memory at the call site, so
# hashing it is effectively free.
_SCAN_CACHE: dict[str, str | None] = {}
# Bounded so a long-lived process scanning many rotating dirs cannot grow it
# without limit; external skill sets are small, so this is never reached in
# normal use.
_SCAN_CACHE_MAX = 512


def expand_external_dirs(raw_paths: list[str]) -> list[Path]:
    """Expand ``~`` and ``${VAR}`` per entry; return existing dirs only.

    Logs a warning for missing or non-dir entries; never raises.
    Deduplicates by resolved path.
    """
    out: list[Path] = []
    seen: set[Path] = set()
    for entry in raw_paths:
        s = (entry or "").strip()
        if not s:
            continue
        expanded = os.path.expandvars(os.path.expanduser(s))
        p = Path(expanded).resolve()
        if p in seen:
            continue
        if not p.is_dir():
            log.warning(
                "external skills dir does not exist or is not a directory: %s", p
            )
            continue
        seen.add(p)
        out.append(p)
    return out


def discover_external_skills(dirs: list[Path]) -> list[dict[str, Any]]:
    """Walk each dir; return one entry per SKILL.md found.

    Entry shape mirrors ``SkillsLoader.list_skills``::

        {"name":        <frontmatter or dir name>,
         "path":        <absolute path to SKILL.md>,
         "source":      "external",
         "description": <frontmatter description, or "">}

    - Excludes ``.git``, ``.github``, ``.hub`` subdirs.
    - Unparseable frontmatter → debug log + skip.
    - Duplicate names within a scan: first-wins by lexicographic path.
    - Content failing ``scan_skill_content`` → warning + skip. A skill whose
      body or description carries an injection/exfiltration pattern is dropped
      whole rather than redacted: unlike a memory note, a skill has no value
      once its instructions are untrustworthy.
    """
    from .guard import scan_skill_content  # noqa: PLC0415  (avoids an import cycle)
    entries: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for root in dirs:
        for skill_md in sorted(root.rglob("SKILL.md")):
            if any(part in _EXCLUDED_DIR_NAMES for part in skill_md.parts):
                continue
            try:
                content = skill_md.read_text(encoding="utf-8")
                frontmatter = _parse_frontmatter(content)
            except Exception as e:
                log.debug("could not parse %s: %s", skill_md, e)
                continue
            name = frontmatter.get("name") or skill_md.parent.name
            if name in seen_names:
                continue
            # Scan the whole document: the description reaches the model via
            # skill_search output, and the body reaches it when the skill loads.
            threat = _cached_scan(content, scan_skill_content)
            if threat is not None:
                log.warning(
                    "external skill %r from %s rejected: %s", name, skill_md, threat
                )
                seen_names.add(name)
                continue
            description = frontmatter.get("description", "")
            seen_names.add(name)
            entries.append({
                "name": name,
                "path": str(skill_md),
                "source": "external",
                "description": description,
            })
    return entries


def _cached_scan(content: str, scanner) -> str | None:
    """``scanner(content)`` memoised on a hash of *content*.

    Identical bytes always yield the identical verdict, so this cannot serve a
    stale answer for changed content the way a timestamp key can.
    """
    key = hashlib.sha256(content.encode("utf-8", "surrogatepass")).hexdigest()
    if key in _SCAN_CACHE:
        return _SCAN_CACHE[key]
    verdict = scanner(content)
    if len(_SCAN_CACHE) >= _SCAN_CACHE_MAX:
        _SCAN_CACHE.clear()
    _SCAN_CACHE[key] = verdict
    return verdict


def _parse_frontmatter(content: str) -> dict[str, str]:
    """Minimal regex-based parser. Strips surrounding single/double quotes
    on values. Skips multiline / nested YAML — callers fall back to the
    skill's directory name when ``name`` is absent.
    """
    m = _FRONTMATTER_RE.match(content)
    if not m:
        return {}
    out: dict[str, str] = {}
    for km in _KEY_VAL_RE.finditer(m.group(1)):
        out[km.group(1).strip()] = km.group(2).strip().strip("'\"")
    return out
