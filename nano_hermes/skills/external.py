"""Discovery for read-only external skill directories.

External dirs are scanned on each indexer refresh — there's no caching.
Each scan is cheap: a single rglob over a typically-small file tree. If
perf matters later we can add an mtime-based skip.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\n(.+?)\n---\n", re.DOTALL)
_KEY_VAL_RE = re.compile(r"^(\w+):\s*(.+)$", re.MULTILINE)
_EXCLUDED_DIR_NAMES = frozenset({".git", ".github", ".hub"})


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
    """
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
            description = frontmatter.get("description", "")
            seen_names.add(name)
            entries.append({
                "name": name,
                "path": str(skill_md),
                "source": "external",
                "description": description,
            })
    return entries


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
