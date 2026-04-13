"""Structural smoke tests for examples/skills/skill-creator.

These are NOT behavioural tests — they don't run the skill through the
agent. They just verify that the files we ship under examples/skills/
would pass the same validation nanobot applies if a user copied them
into their workspace:

- Frontmatter is parseable and matches nanobot's quick_validate.py whitelist
- Referenced files exist
- Body actually points at the references (so the agent reads them)
- Root-dir layout stays within the skill whitelist
- Total size is under the propose_skill max_skill_bytes cap

Catches drift between the plan's structure and the shipped files.
"""
from __future__ import annotations

import re
from pathlib import Path

SKILL_DIR = Path(__file__).parent.parent / "examples" / "skills" / "skill-creator"
SKILL_MD = SKILL_DIR / "SKILL.md"
AUTHORING_GUIDE = SKILL_DIR / "references" / "authoring-guide.md"
QUALITY_CHECKLIST = SKILL_DIR / "references" / "quality-checklist.md"

# Matches nanobot's quick_validate.py whitelist at lines 17-24.
_ALLOWED_FRONTMATTER_KEYS = {
    "name",
    "description",
    "metadata",
    "always",
    "license",
    "allowed-tools",
}
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_MAX_SKILL_BYTES = 256 * 1024
_MAX_DESCRIPTION_CHARS = 1024


def _parse_frontmatter(text: str) -> dict:
    """Minimal YAML-ish frontmatter parser matching nanobot's fallback."""
    assert text.startswith("---"), "SKILL.md must start with --- frontmatter"
    end = text.find("\n---", 3)
    assert end != -1, "SKILL.md frontmatter must be closed with ---"
    raw = text[4:end].strip()
    out: dict = {}
    for line in raw.splitlines():
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip()] = value.strip()
    return out


class TestFrontmatter:
    def test_skill_md_exists(self):
        assert SKILL_MD.exists(), f"SKILL.md missing at {SKILL_MD}"

    def test_frontmatter_parses(self):
        fm = _parse_frontmatter(SKILL_MD.read_text())
        assert "name" in fm
        assert "description" in fm

    def test_name_matches_directory(self):
        fm = _parse_frontmatter(SKILL_MD.read_text())
        assert fm["name"] == "skill-creator"
        assert fm["name"] == SKILL_DIR.name

    def test_name_is_hyphen_case(self):
        fm = _parse_frontmatter(SKILL_MD.read_text())
        assert _NAME_RE.match(fm["name"]), f"name {fm['name']!r} must be hyphen-case"

    def test_description_length_within_cap(self):
        fm = _parse_frontmatter(SKILL_MD.read_text())
        assert 1 <= len(fm["description"]) <= _MAX_DESCRIPTION_CHARS

    def test_description_leads_with_trigger(self):
        """The description should follow its own 'Use when…' advice.

        Accepts 'Use when', 'Use this when', 'Read this first', or similar —
        the key is that it describes WHEN to invoke, not just what it does.
        """
        fm = _parse_frontmatter(SKILL_MD.read_text())
        desc = fm["description"].lower()
        triggers = ("use when", "use this when", "read this", "call when")
        assert any(t in desc for t in triggers), (
            f"description must lead with a trigger phrase (one of {triggers}); "
            f"got: {fm['description']!r}"
        )

    def test_frontmatter_keys_are_whitelisted(self):
        fm = _parse_frontmatter(SKILL_MD.read_text())
        extra = set(fm) - _ALLOWED_FRONTMATTER_KEYS
        assert not extra, (
            f"frontmatter contains keys not in nanobot's whitelist: {extra}. "
            f"Allowed: {_ALLOWED_FRONTMATTER_KEYS}"
        )


class TestReferences:
    def test_authoring_guide_exists_and_nonempty(self):
        assert AUTHORING_GUIDE.exists()
        assert AUTHORING_GUIDE.stat().st_size > 0

    def test_quality_checklist_exists_and_nonempty(self):
        assert QUALITY_CHECKLIST.exists()
        assert QUALITY_CHECKLIST.stat().st_size > 0

    def test_body_points_at_authoring_guide(self):
        """The SKILL.md body must explicitly tell the agent to read the guide."""
        body = SKILL_MD.read_text()
        assert "references/authoring-guide.md" in body

    def test_body_points_at_quality_checklist(self):
        body = SKILL_MD.read_text()
        assert "references/quality-checklist.md" in body


class TestRootDirWhitelist:
    """Nanobot's quick_validate.py allows only SKILL.md + {scripts,references,assets}/ at the root."""

    def test_only_whitelisted_entries_at_root(self):
        entries = {p.name for p in SKILL_DIR.iterdir()}
        allowed_dirs = {"scripts", "references", "assets"}
        allowed_files = {"SKILL.md"}
        allowed = allowed_dirs | allowed_files
        extra = entries - allowed
        assert not extra, (
            f"skill-creator root contains entries outside the whitelist: {extra}"
        )

    def test_has_at_least_skill_md_and_references(self):
        entries = {p.name for p in SKILL_DIR.iterdir()}
        assert "SKILL.md" in entries
        assert "references" in entries


class TestSizeCap:
    def test_total_size_under_propose_skill_cap(self):
        """Total bytes across SKILL.md and references/ must fit the 256 KiB cap.

        This is what the user would hit if they ever self-edit this skill via
        propose_skill — not a strict requirement for shipping, but a useful
        sanity check.
        """
        total = 0
        for p in SKILL_DIR.rglob("*"):
            if p.is_file():
                total += p.stat().st_size
        assert total <= _MAX_SKILL_BYTES, (
            f"skill-creator total size {total:,} bytes exceeds propose_skill "
            f"cap of {_MAX_SKILL_BYTES:,} bytes"
        )

    def test_skill_md_body_under_500_lines(self):
        """The skill-creator's own SKILL.md must follow the size discipline it preaches."""
        lines = SKILL_MD.read_text().splitlines()
        assert len(lines) <= 500, (
            f"skill-creator SKILL.md has {len(lines)} lines; should stay "
            "under 500 (the hard cap the skill itself teaches)"
        )
