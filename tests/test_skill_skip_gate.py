"""Skip-when gate: a skill's closing "Not for <case>." sentence drops it from
skill_search (Skill-V, arXiv:2610.11781)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

import nano_hermes
from conftest import _make_loop
from nano_hermes.embedding.chain import AllProvidersFailed
from nano_hermes.skills.indexer import SkillIndexer, _split_skip

_TOPICS = ("spreadsheet", "shell", "email")


def _vec(text: str) -> np.ndarray:
    """Unit vector over the topics *text* mentions (bag of topics)."""
    v = np.array([float(t in text.lower()) for t in _TOPICS] + [0.01] * 509, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Chain:
    def __init__(self, seen, down=False):
        self.seen, self.down = seen, down

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def embed(self, texts):
        if self.down:
            raise AllProvidersFailed([])
        self.seen.extend(texts)
        return [_vec(t) for t in texts]


def _write(tmp_path, skills: dict[str, str]) -> list[dict]:
    entries = []
    for name, desc in skills.items():
        p = tmp_path / "skills" / name / "SKILL.md"
        p.parent.mkdir(parents=True)
        p.write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody\n")
        entries.append({"name": name, "path": str(p), "source": "workspace"})
    return entries


def _indexer(tmp_path, skills: dict[str, str], seen: list, down=False):
    hook = nano_hermes.install(_make_loop(tmp_path))
    loader = MagicMock()
    loader.list_skills.return_value = _write(tmp_path, skills)
    return SkillIndexer(db=hook.db, skills_loader=loader,
                        embedder_factory=lambda: _Chain(seen, down))


@pytest.mark.parametrize("desc, split", [
    ("Build spreadsheet reports. Not for shell scripts.",
     ("Build spreadsheet reports.", "shell scripts")),
    ("Build reports", ("Build reports", None)),
    # Only the closing sentence: an earlier "not for" is description.
    ("Not for the faint of heart. Builds reports.", ("Not for the faint of heart. Builds reports.", None)),
    ("Does X. Not for Y. Triggers: foo.", ("Does X. Not for Y. Triggers: foo.", None)),
    ("Build web apps. Not for Node.js or Python 2.7 projects.",
     ("Build web apps.", "Node.js or Python 2.7 projects")),
    # All exclusion: nothing would be left to embed or match.
    ("Not for production use.", ("Not for production use.", None)),
])
def test_split_skip(desc, split):
    assert _split_skip(desc) == split


def test_clause_is_valid_yaml_and_kept_by_nanobot(tmp_path):
    """nanobot parses SKILL.md frontmatter as YAML; "Not for: x" would not be."""
    from nanobot.agent.skills import SkillsLoader

    desc = "Make spreadsheet reports. Not for shell automation."
    _write(tmp_path, {"report": desc})
    assert yaml.safe_load(f"description: {desc}") == {"description": desc}
    meta = SkillsLoader(tmp_path).get_skill_metadata("report")
    assert meta and meta["description"] == desc


async def test_query_matching_the_exclusion_is_dropped(tmp_path):
    seen: list = []
    idx = _indexer(tmp_path, {
        "report": "Make spreadsheet reports. Not for shell automation.",
        "plain": "Make spreadsheet reports.",
    }, seen)
    assert [h.name for h in await idx.search("shell automation", k=1)] == ["plain"]
    # The exclusion's words never reach the skill's own vector.
    assert "report: Make spreadsheet reports." in seen and "shell automation" in seen
    names = [h.name for h in await idx.search("spreadsheet totals")]
    assert set(names) == {"report", "plain"}


async def test_lexical_fallback_ignores_the_exclusion(tmp_path):
    idx = _indexer(tmp_path, {"report": "Make spreadsheet reports. Not for shell automation."},
                   [], down=True)
    with pytest.raises(AllProvidersFailed):
        await idx.search("shell automation")


async def test_unchanged_skill_is_not_reembedded(tmp_path):
    seen: list = []
    idx = _indexer(tmp_path, {"report": "Spreadsheets. Not for email."}, seen)
    await idx.refresh()
    n = len(seen)
    await idx.refresh()
    assert len(seen) == n


async def test_skip_vec_of_other_dims_is_ignored(tmp_path):
    idx = _indexer(tmp_path, {"report": "Spreadsheets. Not for email."}, [])
    await idx.search("email")
    idx._db.execute("UPDATE skill_stats SET skip_vec = ?",
                    (np.ones(8, dtype=np.float32).tobytes(),))
    assert [h.name for h in await idx.search("email")] == ["report"]
