"""Skip-when gate: a skill's "Not for:" clause drops it from skill_search
(Skill-V, arXiv:2610.11781)."""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

import nano_hermes
from conftest import _make_loop
from nano_hermes.skills.indexer import SkillIndexer, _content_text, _split_skip

_TOPICS = ("spreadsheet", "shell", "email")


def _vec(text: str) -> np.ndarray:
    """Unit vector over the topics *text* mentions (bag of topics)."""
    v = np.array([float(t in text.lower()) for t in _TOPICS] + [0.01] * 509, dtype=np.float32)
    return v / np.linalg.norm(v)


class _Chain:
    def __init__(self, seen):
        self.seen = seen

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def embed(self, texts):
        self.seen.extend(texts)
        return [_vec(t) for t in texts]


def _indexer(tmp_path, skills: dict[str, str], seen: list):
    hook = nano_hermes.install(_make_loop(tmp_path))
    entries = []
    for name, desc in skills.items():
        p = tmp_path / name / "SKILL.md"
        p.parent.mkdir()
        p.write_text(f"---\nname: {name}\ndescription: {desc}\n---\nbody\n")
        entries.append({"name": name, "path": str(p), "source": "workspace"})
    loader = MagicMock()
    loader.list_skills.return_value = entries
    return hook, SkillIndexer(db=hook.db, skills_loader=loader,
                              embedder_factory=lambda: _Chain(seen))


def test_split_skip():
    assert _split_skip("Build spreadsheet reports. Not for: shell scripts.") == (
        "Build spreadsheet reports.", "shell scripts")
    assert _split_skip("Build reports") == ("Build reports", None)
    assert _content_text("x", "Do A. not for: B") == "x: Do A."


async def test_query_matching_the_exclusion_is_dropped(tmp_path):
    seen: list = []
    _, idx = _indexer(tmp_path, {
        "report": "Make spreadsheet reports. Not for: shell automation.",
        "plain": "Make spreadsheet reports.",
    }, seen)
    names = [h.name for h in await idx.search("shell automation")]
    assert names == ["plain"]
    # The exclusion's words never reach the skill's own vector.
    assert "report: Make spreadsheet reports." in seen and "shell automation" in seen
    names = [h.name for h in await idx.search("spreadsheet totals")]
    assert set(names) == {"report", "plain"}


async def test_unchanged_skill_is_not_reembedded(tmp_path):
    seen: list = []
    _, idx = _indexer(tmp_path, {"report": "Spreadsheets. Not for: email."}, seen)
    await idx.refresh()
    n = len(seen)
    await idx.refresh()
    assert len(seen) == n
