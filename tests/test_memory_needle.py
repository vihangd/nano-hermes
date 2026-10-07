"""memory_patch replace/remove: which text a needle selects.

Mirrors the hermes memory-tool fixes (b9013584b2, 2983c25373, 8079fd01f4):
the store is edited by quoting part of an entry, so a needle that selects the
wrong text — or misses the right one over a quote style — corrupts or blocks
memory writes.
"""
from __future__ import annotations

from nanobot.agent.loop import AgentLoop

import nano_hermes

MEMORY = (
    "User's dog is called “Biscuit” — a beagle.\n"
    "Prefers tea.\n"
    "Works on the   nano-hermes\n  project.\n"
    "Prefers tea in the morning."
)


async def _patch(loop: AgentLoop, **kwargs) -> tuple[str, str]:
    nano_hermes.install(loop)
    loop.context.memory.write_memory(MEMORY)
    out = await loop.tools.get("memory_patch").execute(slot="memory", **kwargs)
    return out, loop.context.memory.read_memory()


async def test_a_punctuation_only_needle_edits_nothing(loop: AgentLoop) -> None:
    out, mem = await _patch(loop, action="remove", needle="—")
    assert out.startswith("Error") and "letter or digit" in out
    assert mem == MEMORY


async def test_an_ambiguous_needle_edits_nothing(loop: AgentLoop) -> None:
    out, mem = await _patch(
        loop, action="replace", needle="Prefers tea", replacement="Prefers coffee."
    )
    assert out.startswith("Error") and "2 places" in out
    assert mem == MEMORY


async def test_straight_quotes_find_a_curly_quoted_entry(loop: AgentLoop) -> None:
    out, mem = await _patch(
        loop,
        action="replace",
        needle='dog is called "Biscuit" - a beagle',
        replacement="dog is called Biscuit, a basset hound",
    )
    assert out.startswith("ok"), out
    assert "basset hound." in mem and "beagle" not in mem
    assert mem.count("Prefers tea") == 2  # nothing else touched


async def test_rewrapped_whitespace_still_matches(loop: AgentLoop) -> None:
    out, mem = await _patch(
        loop, action="remove", needle="Works on the nano-hermes project."
    )
    assert out.startswith("ok"), out
    assert "nano-hermes" not in mem and "Biscuit" in mem


async def test_a_miss_names_the_closest_stored_line(loop: AgentLoop) -> None:
    out, mem = await _patch(
        loop, action="remove", needle="dog named Biscuit the beagle"
    )
    assert out.startswith("Error") and "closest stored lines" in out
    assert "Biscuit" in out
    assert mem == MEMORY
