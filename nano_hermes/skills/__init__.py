"""Skill indexing + semantic retrieval on top of nanobot's SkillsLoader.

Nanobot's own ``nanobot.agent.skills.SkillsLoader`` handles discovery,
progressive-disclosure injection, and requirement checks. nano-hermes
layers on:

- Embedding-indexed retrieval (``SkillIndexer`` + ``SkillSearchTool``).
- Mutable state sidecar — use_count, success_count, status, provenance —
  in the ``skill_stats`` table, read/written during reflection and
  two-phase promotion. (Phase 2.)
"""
from .indexer import SkillHit, SkillIndexer
from .tool import SkillSearchTool

__all__ = ["SkillHit", "SkillIndexer", "SkillSearchTool"]
