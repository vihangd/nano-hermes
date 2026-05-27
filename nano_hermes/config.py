"""Pydantic config schema for nano-hermes.

Trimmed to only what we own. Paths are derived from ``loop.workspace``,
not configured here, so there's no overlap with nanobot's own config.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EmbeddingProvider(BaseModel):
    provider: Literal["deepinfra", "together", "openrouter"]
    api_key_env: str
    base_url: str | None = None  # override the default OpenAI-compat endpoint


class EmbeddingConfig(BaseModel):
    model: str = "BAAI/bge-m3"
    native_dims: int = 1024
    target_dims: int = Field(default=512, gt=0)
    normalize_after_truncate: bool = True
    timeout_seconds: float = 20.0
    # How long (seconds) to skip a provider after an HTTP 402/429/5xx error.
    unhealthy_ttl_seconds: float = 600.0
    chain: list[EmbeddingProvider] = Field(
        default_factory=lambda: [
            EmbeddingProvider(provider="deepinfra", api_key_env="DEEPINFRA_API_KEY"),
            EmbeddingProvider(provider="together", api_key_env="TOGETHER_API_KEY"),
            EmbeddingProvider(provider="openrouter", api_key_env="OPENROUTER_API_KEY"),
        ]
    )


class MemoryBudgets(BaseModel):
    """Token budgets enforced by ``BudgetedMemory`` on writes.

    Nanobot's underlying MemoryStore does not enforce these — we do, on
    behalf of the agent's ``memory_patch`` tool. Writes from nanobot's
    Dream / Consolidator path bypass budgets intentionally: curated
    background writes get latitude.

    Budgets are counted in tokens (cl100k_base encoding) rather than chars
    so that CJK or emoji-heavy content doesn't silently inflate prompt cost.
    Approximate equivalents: 512 tokens ≈ 2000 English chars.
    """
    memory_md_tokens: int = 512
    user_md_tokens: int = 320
    soul_md_tokens: int = 384
    # Cosine similarity threshold for memory_patch(action="consolidate").
    # Entries with similarity >= threshold are merged (longest survives).
    # 0.92 catches near-duplicates while preserving meaningfully distinct entries.
    consolidation_similarity_threshold: float = 0.92
    # Episodic→semantic distillation (memory_patch action="distill").
    # A chunk cluster must span this many distinct successful sessions to surface.
    distill_hub_min_sessions: int = 2
    # Cap the chunk pool before O(N²) clustering (Pi 3B+ memory budget).
    distill_max_chunks: int = 150
    # Cosine similarity threshold for hub clustering. Tighter than consolidation
    # (0.92) — hubs are genuinely recurring topics, not just near-duplicate text.
    distill_cluster_threshold: float = 0.88
    # When True, call the LLM to distill each hub into a semantic fact and
    # write the result to semantic_facts. When False, surface hubs to the
    # agent for manual review only (no LLM call, no DB write).
    distill_llm_enabled: bool = True


class RetrievalConfig(BaseModel):
    fts_k: int = Field(default=25, gt=0)       # FTS5 candidate pool
    vec_k: int = Field(default=25, gt=0)       # vector candidate pool
    rrf_k: int = 60       # RRF smoothing constant (Cormack et al. default)
    final_k: int = Field(default=8, gt=0)
    # MMR diversity reranking applied after RRF fusion.
    # λ=1.0 disables MMR (pure relevance). λ=0.7 recommended balance.
    mmr_lambda: float = 0.7


class ReflectionConfig(BaseModel):
    """Reflexion nudge trigger + injection limits.

    See ``nano_hermes/reflect/salience.py`` for how the score is computed.
    Default threshold of 5.0 fires on ANY of: one error, one tool-burst
    iteration, three user-correction phrases, or a mix that adds up.
    """
    threshold: float = 5.0
    recent_limit: int = 5   # max reflections injected per iteration
    # Minimum cosine similarity (1 - distance) for global cross-session
    # reflection injection. Prevents injecting unrelated reflections when
    # reflection_scope = "global".
    global_inject_min_similarity: float = 0.60


class SkillStatsConfig(BaseModel):
    """Config for skill usage stat tracking."""
    min_uses_for_success_rate: int = 3  # Don't display rate below this threshold
    # Ranking mode for skill_search re-ranking.
    #   "ucb1"         — UCB1 bandit (exploration + exploitation)
    #   "stat_weighted" — pure success-rate boost (legacy)
    #   "off"          — no stat adjustment
    ranking_mode: Literal["ucb1", "stat_weighted", "off"] = "ucb1"
    # UCB1: how much the bandit score shifts effective_distance.
    # Typical L2 gap between skills is ~0.1–1.4; default 0.05 gives a max
    # cold-start adjustment of ~0.12 (meaningful without overwhelming).
    ucb1_coefficient: float = 0.05
    # Legacy stat_weighted fields (still used when ranking_mode="stat_weighted").
    use_stat_weighting: bool = True
    success_rate_boost: float = 0.3
    # Phase 4: two-phase skill promotion thresholds.
    promotion_threshold: int = Field(default=3, gt=0)  # successful uses to promote draft -> active
    deprecation_min_uses: int = 5         # minimum uses before deprecation check
    deprecation_max_success_rate: float = 0.2  # below this rate -> deprecated
    # Maximum total bytes (body + companion files) allowed per propose_skill call.
    max_skill_bytes: int = 256 * 1024  # 256 KiB
    # Cosine similarity threshold for the diversity gate at draft→active promotion.
    # A draft skill whose embedding is >= this similar to ANY active skill is blocked
    # from promotion (FactorMiner insight: uncurated duplicates hurt retrieval quality).
    # 0.88 blocks near-duplicates while allowing meaningfully distinct variants.
    diversity_similarity_threshold: float = 0.88
    # Search-time greedy diversity dedup (GoSkills pattern).
    # After ranking, hits with cosine similarity >= threshold to a higher-ranked
    # kept hit are suppressed and surfaced as siblings. 0.82 keeps semantically
    # distinct skills while collapsing near-identical variants.
    skill_search_dedup_threshold: float = 0.82
    # GEPA (Genetic-Pareto Prompt Evolution) — iterative skill text evolution.
    # Off by default; enable once you have ≥5 sessions worth of failure data.
    # Each run costs 2 LLM calls × gepa_max_mutations rounds per eligible skill.
    gepa_enabled: bool = False
    # Lower threshold than rewrite_failure_threshold — GEPA runs as a gentler
    # first pass; severe failures fall through to the SkillForge rewriter.
    gepa_failure_threshold: float = 0.4
    gepa_min_uses: int = 5               # minimum uses before GEPA trigger
    gepa_max_mutations: int = 3           # Pareto evolution rounds per skill
    gepa_minibatch_size: int = 3          # failed examples used per evaluation
    # Phase 3.1: failure-driven skill rewriter thresholds.
    # Skills with failure_rate > rewrite_failure_threshold AND use_count >= rewrite_min_uses
    # are candidates for automatic rewriting.
    rewrite_failure_threshold: float = 0.6  # >60% failure rate
    rewrite_min_uses: int = 5               # minimum uses before rewrite trigger
    rewrite_context_chunks: int = 5         # how many failed-session chunks to send the LLM
    # Auto-evolution trigger: run GEPA (if enabled) then rewriter every N completed sessions.
    # 0 = disabled (off by default). Recommended starting value: 5–10.
    rewrite_session_interval: int = 0
    # SkillForge critic: second independent LLM call that must approve a rewrite
    # before it is committed (covers-use-case, avoids-cited-failure, not-overfit).
    # Default ON — adds one LLM call per rewrite candidate but prevents regressions.
    rewrite_critic_enabled: bool = True
    # AgentPRM-lite (arXiv 2511.08325): localize the failing step with one
    # LLM judge call before rewriting; the resulting one-liner is spliced
    # into the rewrite prompt. Default ON — one extra cheap call per rewrite.
    rewrite_step_localization_enabled: bool = True
    # ASG-SI replay gate (arXiv 2512.23760): after the critic approves, replay
    # the last N failing trajectories against the new body via an LLM judge.
    # Default ON. min_pass_rate is the fraction of replays that must judge
    # IMPROVED (any single WORSE verdict vetoes regardless of rate).
    rewrite_replay_gate_enabled: bool = True
    rewrite_replay_min_pass_rate: float = 0.6
    rewrite_replay_max_trajectories: int = 3
    # MIND-Skill reconstruction check: before draft→active promotion, ask an LLM
    # to verify the skill body actually implements what the description claims.
    # Fails open on LLM error (promotion allowed). Default ON.
    reconstruction_check_enabled: bool = True
    # Phase 4.4: minimum distinct sessions before a skill is exported for
    # offline GEPA/MIPROv2 optimisation. Kept high (50) so early-corpus noise
    # doesn't pollute the training set.
    export_min_sessions: int = 50


class SkillsConfig(BaseModel):
    """Skill-discovery overlay on top of nanobot's SkillsLoader.

    nano-hermes still uses ``SkillsLoader`` for workspace + builtin skills.
    This block adds N read-only external dirs whose SKILL.md files are
    discovered and indexed parallel to builtin (status='active' on first
    index). External skills are immutable from propose_skill — copy to
    workspace first to modify.

    Entries support ``~`` and ``${VAR}`` expansion; missing dirs are
    logged and skipped (never raised, so a typo doesn't break startup).
    """
    external_dirs: list[str] = Field(default_factory=list)


class TrajectoryConfig(BaseModel):
    """Config for Phase 3 trajectory replay."""
    # Inject the top-1 matching trajectory into before_iteration context.
    # Off by default — opt-in since it adds tokens on every turn.
    inject_context: bool = False
    # Minimum similarity (1 - distance) to inject a trajectory.
    # Prevents injecting a vaguely-related trajectory as "relevant".
    inject_min_similarity: float = 0.75


class WorkflowInductionConfig(BaseModel):
    """Trajectory-to-workflow induction (Phase 4.1).

    Off by default — enable once you have ≥25 successful sessions.
    The agent calls workflow_suggest to surface recurring task clusters;
    it then decides whether to write a workflow skill via propose_skill.
    """
    enabled: bool = False
    # Minimum number of similar successful trajectories to form a cluster.
    min_cluster_size: int = 3
    # Cap before O(N²) clustering (Pi budget).
    max_trajectories: int = 100
    # Cosine similarity threshold for trajectory clustering.
    cluster_threshold: float = 0.85


class NanoHermesConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    memory: MemoryBudgets = Field(default_factory=MemoryBudgets)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    skill_stats: SkillStatsConfig = Field(default_factory=SkillStatsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    workflow_induction: WorkflowInductionConfig = Field(default_factory=WorkflowInductionConfig)
    trajectory_retention_days: int = 45
    reflection_scope: Literal["session", "global"] = "session"
    # Apply regex-based secret redaction to user-supplied content before it
    # lands on disk (skill bodies, companion files, memory entries,
    # reflections). Default on — opt out only for debugging.
    redact_secrets: bool = True
