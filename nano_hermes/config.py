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
    # Bi-temporal supersession: after distilling a new fact, check its nearest
    # prior facts and stamp invalid_at on any the new one contradicts/replaces.
    # One LLM call, gated by the similarity prefilter below — fires only on a
    # near-duplicate write. Set False to keep all facts "current".
    bitemporal_invalidation_enabled: bool = True
    # Cosine-similarity floor for a fact to be a supersession candidate.
    # Tighter than distill_cluster_threshold (0.88) would be too loose here —
    # only near-duplicates plausibly contradict. 0.86 catches paraphrases of
    # the same subject while excluding merely-related facts.
    bitemporal_supersede_threshold: float = 0.86
    # memory_patch(action="audit"): standing hygiene sweep over stored facts.
    # Caps the number of (newest) anchor facts scanned per run — one LLM call
    # per anchor that has near-duplicate older neighbours. Bounds Pi cost.
    contradiction_sweep_max_anchors: int = 20
    # A-MEM neighbour evolution: when a new fact links to a near neighbour,
    # union the new fact's keywords/tags into that single closest neighbour
    # (zero-LLM) so older facts stay discoverable as the graph grows. Capped
    # to bound tag/keyword growth per fact.
    amem_evolve_neighbours: bool = True
    amem_neighbour_max_tags: int = 12


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
    # Memory-save nudge: every N user turns, deliver a system prompt asking
    # the agent whether anything is worth saving via memory_patch. Cadence
    # complement to the reactive salience-threshold nudge. 0 = disabled.
    memory_save_nudge_interval: int = 8


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
    # Multi-view skill retrieval (CRAFT idea, Pi-fit variant): blend the
    # description-vector match with a lexical overlap against the skill NAME.
    # Names are precise handles a fused "name: description" vector dilutes;
    # this recovers name-led hits with zero extra embeddings/storage. The
    # weight shifts effective_distance, same scale as ucb1_coefficient.
    # Off by default (changes ranking).
    multi_view_retrieval: bool = False
    multi_view_name_weight: float = 0.05
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
    # Skill umbrella consolidation: cluster near-duplicate sibling skills
    # (agent-authored, unpinned, active) by cosine over their stored vectors,
    # then merge each cluster into one broader umbrella skill via a single LLM
    # call and deprecate the absorbed siblings (status='deprecated', reason
    # "absorbed_into: <umbrella>"). Runs inside the evolution cycle, which is
    # already snapshot-protected. Off by default.
    umbrella_merge_enabled: bool = False
    umbrella_sim_threshold: float = 0.86
    umbrella_min_cluster: int = 2
    umbrella_max_cluster: int = 5
    umbrella_max_merges_per_run: int = 2
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
    # Curator (Phase 8): periodic stale-skill maintenance.
    # On a new session's iteration 0, deprecate active skills that have been
    # unused for *curator_stale_after_days* (no last_used_at update in N days)
    # provided they've already been exercised at least *curator_min_uses*
    # times — never touches untested skills. Cooldown prevents firing more
    # than once per *curator_cooldown_hours*.
    # Set curator_enabled=False or curator_stale_after_days=0 to disable.
    curator_enabled: bool = True
    # active -> stale: dormant this long. Stale skills stay searchable but
    # demoted; using one again reactivates it (stale -> active).
    curator_stale_after_days: int = 30
    # stale -> deprecated: still dormant this long after going stale. Must be
    # > curator_stale_after_days or staling and archiving collapse into one step.
    curator_archive_after_days: int = 90
    curator_min_uses: int = 3
    curator_cooldown_hours: int = 24
    # Pre-evolution snapshot: before each GEPA/rewriter cycle, snapshot the
    # state DB + skills/ dir so a bad batch can be rolled back as one unit
    # (python -m nano_hermes.skills.evolution_snapshot <workspace>, offline).
    snapshot_before_evolution: bool = True
    # Each snapshot copies the whole state DB; keep few on an SD card.
    snapshot_retain: int = 3
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
    # Max past cases injected (MMR-diversified). Memento (arXiv 2508.16153)
    # finds case-based retrieval peaks around K=4 and declines beyond — a
    # small, high-quality set beats a large one, and costs fewer tokens on a Pi.
    # Both successes ("what worked") and failures ("what to avoid") are kept.
    inject_k: int = 4


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


class PrincipleEvolutionConfig(BaseModel):
    """ACE-style automatic curation of the `principles` playbook.

    The principles store is injected each turn by FTS match. This block adds
    the ACE evolution layer on top: embedding dedup on write, a session-
    boundary LLM curator that proposes add/update/prune *deltas* (never a
    full rewrite — that's the context-collapse failure mode), helpful/harmful
    attribution from session outcomes, and counter+recency injection ranking.

    Off by default (like GEPA) — it needs accumulated failure data and one
    hosted LLM call per cycle.
    """
    enabled: bool = False
    # Run the curator every N completed sessions (0 = never). Mirrors
    # skill_stats.rewrite_session_interval.
    session_interval: int = 0
    # Cooldown between curator runs regardless of session cadence.
    cooldown_hours: int = 24
    # Cosine sim at/above which a new principle is merged into an existing one
    # rather than inserted (used by both the manual tool and the curator).
    dedup_threshold: float = 0.85
    # Hard cap on stored principles; over this, lowest-value curator-authored
    # rows are pruned (pure SQL, never manual/pinned).
    max_principles: int = 200
    # Max delta ops applied per curator run (bounds LLM blast radius + work).
    max_ops_per_run: int = 8
    # Only sessions failing at/above this rate feed the curator's reflect step.
    failure_threshold: float = 0.4
    # Recency half-life (days) for the injection-ranking decay term.
    inject_rank_recency_half_life_days: float = 30.0


class DecayConfig(BaseModel):
    """Memory decay: bound the unbounded ``semantic_facts`` table by eviction,
    and gently demote stale items in the two retrieval paths that rank results.

    Facts are a write-mostly staging store (distill → agent promotes the good
    ones into MEMORY.md) and are never otherwise deleted, so on a long-lived
    Pi they grow without bound. Eviction here is conservative: superseded facts
    are dropped after a grace window, and *valid* facts are dropped only once
    they are both old AND low-importance. High-importance facts never auto-evict.

    Ranking decay is a recency multiplier/term applied to trajectory_search and
    global-reflection injection — the only paths that actually rank stored rows.
    It nudges fresher rows up without erasing strong semantic matches.
    """
    enabled: bool = True
    # --- Fact eviction (semantic_facts) ---
    # Valid facts older than this AND below the importance floor are eligible.
    fact_retention_days: int = 90
    # importance is the 1–10 distiller score; facts at/above this never evict.
    fact_evict_importance_floor: int = 4
    # Superseded (invalid_at set) facts are deleted this long after invalidation.
    superseded_grace_days: int = 14
    # Hard cap on deletions per purge pass — bounds work/IO on a Pi.
    max_evictions_per_run: int = Field(default=500, gt=0)
    # --- Ranking recency decay ---
    # Days for the recency factor to halve (0.5 ** age/half_life).
    ranking_half_life_days: float = Field(default=30.0, gt=0)
    # Max fraction by which an ancient trajectory's fused score is demoted.
    trajectory_decay_weight: float = Field(default=0.3, ge=0, le=1)
    # Additive weight of the recency term in global-reflection scoring
    # (comparable to the existing citation_weight=0.2).
    reflection_decay_weight: float = Field(default=0.15, ge=0)


class NanoHermesConfig(BaseModel):
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    principles: PrincipleEvolutionConfig = Field(
        default_factory=PrincipleEvolutionConfig
    )
    decay: DecayConfig = Field(default_factory=DecayConfig)
    memory: MemoryBudgets = Field(default_factory=MemoryBudgets)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    reflection: ReflectionConfig = Field(default_factory=ReflectionConfig)
    skill_stats: SkillStatsConfig = Field(default_factory=SkillStatsConfig)
    skills: SkillsConfig = Field(default_factory=SkillsConfig)
    trajectory: TrajectoryConfig = Field(default_factory=TrajectoryConfig)
    workflow_induction: WorkflowInductionConfig = Field(default_factory=WorkflowInductionConfig)
    trajectory_retention_days: int = 45
    # Minimum days between full VACUUMs. Once data ages past the retention
    # window, every startup purges a fresh day of rows; without this gate
    # each purge would trigger a whole-DB VACUUM (exclusive lock, slow on a
    # Pi's microSD), contending with the live archiver. Reclaim space at most
    # this often instead.
    vacuum_min_interval_days: int = 7
    # SQLite busy-wait before raising SQLITE_BUSY. Lets the archiver wait out
    # a background VACUUM/purge lock instead of dropping the turn's archive.
    sqlite_busy_timeout_ms: int = 10000
    reflection_scope: Literal["session", "global"] = "session"
    # Apply regex-based secret redaction to user-supplied content before it
    # lands on disk (skill bodies, companion files, memory entries,
    # reflections). Default on — opt out only for debugging.
    redact_secrets: bool = True
    # Scan MEMORY.md at prompt-load time and replace poisoned lines with
    # [BLOCKED: …] placeholders. Closes the gap where a direct on-disk edit,
    # a sync, or a DB restore reintroduces an entry that bypassed the
    # write-time guard. The on-disk file is never modified.
    memory_loadtime_scan: bool = True
