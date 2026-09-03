"""Script-first corpus generation for the retrieval harness.

The facts are generated BEFORE the transcripts that mention them, following
Ground Truth First (arXiv:2607.21962). That ordering is the whole point:

* Gold is correct by construction — a fact is planted, then rendered, so the
  answer key never depends on an LLM reading a transcript and guessing.
* Every fact carries a validity interval, so a superseded value is a *different
  version* of the fact rather than a contradiction. A question asked "as of
  now" has exactly one right answer, and a later restatement of an old value
  cannot silently turn into a leak.
* Each rendered chunk records which fact it states, so gold *evidence ids* come
  free. That is what lets the harness score retrieval without an LLM.

The domain is deliberately agentic-coding (services, deploys, error codes, PR
numbers, file paths) rather than the chat/email life-scripts used by the
published instruments. nano-hermes retrieves over coding sessions, where exact
identifiers dominate and lexical matching is unusually strong; measuring it on
a conversational corpus would answer a question we are not asking.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# Deterministic vocabulary. Kept small and concrete so that a planted value is
# a rare token in the corpus — a fact must be *retrieved*, not guessed from
# surrounding prose.
_SERVICES = [
    "billing-api", "auth-gateway", "search-indexer", "media-worker",
    "notify-relay", "ledger-sync", "quota-broker", "session-store",
]
_ATTRS = {
    "rollback_command": lambda r: f"kubectl rollout undo deploy/{r.choice(_SERVICES)} --to-revision={r.randint(2, 40)}",
    "oncall_owner": lambda r: r.choice(["priya", "marcus", "wen", "ola", "sam", "devi"]),
    "error_code": lambda r: f"E{r.randint(1000, 9999)}",
    "config_path": lambda r: f"/srv/{r.choice(['etc','conf','opt'])}/{r.choice(_SERVICES)}/{r.choice(['main','prod','edge'])}.yaml",
    "pr_number": lambda r: f"#{r.randint(1200, 9800)}",
    "timeout_seconds": lambda r: str(r.choice([15, 30, 45, 60, 90, 120])),
}


@dataclass
class Fact:
    fact_id: str
    subject: str
    attr: str
    value: str
    version: int          # 0 = original, 1+ = supersedes the previous version
    superseded: bool      # True when a later version exists


@dataclass
class Chunk:
    """One rendered turn. ``fact_id`` is the gold evidence link."""
    session_idx: int
    turn_index: int
    role: str
    content: str
    fact_id: str | None = None


@dataclass
class Question:
    qid: str
    kind: str             # extraction | knowledge_update | abstention
    text: str
    gold: str
    gold_fact_id: str | None
    gold_chunk_keys: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class Corpus:
    facts: list[Fact]
    chunks: list[Chunk]
    questions: list[Question]

    @property
    def n_sessions(self) -> int:
        return len({c.session_idx for c in self.chunks})


# Filler that is topically plausible but states no planted fact. Retrieval has
# to discriminate against this, not against unrelated noise.
_FILLER = [
    "Checked the dashboard for {svc}; latency looked flat over the last hour.",
    "Re-ran the {svc} test suite locally, everything green on the second pass.",
    "Rebased the {svc} branch onto main and force-pushed after the review.",
    "Paged through the {svc} logs but nothing stood out around that window.",
    "Left a comment on the {svc} rollout doc asking about the staging soak.",
    "Bumped the {svc} dependency lockfile, no behaviour change expected.",
]


def generate_corpus(
    *,
    seed: int = 7,
    n_facts: int = 24,
    supersede_frac: float = 0.25,
    sessions: int = 12,
    filler_per_session: int = 6,
    n_abstention: int = 4,
) -> Corpus:
    """Build a corpus and its answer key.

    *supersede_frac* of facts get a second version written into a later
    session; the question for those asks for the current value, so an arm that
    retrieves only the stale mention scores zero. That is the knowledge-update
    axis every published instrument tests.
    """
    r = random.Random(seed)
    facts: list[Fact] = []

    # Each (subject, attr) pair may be used at most once: two live facts for
    # the same pair would make "what is the current X for Y?" ambiguous, and an
    # ambiguous gold silently scores correct answers as wrong.
    pairs = [(s_, a_) for s_ in _SERVICES for a_ in _ATTRS]
    if n_facts > len(pairs):
        raise ValueError(
            f"n_facts={n_facts} exceeds {len(pairs)} unique (service, attribute) "
            "pairs; add vocabulary before asking for more facts"
        )
    for i, (subject, attr) in enumerate(pairs[:n_facts]):
        facts.append(
            Fact(fact_id=f"f{i:03d}", subject=subject, attr=attr,
                 value=_ATTRS[attr](r), version=0, superseded=False)
        )

    # Choose which facts get a newer version.
    n_super = int(n_facts * supersede_frac)
    superseded_ids = set(r.sample([f.fact_id for f in facts], n_super)) if n_super else set()
    updates: list[Fact] = []
    for f in facts:
        if f.fact_id in superseded_ids:
            f.superseded = True
            new_value = _ATTRS[f.attr](r)
            while new_value == f.value:          # a no-op update tests nothing
                new_value = _ATTRS[f.attr](r)
            updates.append(
                Fact(fact_id=f.fact_id, subject=f.subject, attr=f.attr,
                     value=new_value, version=1, superseded=False)
            )

    # ---- render sessions -------------------------------------------------
    # Originals go in the first half, updates strictly after, so "current"
    # always means "later in the timeline".
    chunks: list[Chunk] = []
    half = max(1, sessions // 2)

    def _state(fact: Fact) -> str:
        verb = "updated to" if fact.version else "is"
        return (
            f"For {fact.subject}, the {fact.attr.replace('_', ' ')} {verb} {fact.value}."
        )

    def _emit(session_idx: int, items: list[Fact]) -> None:
        turn = 0
        for fact in items:
            chunks.append(Chunk(session_idx, turn, "user",
                                f"Working on {fact.subject} today.", None))
            turn += 1
            chunks.append(Chunk(session_idx, turn, "assistant", _state(fact), fact.fact_id))
            turn += 1
            # At least one distractor per fact: integer division alone drops to
            # zero when a session's fact group is larger than filler_per_session,
            # silently removing the noise arms are supposed to discriminate
            # against and inflating recall for every arm equally.
            for _ in range(max(1, filler_per_session // max(1, len(items)))):
                chunks.append(Chunk(session_idx, turn, "assistant",
                                    r.choice(_FILLER).format(svc=r.choice(_SERVICES)), None))
                turn += 1

    for idx, group in enumerate(_chunked(facts, half)):
        _emit(idx, group)
    for idx, group in enumerate(_chunked(updates, max(1, sessions - half))):
        _emit(half + idx, group)

    # ---- questions -------------------------------------------------------
    # Walk chunks in order, tracking which version of each fact a chunk states:
    # the Nth mention of a fact is version N.
    seen: dict[str, int] = {}
    evidence: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for c in chunks:
        if c.fact_id is None:
            continue
        v = seen.get(c.fact_id, 0)
        evidence.setdefault((c.fact_id, v), []).append((c.session_idx, c.turn_index))
        seen[c.fact_id] = v + 1

    questions: list[Question] = []
    latest: dict[str, Fact] = {}
    for f in facts:
        latest[f.fact_id] = f
    for f in updates:
        latest[f.fact_id] = f

    for fid, fact in sorted(latest.items()):
        kind = "knowledge_update" if fact.version else "extraction"
        questions.append(Question(
            qid=f"q_{fid}",
            kind=kind,
            text=(f"What is the current {fact.attr.replace('_', ' ')} for {fact.subject}?"),
            gold=fact.value,
            gold_fact_id=fid,
            gold_chunk_keys=evidence.get((fid, fact.version), []),
        ))

    # Abstention: services that appear nowhere. An arm that hallucinates a
    # plausible value should be penalised, not rewarded for fluency.
    for i in range(n_abstention):
        ghost = f"phantom-{i}-svc"
        questions.append(Question(
            qid=f"q_abs{i}", kind="abstention",
            text=f"What is the current rollback command for {ghost}?",
            gold="NOT IN CONTEXT", gold_fact_id=None, gold_chunk_keys=[],
        ))

    return Corpus(facts=facts + updates, chunks=chunks, questions=questions)


def _chunked(items: list, n_groups: int) -> list[list]:
    if not items:
        return []
    n_groups = max(1, min(n_groups, len(items)))
    size = (len(items) + n_groups - 1) // n_groups
    return [items[i:i + size] for i in range(0, len(items), size)]
