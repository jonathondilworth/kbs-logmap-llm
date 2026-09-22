"""
logmap_llm.oracle.rag.types

Immutable value types for query-specific RAG few-shot retrieval.

Everything here is a frozen dataclass or a str-enum, so a RetrievalResult can be shared
across concurrent oracle consultations without cross-contamination. The retriever produces
a fresh RetrievalResult per query.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityKind(str, Enum):
    """The four disjoint typed pools. Values match LogMap's entityType column codes."""
    CLS = "CLS"
    OPROP = "OPROP"
    DPROP = "DPROP"
    INST = "INST"

    @classmethod
    def coerce(cls, value: str) -> "EntityKind":
        v = str(value).strip().upper()
        # LogMap sometimes emits UNKNO / PROP; map defensively but never silently to CLS
        if v in cls._value2member_map_:
            return cls(v)
        raise ValueError(f"Unknown entity kind: {value!r} (expected one of {[e.value for e in cls]})")


class Mode(str, Enum):
    """Retrieval modes. STATIC_* preserve the legacy query-agnostic sampler for the ablation."""
    ZERO_SHOT = "zero_shot"
    STATIC_RANDOM = "static_random"
    STATIC_HARD = "static_hard"
    QUERY_RAG = "query_rag"


class Source(str, Enum):
    """Provenance of a positive example. Anchors are pseudo-labels, kept distinct from gold."""
    GOLD = "gold"          # authorised training alignment
    ANCHOR = "anchor"      # high-confidence LogMap anchor (pseudo-label)
    CONSTRUCTED = "constructed"  # a synthesised negative (near-miss / random)


# negative construction layouts

#: How the k/2 pseudo-negatives are built.
#:
#:   paired-sibling-v2  Ni = (Pi.src, highest-ranked eligible ontology sibling of Pi.tgt)
#:   paired-donor-v2    Ni = (Pi.src, target of the next eligible ranked donor after Pi)
#:   donor-cross-v1     Frozen: every negative takes the top-ranked donor's source and a
#:                      successive donor's target. Kept only to reproduce the completed
#:                      campaigns byte-for-byte.
#:
#: Both v2 layouts guarantee ``Ni.src == Pi.src`` by construction, so the rendered block is
#: k/2 independent minimal pairs.
LAYOUT_PAIRED_SIBLING_V2 = "paired-sibling-v2"
LAYOUT_PAIRED_DONOR_V2 = "paired-donor-v2"
LAYOUT_DONOR_CROSS_V1 = "donor-cross-v1"

NEGATIVE_LAYOUTS = (
    LAYOUT_PAIRED_SIBLING_V2,
    LAYOUT_PAIRED_DONOR_V2,
    LAYOUT_DONOR_CROSS_V1,
)

#: The layouts that obey the shared pairing rule.
PAIRED_LAYOUTS = (LAYOUT_PAIRED_SIBLING_V2, LAYOUT_PAIRED_DONOR_V2)

#: ``Example.construction`` values.
CONSTRUCTION_SIBLING = "sibling"
CONSTRUCTION_PAIRED_DONOR = "paired-donor"
CONSTRUCTION_DONOR_CROSS_V1 = "donor-cross-v1"
CONSTRUCTION_RANDOM = "random"


class Direction(str, Enum):
    FORWARD = "forward"
    REVERSE = "reverse"


# fallback policy

# What to do when a pool is empty / too small / the budget overflows.
# Strict experiments pick "error"; lenient runs pick an explicit degradation that is recorded.
_EMPTY_CHOICES = ("error", "zero_shot", "random")
_UNDERSIZED_CHOICES = ("error", "reduce_k", "pad_other")
_BUDGET_CHOICES = ("error", "reduce_k", "zero_shot")


@dataclass(frozen=True)
class FallbackPolicy:
    on_empty_pool: str = "zero_shot"       # no positives of this type available
    on_undersized: str = "reduce_k"        # fewer usable examples than k
    on_budget_overflow: str = "reduce_k"   # examples do not fit the token budget
    allow_random_negatives: bool = True    # may hard-negative construction fall back to random?

    def __post_init__(self):
        if self.on_empty_pool not in _EMPTY_CHOICES:
            raise ValueError(f"on_empty_pool must be one of {_EMPTY_CHOICES}")
        if self.on_undersized not in _UNDERSIZED_CHOICES:
            raise ValueError(f"on_undersized must be one of {_UNDERSIZED_CHOICES}")
        if self.on_budget_overflow not in _BUDGET_CHOICES:
            raise ValueError(f"on_budget_overflow must be one of {_BUDGET_CHOICES}")

    @classmethod
    def strict(cls) -> "FallbackPolicy":
        return cls(on_empty_pool="error", on_undersized="error",
                   on_budget_overflow="error", allow_random_negatives=False)


@dataclass(frozen=True)
class RagConfig:
    mode: Mode = Mode.QUERY_RAG
    k: int = 4
    positive_fraction: float = 0.5          # target share of positive examples (balance)
    seed: int = 42                          # only perturbs an explicit random fallback
    token_budget: int | None = None         # budget for the *examples* block; None = unbounded
    answer_pos: str = '{"answer": true}'    # exact positive answer string (format-matched)
    answer_neg: str = '{"answer": false}'   # exact negative answer string
    bidirectional: bool = False             # emit forward+reverse per selected pair
    preprocessing_version: str = "v2"       # part of the index cache key
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)

    # How pseudo-negatives are constructed. The default is the frozen legacy layout so the
    # CPU-only unit tests that build a bare RagConfig() keep exercising the reproduction
    # path; every pipeline entry point requires the layout explicitly, so no campaign can
    # acquire one by default.
    negative_layout: str = LAYOUT_DONOR_CROSS_V1

    # How many ranked sibling candidates to request per negative under paired-sibling-v2.
    # Eligibility filtering can reject several, so this is deliberately > 1.
    sibling_candidate_count: int = 8

    # Recorded in the trace so the sibling regime is evidence, not just a condition label.
    # The retriever never interprets these; the pipeline sets them from the resolved selector.
    sibling_strategy: str = ""
    sibling_encoder_revision: str = ""

    def __post_init__(self):
        if self.k < 0:
            raise ValueError("k must be >= 0")
        if not (0.0 <= self.positive_fraction <= 1.0):
            raise ValueError("positive_fraction must be in [0, 1]")
        if self.negative_layout not in NEGATIVE_LAYOUTS:
            raise ValueError(
                f"negative_layout must be one of {list(NEGATIVE_LAYOUTS)}; "
                f"got {self.negative_layout!r}"
            )
        if self.sibling_candidate_count < 1:
            raise ValueError("sibling_candidate_count must be >= 1")


# query + example

@dataclass(frozen=True)
class QueryMapping:
    """The live M_ask candidate being asked. embed_text is what the retriever ranks against."""
    src_iri: str
    tgt_iri: str
    kind: EntityKind
    embed_text: str

    @property
    def unordered(self) -> frozenset:
        return frozenset({self.src_iri, self.tgt_iri})


#: Per-negative provenance carried by Example/ExampleRef and serialised into the trace.
#: All None on positives and on random negatives; all None on donor-cross-v1 except
#: `construction`, so a v1 trace's `selected[]` entry stays a strict superset of the frozen
#: shape (no existing key changes name, type or value).
NEGATIVE_PROVENANCE_FIELDS = (
    "construction",
    "derived_from_example_id",
    "donor_example_id",
    "sibling_rank",
    "sibling_score",
    "sibling_type_specificity",
    "sibling_rule",
    "negative_fallback_reason",
)


@dataclass(frozen=True)
class Example:
    """A candidate few-shot example in a typed corpus (before rendering/selection)."""
    example_id: str
    src_iri: str
    tgt_iri: str
    kind: EntityKind
    label: bool                 # True = positive (equivalent), False = negative
    source: Source
    embed_text: str
    payload: Any = None         # opaque; consumed by render_fn (e.g. entity objects / label strings)

    # negative provenance (see NEGATIVE_PROVENANCE_FIELDS)
    construction: str | None = None
    derived_from_example_id: str | None = None   # the positive Pi this negative was built from
    donor_example_id: str | None = None          # the donor whose target was borrowed, if any
    sibling_rank: int | None = None              # 0-based rank among ranked siblings
    sibling_score: float | None = None
    sibling_type_specificity: str | None = None  # "specific" | "uninformative" (INST siblings)
    #: Which sibling rule produced the candidate: "class" (parent's other children),
    #: "instance" (shared rdf:type) or "property" (shared rdfs:domain). Recorded because the
    #: rule is decided by how the target resolves, not by the query's LogMap lane -- e.g. on
    #: starwars-swtor every OPROP/DPROP target resolves as an InstanceEntity, so the property
    #: lanes actually run the instance rule.
    sibling_rule: str | None = None
    negative_fallback_reason: str | None = None  # why the branch's primary rule was not used

    @property
    def unordered(self) -> frozenset:
        return frozenset({self.src_iri, self.tgt_iri})

    def provenance(self) -> dict:
        """The per-negative provenance as a plain dict, for the retrieval trace."""
        return {name: getattr(self, name) for name in NEGATIVE_PROVENANCE_FIELDS}


@dataclass(frozen=True)
class ExampleRef:
    """A *selected*, rendered example returned to the caller. Immutable."""
    example_id: str
    src_iri: str
    tgt_iri: str
    kind: EntityKind
    label: bool
    source: Source
    direction: Direction
    rank: int
    similarity: float
    prompt_text: str
    answer_text: str
    tokens: int

    # --- negative provenance, copied verbatim from the selected Example ---
    construction: str | None = None
    derived_from_example_id: str | None = None
    donor_example_id: str | None = None
    sibling_rank: int | None = None
    sibling_score: float | None = None
    sibling_type_specificity: str | None = None
    sibling_rule: str | None = None
    negative_fallback_reason: str | None = None

    def provenance(self) -> dict:
        return {name: getattr(self, name) for name in NEGATIVE_PROVENANCE_FIELDS}

    def as_message_pair(self) -> tuple[str, str]:
        """(user_prompt, assistant_answer) — matches the legacy few-shot tuple contract."""
        return (self.prompt_text, self.answer_text)


@dataclass(frozen=True)
class RetrievalTrace:
    requested_mode: str
    effective_mode: str
    requested_k: int
    effective_k: int
    entity_type: str
    relation: str
    prompt_family: str
    answer_format: str
    selected: tuple = ()          # tuple[dict]: id,label,source,direction,rank,similarity
    exclusions: tuple = ()        # tuple[dict]: id,reason
    # Retrieval-mode degradation only: the retriever could not run the requested mode/k.
    # `stage_two._publish_recorded_trace_fallbacks` keys on this field and publishes
    # rag_fallback.json, which `run._is_degraded` turns into a hard BatchRunError unless
    # rag_failure_policy == "record_zero_shot". A negative-construction fallback is a
    # different thing and must never be recorded here.
    fallback_reason: str | None = None
    # Negative-construction fallback: the layout's primary target rule was unavailable for
    # at least one negative. The block is complete and the query is not degraded.
    negative_fallback_reason: str | None = None
    negative_layout: str = ""
    sibling_strategy: str = ""
    sibling_encoder_revision: str = ""
    corpus_hash: str = ""
    index_hash: str = ""
    encoder_repo: str = ""
    encoder_revision: str = ""
    #: The library versions that produced the embeddings. Recorded because two runs can agree
    #: on every other encoder identity field and still hold different vectors.
    encoder_runtime: str = ""
    preprocessing_version: str = ""
    token_budget: int | None = None
    tokens_used: int = 0

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


@dataclass(frozen=True)
class RetrievalResult:
    examples: tuple  # tuple[ExampleRef, ...]
    trace: RetrievalTrace

    def message_pairs(self) -> list:
        """[(user_prompt, assistant_answer), ...] in selected order — feeds the oracle manager."""
        return [e.as_message_pair() for e in self.examples]
