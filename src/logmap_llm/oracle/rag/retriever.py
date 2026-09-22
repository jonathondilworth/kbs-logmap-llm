"""
logmap_llm.oracle.rag.retriever

Query-conditioned, per-candidate, immutable few-shot retrieval. Concurrent calls share only
read-only state (corpus, encoder, cached indexes) and each returns a fresh RetrievalResult.

Modes:
  ZERO_SHOT      -> no examples.
  STATIC_RANDOM  -> legacy: query-agnostic order, random (column-swap) negatives.
  STATIC_HARD    -> legacy: query-agnostic order, near-miss (recombination) negatives.
  QUERY_RAG      -> examples ranked by similarity to the query; hard-similar negatives.

Negative construction is selected by `RagConfig.negative_layout` (see types.py):

  paired-sibling-v2  Ni = (Pi.src, highest-ranked eligible ontology sibling of Pi.tgt)
  paired-donor-v2    Ni = (Pi.src, target of the next eligible ranked donor after Pi)
  donor-cross-v1     Frozen legacy layout; kept only to reproduce completed campaigns
                     byte-for-byte.

Both v2 layouts guarantee `Ni.src == Pi.src` by construction, making the rendered block k/2
independent minimal pairs. Random negatives are reachable only via STATIC_RANDOM or an
explicitly configured fallback, and are always recorded.
"""
from __future__ import annotations

from typing import Callable, Optional
import hashlib
import math
import random as _random
import threading

import numpy as np

from .types import (
    EntityKind, Mode, Source, Direction,
    RagConfig, QueryMapping, Example, ExampleRef, RetrievalTrace, RetrievalResult,
    LAYOUT_DONOR_CROSS_V1, LAYOUT_PAIRED_SIBLING_V2, PAIRED_LAYOUTS,
    CONSTRUCTION_DONOR_CROSS_V1, CONSTRUCTION_PAIRED_DONOR, CONSTRUCTION_RANDOM,
    CONSTRUCTION_SIBLING,
)
from .corpus import TypedCorpus
from .encoder import Encoder
from .tokenizer import TokenCounter, HeuristicTokenCounter
from .index import EmbeddingIndex, compute_index_hash


# render_fn(src_iri, tgt_iri, payload) -> user_prompt_text
RenderFn = Callable[[str, str, object], str]

# sibling_fn(target_iri, kind, max_count) -> (ranked [candidate dict], reason_when_empty)
#
# Each candidate is {"iri", "score", "type_specificity", "rule"}; `rule` records which
# sibling notion produced the candidate (decided by how the target resolves, not by the
# query's LogMap lane).
#
# Injected, never imported: this module is deliberately owlready2-free. The production
# implementation (pipeline/rag_fewshot.py, backed by SiblingSelector) must be deterministic,
# must never raise, and must return a reason string whenever the candidate list is empty so
# the fallback is recorded rather than silent.
SiblingFn = Callable[[str, EntityKind, int], "tuple[list[dict], Optional[str]]"]


class RagRetrievalError(RuntimeError):
    """Raised when a strict FallbackPolicy forbids the degradation the situation requires."""


class RagRetriever:
    def __init__(
        self,
        corpus: TypedCorpus,
        encoder: Encoder,
        render_fn: RenderFn,
        config: RagConfig,
        token_counter: Optional[TokenCounter] = None,
        cache_dir: Optional[str] = None,
        sibling_fn: Optional[SiblingFn] = None,
    ):
        self.corpus = corpus
        self.encoder = encoder
        self.render_fn = render_fn
        self.sibling_fn = sibling_fn
        self.config = config
        if config.negative_layout == LAYOUT_PAIRED_SIBLING_V2 and sibling_fn is None:
            raise RagRetrievalError(
                "negative_layout='paired-sibling-v2' requires an injected sibling_fn; "
                "refusing to silently run the donor rule under the sibling condition's name"
            )
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.cache_dir = cache_dir
        self._indexes: dict = {}  # kind -> EmbeddingIndex (positives), read-only after build
        self._index_lock = threading.Lock()  # guards the idempotent lazy index build only

    def warmup(self) -> None:
        """Pre-build all typed indexes so retrieve() touches no shared mutable state afterwards."""
        for kind in EntityKind:
            if self.corpus.positive_pool(kind):
                self._positive_index(kind)

    # index management

    def _positive_index(self, kind: EntityKind) -> EmbeddingIndex:
        # fast path: already built (dict reads are safe alongside the guarded write below)
        idx = self._indexes.get(kind)
        if idx is not None:
            return idx
        with self._index_lock:
            if kind in self._indexes:
                return self._indexes[kind]
            examples = self.corpus.positive_pool(kind)
            corpus_hash = self.corpus.corpus_hash(kind)
            index_hash = compute_index_hash(
                dataset_sha=self.corpus.dataset_sha,
                encoder_repo=self.encoder.repo,
                encoder_revision=self.encoder.revision,
                encoder_preprocessing=getattr(self.encoder, "preprocessing_version", ""),
                retriever_preprocessing=self.config.preprocessing_version,
                language=self.corpus.language, kind=kind, corpus_hash=corpus_hash,
                encoder_runtime=str(getattr(self.encoder, "runtime_versions", "")),
            )
            self._indexes[kind] = EmbeddingIndex.build(
                examples, self.encoder, index_hash, cache_dir=self.cache_dir)
        return self._indexes[kind]

    # public API

    def retrieve(
        self,
        query_mapping: QueryMapping,
        entity_type: EntityKind,
        relation: str,
        prompt_template: str,
        k: int,
        corpus_id: str,
        exclude_keys: Optional[set] = None,
    ) -> RetrievalResult:
        cfg = self.config
        kind = entity_type if isinstance(entity_type, EntityKind) else EntityKind.coerce(entity_type)
        answer_format = "structured" if cfg.answer_pos.strip().startswith("{") else "plain"

        # exclusions: query (+ its reverse, same unordered key) and the caller's set (M_ask, test refs)
        excl_keys = set(exclude_keys or set())
        excl_keys.add(query_mapping.unordered)

        def _trace(effective_mode, effective_k, selected, exclusions, fallback_reason,
                   tokens_used, index_hash="", corpus_hash="", negative_fallback_reason=None):
            return RetrievalTrace(
                requested_mode=cfg.mode.value, effective_mode=effective_mode,
                requested_k=k, effective_k=effective_k,
                entity_type=kind.value, relation=str(relation), prompt_family=str(prompt_template),
                answer_format=answer_format, selected=tuple(selected), exclusions=tuple(exclusions),
                fallback_reason=fallback_reason,
                negative_fallback_reason=negative_fallback_reason,
                negative_layout=cfg.negative_layout,
                sibling_strategy=cfg.sibling_strategy,
                sibling_encoder_revision=cfg.sibling_encoder_revision,
                corpus_hash=corpus_hash, index_hash=index_hash,
                encoder_repo=self.encoder.repo, encoder_revision=self.encoder.revision,
                encoder_runtime=str(getattr(self.encoder, "runtime_versions", "")),
                preprocessing_version=cfg.preprocessing_version,
                token_budget=cfg.token_budget, tokens_used=tokens_used,
            )

        # (1) zero-shot / k<=0
        if cfg.mode == Mode.ZERO_SHOT or k <= 0:
            reason = "zero_shot mode" if cfg.mode == Mode.ZERO_SHOT else "k<=0"
            return RetrievalResult((), _trace("zero_shot", 0, [], [], reason, 0))

        # (2) empty pool handling
        if self.corpus.is_empty(kind) or not self.corpus.positive_pool(kind):
            pol = cfg.fallback.on_empty_pool
            if pol == "error":
                raise RagRetrievalError(f"No positive examples for kind {kind.value}; strict policy.")
            reason = f"empty positive pool for {kind.value} -> {pol} (zero-shot)"
            return RetrievalResult((), _trace("zero_shot", 0, [], [], reason, 0))

        pos_index = self._positive_index(kind)
        corpus_hash = self.corpus.corpus_hash(kind)
        exclusions_log: list = []

        # (3) rank / order positives
        if cfg.mode == Mode.QUERY_RAG:
            qvec = self.encoder.encode([query_mapping.embed_text])[0]
            ranked_pos = pos_index.rank(qvec, exclude_keys=excl_keys)  # [(Example, sim)]
        else:
            # STATIC_* : query-agnostic, deterministic seeded order
            pool = [ex for ex in self.corpus.positive_pool(kind) if ex.unordered not in excl_keys]
            rng = _random.Random(cfg.seed)
            order = sorted(pool, key=lambda e: e.example_id)  # stable base order
            rng.shuffle(order)
            ranked_pos = [(ex, float("nan")) for ex in order]

        # record positives excluded by leakage
        n_leak = sum(1 for ex in self.corpus.positive_pool(kind) if ex.unordered in excl_keys)
        if n_leak:
            exclusions_log.append({"count": n_leak, "reason": "leakage: query/reverse/M_ask/test-ref"})

        known_pos_keys = self.corpus.known_positive_keys(kind)

        # (4) how many of each label
        n_pairs = _num_pairs(k, cfg.bidirectional)
        # int(x + 0.5) rounds half up, so the exact-0.5 tie (e.g. k=1) goes to a positive.
        n_pos = max(0, int(n_pairs * cfg.positive_fraction + 0.5))
        n_neg = n_pairs - n_pos
        if cfg.positive_fraction >= 1.0:
            n_pos, n_neg = n_pairs, 0

        # (5) gather positive selections (pairs)
        chosen_pos = [ex for ex, _sim in ranked_pos[:n_pos]]
        pos_sims = {ex.example_id: sim for ex, sim in ranked_pos}

        paired = cfg.negative_layout in PAIRED_LAYOUTS and cfg.mode in (Mode.QUERY_RAG, Mode.STATIC_HARD)

        # (6) build negatives
        if paired:
            # The pairing rule requires exactly one negative per positive; an asymmetric
            # block is a hard error rather than a degradation.
            if n_pos != n_neg:
                raise RagRetrievalError(
                    f"negative_layout={cfg.negative_layout!r} requires one negative per positive, "
                    f"but k={k} with positive_fraction={cfg.positive_fraction} gives "
                    f"{n_pos} positive(s) and {n_neg} negative(s). Use an even k with "
                    "positive_fraction=0.5."
                )
            neg_selected, neg_fallback = self._build_paired_negatives(
                kind=kind, positives=chosen_pos, ranked_pos=ranked_pos,
                excl_keys=excl_keys, known_pos_keys=known_pos_keys, cfg=cfg)
        else:
            neg_selected, neg_fallback = self._build_negatives(
                kind=kind, mode=cfg.mode, n_neg=n_neg, ranked_pos=ranked_pos,
                excl_keys=excl_keys, known_pos_keys=known_pos_keys, cfg=cfg)

        # (7) balance / undersize handling
        fallback_reason = None if paired else neg_fallback
        negative_fallback_reason = neg_fallback if paired else None
        pairs: list = []  # (Example, label_bool, similarity)
        if paired:
            # Emit only complete minimal pairs, in (Pi, Ni) order. A positive whose negative
            # could not be built is dropped together with it (only that pair), so
            # `Ni.src == Pi.src` holds for every i in the rendered block by construction.
            for positive, negative in zip(chosen_pos, neg_selected):
                if negative is None:
                    continue
                example, similarity = negative
                pairs.append((positive, True, pos_sims.get(positive.example_id, float("nan"))))
                pairs.append((example, False, similarity))
        else:
            # interleave positive/negative for contrast
            pi, ni = 0, 0
            while len(pairs) < n_pairs and (pi < len(chosen_pos) or ni < len(neg_selected)):
                take_pos = (len(pairs) % 2 == 0)
                if take_pos and pi < len(chosen_pos):
                    ex = chosen_pos[pi]; pi += 1
                    pairs.append((ex, True, pos_sims.get(ex.example_id, float("nan"))))
                elif (not take_pos) and ni < len(neg_selected):
                    ex, sim = neg_selected[ni]; ni += 1
                    pairs.append((ex, False, sim))
                elif pi < len(chosen_pos):
                    ex = chosen_pos[pi]; pi += 1
                    pairs.append((ex, True, pos_sims.get(ex.example_id, float("nan"))))
                elif ni < len(neg_selected):
                    ex, sim = neg_selected[ni]; ni += 1
                    pairs.append((ex, False, sim))
                else:
                    break

        if len(pairs) < n_pairs:
            if cfg.fallback.on_undersized == "error":
                raise RagRetrievalError(
                    f"Only {len(pairs)} of {n_pairs} example pairs available for {kind.value}; strict policy.")
            fallback_reason = (fallback_reason + "; " if fallback_reason else "") + \
                f"undersized pool: {len(pairs)}/{n_pairs} pairs (reduce_k)"

        # (8) render + token budget
        refs, tokens_used, budget_reason = self._render_with_budget(pairs, cfg, paired=paired)
        if budget_reason:
            overflow_policy = cfg.fallback.on_budget_overflow
            if overflow_policy == "error":
                raise RagRetrievalError(
                    f"Token budget overflow for {kind.value}: {budget_reason}; strict policy."
                )
            if overflow_policy == "zero_shot":
                reason = budget_reason + " -> zero_shot"
                return RetrievalResult(
                    (), _trace("zero_shot", 0, [], exclusions_log, reason, 0,
                               index_hash=pos_index.index_hash, corpus_hash=corpus_hash))
            # reduce_k: keep what fitted, record why
            fallback_reason = (fallback_reason + "; " if fallback_reason else "") + budget_reason

        # Per-negative provenance is serialised here: this dict is the only thing that
        # reaches rag_traces.json. Positives and donor-cross-v1 negatives carry nulls, so
        # the frozen v1 trace shape stays a strict subset.
        selected_log = [{
            "example_id": r.example_id, "label": r.label, "source": r.source.value,
            "direction": r.direction.value, "rank": r.rank, "similarity": r.similarity,
            "src_iri": r.src_iri, "tgt_iri": r.tgt_iri, "tokens": r.tokens,
            "prompt_sha256": _text_sha256(r.prompt_text),
            **r.provenance(),
        } for r in refs]

        effective_mode = cfg.mode.value
        return RetrievalResult(
            tuple(refs),
            _trace(effective_mode, len(refs), selected_log, exclusions_log, fallback_reason,
                   tokens_used, index_hash=pos_index.index_hash, corpus_hash=corpus_hash,
                   negative_fallback_reason=negative_fallback_reason),
        )

    # negative construction: the shared pairing rule (v2)

    def _build_paired_negatives(self, *, kind, positives, ranked_pos, excl_keys,
                                known_pos_keys, cfg):
        """Build exactly one negative per positive: ``Ni = (Pi.src, ti')``.

        Returns ``(list[(Example, similarity) | None], negative_fallback_reason | None)``,
        index-aligned with ``positives``. ``None`` marks a positive with no eligible target;
        the caller drops that pair rather than emitting a dangling positive, so
        ``Ni.src == Pi.src`` holds for every rendered i. The two branches differ only in
        how ``ti'`` is chosen; eligibility is shared.
        """
        donors = [ex for ex, _sim in ranked_pos]
        sim_by_id = {ex.example_id: sim for ex, sim in ranked_pos}
        used_keys: set = set()
        selected: list = []
        reasons: list = []

        for index, positive in enumerate(positives):
            def _eligible(target_iri: str) -> bool:
                if target_iri in (positive.tgt_iri, positive.src_iri):
                    return False
                key = frozenset({positive.src_iri, target_iri})
                return not (key in known_pos_keys or key in excl_keys or key in used_keys)

            chosen = None
            sibling_reason = None
            if cfg.negative_layout == LAYOUT_PAIRED_SIBLING_V2:
                chosen, sibling_reason = self._choose_sibling_target(
                    positive=positive, kind=kind, cfg=cfg, eligible=_eligible)

            if chosen is None:
                # Branch (i) falls back to the branch (ii) rule for this negative; the
                # reason is recorded on the negative itself, never silently dropped.
                donor = self._choose_donor_target(
                    positive=positive, index=index, donors=donors, eligible=_eligible)
                if donor is not None:
                    parts = [part for part in (sibling_reason,
                                               donor.get("negative_fallback_reason")) if part]
                    donor["negative_fallback_reason"] = "; ".join(parts) if parts else None
                    chosen = donor

            if chosen is None:
                reasons.append(f"no eligible negative target for positive rank {index}")
                selected.append(None)
                continue
            if chosen.get("negative_fallback_reason"):
                reasons.append(chosen["negative_fallback_reason"])

            target_iri = chosen["target_iri"]
            used_keys.add(frozenset({positive.src_iri, target_iri}))
            negative = Example(
                example_id=chosen["example_id"],
                src_iri=positive.src_iri, tgt_iri=target_iri, kind=kind, label=False,
                source=Source.CONSTRUCTED,
                # The negative is never embedded or indexed; it inherits its positive's
                # similarity text so the record stays interpretable.
                embed_text=positive.embed_text,
                # Carry the authoritative lane, exactly as the positives do; donor-cross-v1
                # deliberately keeps its (src, tgt) tuple payload so its prompts reproduce.
                payload=kind.value,
                construction=chosen["construction"],
                derived_from_example_id=positive.example_id,
                donor_example_id=chosen.get("donor_example_id"),
                sibling_rank=chosen.get("sibling_rank"),
                sibling_score=chosen.get("sibling_score"),
                sibling_type_specificity=chosen.get("sibling_type_specificity"),
                sibling_rule=chosen.get("sibling_rule"),
                negative_fallback_reason=chosen.get("negative_fallback_reason"),
            )
            selected.append((negative, sim_by_id.get(positive.example_id, float("nan"))))

        # De-duplicate while preserving order so the reason string is stable and readable.
        unique_reasons = list(dict.fromkeys(r for r in reasons if r))
        return selected, ("; ".join(unique_reasons) if unique_reasons else None)

    def _choose_sibling_target(self, *, positive, kind, cfg, eligible):
        """Branch (i): the highest-ranked eligible ontology sibling of the positive's target."""
        try:
            candidates, reason = self.sibling_fn(
                positive.tgt_iri, kind, cfg.sibling_candidate_count)
        except Exception as exc:  # the injected callable must not be able to abort a query
            return None, f"sibling-lookup-failed:{type(exc).__name__}"
        if not candidates:
            return None, (reason or "no-siblings")
        for rank, candidate in enumerate(candidates):
            sibling_iri = candidate["iri"]
            if not eligible(sibling_iri):
                continue
            return {
                "target_iri": sibling_iri,
                "construction": CONSTRUCTION_SIBLING,
                "example_id": f"neg:sib:{positive.example_id}->{_pairid(positive.tgt_iri, sibling_iri)}",
                "sibling_rank": rank,
                "sibling_score": float(candidate["score"]),
                "sibling_type_specificity": candidate.get("type_specificity"),
                "sibling_rule": candidate.get("rule"),
            }, None
        return None, "no-eligible-sibling"

    @staticmethod
    def _choose_donor_target(*, positive, index, donors, eligible):
        """Branch (ii): the target of the next eligible ranked donor after this positive.

        Donors strictly after ``index`` are tried first (at k=2 this reproduces
        donor-cross-v1 exactly); only if none is eligible do we wrap around to the donors
        before it, which is recorded rather than silent.
        """
        for wrapped, order in ((False, range(index + 1, len(donors))),
                               (True, range(0, index))):
            for position in order:
                donor = donors[position]
                if not eligible(donor.tgt_iri):
                    continue
                return {
                    "target_iri": donor.tgt_iri,
                    "construction": CONSTRUCTION_PAIRED_DONOR,
                    "example_id": f"neg:pdn:{positive.example_id}->{donor.example_id}",
                    "donor_example_id": donor.example_id,
                    "negative_fallback_reason": "donor-wraparound" if wrapped else None,
                }
        return None

    # negative construction: the frozen v1 layout

    def _build_negatives(self, *, kind, mode, n_neg, ranked_pos, excl_keys, known_pos_keys, cfg):
        """Return (list[(Example, similarity)], fallback_reason|None).

        Frozen reproduction of ``donor-cross-v1`` plus the STATIC_RANDOM path: every
        negative takes the top-ranked donor's source and a successive donor's target.
        Preserved unchanged so the completed campaigns can be re-rendered.
        """
        if n_neg <= 0:
            return [], None

        selected: list = []
        used_keys: set = set()

        if mode in (Mode.QUERY_RAG, Mode.STATIC_HARD):
            # hard-similar near-miss: source of one query-relevant positive + target of another.
            donors = [ex for ex, _s in ranked_pos]
            sim_by_id = {ex.example_id: s for ex, s in ranked_pos}
            for a_i in range(len(donors)):
                if len(selected) >= n_neg:
                    break
                for b_i in range(len(donors)):
                    if a_i == b_i:
                        continue
                    a, b = donors[a_i], donors[b_i]
                    src, tgt = a.src_iri, b.tgt_iri
                    if src == tgt:
                        continue
                    key = frozenset({src, tgt})
                    if key in known_pos_keys or key in excl_keys or key in used_keys:
                        continue
                    used_keys.add(key)
                    neg = Example(
                        example_id=f"neg:hard:{a.example_id}->{b.example_id}",
                        src_iri=src, tgt_iri=tgt, kind=kind, label=False,
                        source=Source.CONSTRUCTED, embed_text=a.embed_text,
                        payload=_merge_payload(a.payload, b.payload, src, tgt),
                        construction=CONSTRUCTION_DONOR_CROSS_V1,
                        donor_example_id=b.example_id,
                    )
                    # similarity of the negative ~ the donor-a query similarity (query-relevant)
                    selected.append((neg, sim_by_id.get(a.example_id, float("nan"))))
                    if len(selected) >= n_neg:
                        break
            if len(selected) >= n_neg:
                return selected[:n_neg], None
            # not enough hard negatives
            if not cfg.fallback.allow_random_negatives:
                reason = f"hard negatives insufficient ({len(selected)}/{n_neg}); random disallowed"
                return selected, reason
            rand_needed = n_neg - len(selected)
            rand, _ = self._random_negatives(kind, rand_needed, excl_keys | used_keys, known_pos_keys, cfg)
            reason = f"hard negatives insufficient; padded {len(rand)} random"
            return selected + rand, (reason if rand else
                                     f"hard negatives insufficient ({len(selected)}/{n_neg})")

        # STATIC_RANDOM (or any random path)
        rand, reason = self._random_negatives(kind, n_neg, excl_keys, known_pos_keys, cfg)
        return rand, reason

    def _random_negatives(self, kind, n_neg, excl_keys, known_pos_keys, cfg):
        """Random column-swap negatives from the positive pool's entities (seeded)."""
        pool = self.corpus.positive_pool(kind)
        srcs = [ex.src_iri for ex in pool]
        tgts = [ex.tgt_iri for ex in pool]
        if not srcs or not tgts:
            return [], f"no entities to form random negatives for {kind.value}"
        rng = _random.Random(cfg.seed + 1)
        selected: list = []
        used: set = set()
        attempts = 0
        max_attempts = max(50, n_neg * 50)
        while len(selected) < n_neg and attempts < max_attempts:
            attempts += 1
            src = rng.choice(srcs)
            tgt = rng.choice(tgts)
            if src == tgt:
                continue
            key = frozenset({src, tgt})
            if key in known_pos_keys or key in excl_keys or key in used:
                continue
            used.add(key)
            neg = Example(
                example_id=f"neg:rand:{len(selected)}:{_pairid(src, tgt)}",
                src_iri=src, tgt_iri=tgt, kind=kind, label=False,
                source=Source.CONSTRUCTED, embed_text=f"{src} {tgt}", payload=(src, tgt),
                construction=CONSTRUCTION_RANDOM,
            )
            selected.append((neg, float("nan")))
        reason = None if len(selected) == n_neg else f"random negatives insufficient ({len(selected)}/{n_neg})"
        return selected, reason

    # rendering + token budget

    def _render_with_budget(self, pairs, cfg, *, paired: bool = False):
        """Render each selected pair (forward + reverse if bidirectional); enforce token budget."""
        refs: list = []
        tokens_used = 0
        budget = cfg.token_budget
        budget_reason = None
        rank = 0
        # Under a paired layout, `pairs` is [P1, N1, P2, N2, ...] and the block invariant is
        # complete minimal pairs, so the greedy budget stop must not cut mid-pair: track the
        # last complete (Pi, Ni) boundary and roll back to it.
        pair_start_refs = 0
        pair_start_tokens = 0
        for position, (ex, label, sim) in enumerate(pairs):
            if paired and position % 2 == 0:
                pair_start_refs = len(refs)
                pair_start_tokens = tokens_used
            answer = cfg.answer_pos if label else cfg.answer_neg
            directions = [Direction.FORWARD]
            if cfg.bidirectional:
                directions.append(Direction.REVERSE)
            for direction in directions:
                if direction == Direction.FORWARD:
                    prompt = self.render_fn(ex.src_iri, ex.tgt_iri, ex.payload)
                else:
                    prompt = self.render_fn(ex.tgt_iri, ex.src_iri, ex.payload)
                cost = self.token_counter.count(prompt) + self.token_counter.count(answer)
                if budget is not None and tokens_used + cost > budget:
                    # greedy stop; never truncates the system prompt or the live query
                    if paired:
                        refs = refs[:pair_start_refs]
                        tokens_used = pair_start_tokens
                    budget_reason = (f"token budget {budget} reached at {tokens_used} tokens; "
                                     f"included {len(refs)} example(s)")
                    return refs, tokens_used, budget_reason
                refs.append(ExampleRef(
                    example_id=ex.example_id, src_iri=ex.src_iri, tgt_iri=ex.tgt_iri, kind=ex.kind,
                    label=label, source=ex.source, direction=direction, rank=rank, similarity=sim,
                    prompt_text=prompt, answer_text=answer, tokens=cost,
                    **ex.provenance(),
                ))
                tokens_used += cost
                rank += 1
        return refs, tokens_used, budget_reason


# helpers

def _num_pairs(k: int, bidirectional: bool) -> int:
    """Number of entity pairs needed; bidirectional yields 2 rendered examples per pair."""
    return math.ceil(k / 2) if bidirectional else k


def _merge_payload(pa, pb, src, tgt):
    # for constructed negatives, prefer a payload that lets render_fn resolve (src, tgt)
    if isinstance(pa, dict) and isinstance(pb, dict):
        return {"src": pa.get("src", src), "tgt": pb.get("tgt", tgt)}
    return (src, tgt)


def _pairid(src: str, tgt: str) -> str:
    return hashlib.blake2b(f"{src}\x1f{tgt}".encode(), digest_size=5).hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
