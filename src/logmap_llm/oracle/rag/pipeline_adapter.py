"""
logmap_llm.oracle.rag.pipeline_adapter

Glue between the LogMapLLM pipeline (initial alignment + M_ask DataFrames, owlready2 ontology
access, the prompt-template registry) and the owlready2-free RagRetriever. The functions here
that touch only DataFrames are unit-testable on CPU; the ontology-dependent embed_text/render
callables are injected by stage_two (production), keeping this module import-light.
"""
from __future__ import annotations

from typing import Callable, Optional
import os

import pandas as pd

from .types import EntityKind, Mode, Source, QueryMapping, RagConfig, FallbackPolicy
from .corpus import TypedCorpus
from .retriever import RagRetriever


# legacy few_shot_negative_strategy -> RAG Mode (the allowed set is validated upstream)
_STRATEGY_TO_MODE = {
    "hard": Mode.STATIC_HARD,
    "random": Mode.STATIC_RANDOM,
    "static-hard": Mode.STATIC_HARD,
    "static-random": Mode.STATIC_RANDOM,
    "hard-similar": Mode.QUERY_RAG,
    "query-rag": Mode.QUERY_RAG,
    "zero-shot": Mode.ZERO_SHOT,
}


def mode_from_strategy(strategy: str) -> Mode:
    if strategy not in _STRATEGY_TO_MODE:
        raise ValueError(f"Unknown few-shot strategy {strategy!r}; expected {sorted(_STRATEGY_TO_MODE)}")
    return _STRATEGY_TO_MODE[strategy]


def m_ask_exclusion_keys(m_ask_df: pd.DataFrame) -> set:
    """Complete-IRI pair keys for every M_ask candidate (excluded from retrieval as leakage)."""
    keys = set()
    for _, row in m_ask_df.iterrows():
        keys.add(frozenset({str(row.iloc[0]), str(row.iloc[1])}))
    return keys


def build_typed_corpus_from_anchors(
    initial_alignment_df: pd.DataFrame,
    m_ask_df: pd.DataFrame,
    embed_text_fn: Callable[[str, str, EntityKind], str],
    payload_fn: Optional[Callable[[str, str, EntityKind], object]] = None,
    dataset_sha: str = "",
    language: str = "en",
    train_tsv_path: Optional[str] = None,
    train_kind: EntityKind = EntityKind.CLS,
) -> TypedCorpus:
    """
    Build a typed corpus of positive few-shot examples from:
      - LogMap high-confidence anchors = initial-alignment rows not in M_ask (Source.ANCHOR,
        pseudo-label), typed by the alignment's entityType column into distinct
        CLS/OPROP/DPROP/INST pools;
      - an optional authorised training alignment train.tsv (Source.GOLD) added to `train_kind`.
    The M_ask candidates themselves are never added (leakage). Only authorised sources are used.
    """
    corpus = TypedCorpus(dataset_sha=dataset_sha, language=language)
    excl = m_ask_exclusion_keys(m_ask_df)

    # group anchors by kind
    per_kind: dict = {k: [] for k in EntityKind}
    for _, row in initial_alignment_df.iterrows():
        src, tgt = str(row.iloc[0]), str(row.iloc[1])
        # Few-shot positives are equivalence anchors. Do not silently treat a
        # subsumption/disjointness row as a positive equivalence demonstration.
        if len(row) > 2 and str(row.iloc[2]).strip() != "=":
            continue
        if frozenset({src, tgt}) in excl:
            continue
        etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
        try:
            kind = EntityKind.coerce(etype)
        except ValueError:
            continue  # UNKNO / unexpected -> skip (never silently mistype)
        per_kind[kind].append((src, tgt))

    for kind, pairs in per_kind.items():
        if not pairs:
            continue
        corpus.add_pairs(
            pairs, kind=kind, source=Source.ANCHOR,
            embed_text_fn=lambda s, t, K=kind: embed_text_fn(s, t, K),
            payload_fn=(lambda s, t, K=kind: payload_fn(s, t, K)) if payload_fn else None,
        )

    if train_tsv_path:
        # A supplied-but-missing path raises rather than silently yielding a corpus
        # without the requested gold examples (matches rag_dataset_fingerprint).
        if not os.path.isfile(str(train_tsv_path)):
            raise FileNotFoundError(
                f"few-shot training alignment does not exist: {train_tsv_path}"
            )
        tdf = pd.read_csv(train_tsv_path, sep="\t", header=None, usecols=[0, 1])
        pairs = [(str(a), str(b)) for a, b in zip(tdf.iloc[:, 0], tdf.iloc[:, 1])
                 if frozenset({str(a), str(b)}) not in excl]
        if pairs:
            corpus.add_pairs(
                pairs, kind=train_kind, source=Source.GOLD,
                embed_text_fn=lambda s, t, K=train_kind: embed_text_fn(s, t, K),
                payload_fn=(lambda s, t, K=train_kind: payload_fn(s, t, K)) if payload_fn else None,
            )
    return corpus


def query_mappings_from_m_ask(
    m_ask_df: pd.DataFrame,
    embed_text_fn: Callable[[str, str, EntityKind], str],
    pairs_separator: str = "|",
) -> list:
    """Return [(key, QueryMapping)] for each M_ask row; key = 'src<sep>tgt' (matches consultation keys)."""
    out = []
    for _, row in m_ask_df.iterrows():
        src, tgt = str(row.iloc[0]), str(row.iloc[1])
        etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
        try:
            kind = EntityKind.coerce(etype)
        except ValueError as exc:
            # An unmappable entity type raises rather than silently retrieving from the
            # CLS pool, matching the strict prebuilt path
            # (rag_fewshot.load_prebuilt_few_shot_bundle).
            raise ValueError(
                f"M_ask row ({src}, {tgt}) has entity type {etype!r}, which maps to no "
                "typed retrieval pool; refusing to silently retrieve CLS examples for it"
            ) from exc
        key = f"{src}{pairs_separator}{tgt}"
        out.append((key, QueryMapping(src_iri=src, tgt_iri=tgt, kind=kind,
                                      embed_text=embed_text_fn(src, tgt, kind))))
    return out


def build_retriever_from_pipeline(
    initial_alignment_df: pd.DataFrame,
    m_ask_df: pd.DataFrame,
    encoder,
    render_fn,
    embed_text_fn: Callable[[str, str, EntityKind], str],
    strategy: str,
    k: int,
    *,
    negative_layout: str,
    seed: int = 42,
    token_budget: Optional[int] = None,
    answer_pos: str = '{"answer": true}',
    answer_neg: str = '{"answer": false}',
    bidirectional: bool = False,
    dataset_sha: str = "",
    language: str = "en",
    train_tsv_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    token_counter=None,
    payload_fn: Optional[Callable[[str, str, EntityKind], object]] = None,
    fallback: Optional[FallbackPolicy] = None,
    sibling_fn=None,
    sibling_strategy: str = "",
    sibling_encoder_revision: str = "",
    sibling_candidate_count: int = 8,
) -> RagRetriever:
    """Build the retriever for one pipeline run.

    ``negative_layout`` is keyword-only with no default: a caller that forgets it gets a
    TypeError rather than a silent default (``plan._validate_native`` enforces the
    campaign side).
    """
    corpus = build_typed_corpus_from_anchors(
        initial_alignment_df, m_ask_df, embed_text_fn, payload_fn=payload_fn,
        dataset_sha=dataset_sha, language=language, train_tsv_path=train_tsv_path)
    cfg = RagConfig(
        mode=mode_from_strategy(strategy), k=k, seed=seed, token_budget=token_budget,
        answer_pos=answer_pos, answer_neg=answer_neg, bidirectional=bidirectional,
        fallback=fallback or FallbackPolicy(),
        negative_layout=negative_layout,
        sibling_strategy=sibling_strategy,
        sibling_encoder_revision=sibling_encoder_revision,
        sibling_candidate_count=sibling_candidate_count,
    )
    return RagRetriever(corpus=corpus, encoder=encoder, render_fn=render_fn, config=cfg,
                        token_counter=token_counter, cache_dir=cache_dir,
                        sibling_fn=sibling_fn)
