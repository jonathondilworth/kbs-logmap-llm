"""Unit tests for the owlready2-free RAG pipeline adapter (embed_text/render are injected)."""
from __future__ import annotations

import pandas as pd
import pytest

from logmap_llm.oracle.rag import (
    EntityKind, Mode, Source, mode_from_strategy, m_ask_exclusion_keys,
    build_typed_corpus_from_anchors, query_mappings_from_m_ask,
    build_retriever_from_pipeline, HashingEncoder,
)

# initial alignment df: src|tgt|rel|conf|entityType (5 cols)
INIT = pd.DataFrame([
    ["http://a#C1", "http://b#C1", "=", 0.99, "CLS"],
    ["http://a#C2", "http://b#C2", "=", 0.98, "CLS"],
    ["http://a#P1", "http://b#P1", "=", 0.97, "OPROP"],
    ["http://a#D1", "http://b#D1", "=", 0.96, "DPROP"],
    ["http://a#I1", "http://b#I1", "=", 0.95, "INST"],
    ["http://a#Cx", "http://b#Cx", "=", 0.80, "CLS"],   # this one is also in M_ask -> excluded
    ["http://a#Sub", "http://b#Super", "<", 0.99, "CLS"],
])
MASK = pd.DataFrame([
    ["http://a#Cx", "http://b#Cx", "=", 0.80, "CLS"],
    ["http://a#Pq", "http://b#Pq", "=", 0.79, "OPROP"],
])


def _embed(s, t, kind):
    return f"{s.rsplit('#',1)[-1]} {t.rsplit('#',1)[-1]} {kind.value}"


def _render(s, t, payload):
    return f"{s} <=> {t}"


def test_mode_from_strategy():
    assert mode_from_strategy("hard") == Mode.STATIC_HARD
    assert mode_from_strategy("random") == Mode.STATIC_RANDOM
    assert mode_from_strategy("hard-similar") == Mode.QUERY_RAG
    assert mode_from_strategy("zero-shot") == Mode.ZERO_SHOT
    with pytest.raises(ValueError):
        mode_from_strategy("nonsense")


def test_exclusion_keys():
    keys = m_ask_exclusion_keys(MASK)
    assert frozenset({"http://a#Cx", "http://b#Cx"}) in keys
    assert len(keys) == 2


def test_typed_corpus_from_anchors_types_and_excludes():
    c = build_typed_corpus_from_anchors(INIT, MASK, _embed, dataset_sha="s")
    # Cx is in M_ask -> excluded from the CLS anchor pool; C1,C2 remain
    cls_ids = {(e.src_iri, e.tgt_iri) for e in c.positive_pool(EntityKind.CLS)}
    assert ("http://a#C1", "http://b#C1") in cls_ids
    assert ("http://a#Cx", "http://b#Cx") not in cls_ids
    assert ("http://a#Sub", "http://b#Super") not in cls_ids
    # each kind lands in its own pool
    assert len(c.positive_pool(EntityKind.CLS)) == 2
    assert len(c.positive_pool(EntityKind.OPROP)) == 1
    assert len(c.positive_pool(EntityKind.DPROP)) == 1
    assert len(c.positive_pool(EntityKind.INST)) == 1
    # anchors are pseudo-labels
    assert all(e.source == Source.ANCHOR for e in c.positive_pool(EntityKind.CLS))


def test_query_mappings_from_m_ask():
    qms = query_mappings_from_m_ask(MASK, _embed)
    assert len(qms) == 2
    key0, qm0 = qms[0]
    assert key0 == "http://a#Cx|http://b#Cx"
    assert qm0.kind == EntityKind.CLS
    assert qms[1][1].kind == EntityKind.OPROP


def test_build_retriever_end_to_end():
    r = build_retriever_from_pipeline(
        INIT, MASK, encoder=HashingEncoder(dim=64), render_fn=_render, embed_text_fn=_embed,
        strategy="hard-similar", k=2, negative_layout="donor-cross-v1", dataset_sha="s")
    _key, qm = query_mappings_from_m_ask(MASK, _embed)[0]  # a CLS query
    res = r.retrieve(qm, qm.kind, "=", "syn", k=2, corpus_id="s",
                     exclude_keys=m_ask_exclusion_keys(MASK))
    assert all(e.kind == EntityKind.CLS for e in res.examples)  # typed pool honoured
    # never returns an M_ask pair
    mask_keys = m_ask_exclusion_keys(MASK)
    assert all(frozenset({e.src_iri, e.tgt_iri}) not in mask_keys for e in res.examples)
