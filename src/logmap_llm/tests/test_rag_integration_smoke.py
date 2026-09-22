"""
Local smoke test: the query-specific RAG retriever wired into the live pipeline, exercised
end-to-end with real owlready2 ontologies and the real prompt templates.

Validates, on a mixed-type M_ask (CLS + OPROP + DPROP): per-query typed retrieval; DPROP
examples render their datatype range ("has data values of type ..."), not object-property
phrasing ("connects ... to ..."); fixed-seed determinism; no cross-query contamination
(an M_ask pair never appears as an example); and all few-shot ablation modes are runnable.

Requires owlready2; skips in environments without it.
Run: python -m pytest logmap_llm/tests/test_rag_integration_smoke.py -q
"""
from __future__ import annotations

import datetime
import pytest

owlready2 = pytest.importorskip("owlready2")

import pandas as pd
from logmap_llm.ontology.access import load_ontologies
from logmap_llm.ontology.vocabularies import get_preset
import logmap_llm.oracle.prompts.templates as opb
from logmap_llm.oracle.prompts.context import PromptContext
from logmap_llm.pipeline.rag_fewshot import build_query_specific_few_shot

SRC = "http://ex.org/ragsmoke/src#"
TGT = "http://ex.org/ragsmoke/tgt#"


def _build_onto(iri, names, oprops, dprops, path):
    onto = owlready2.get_ontology(iri)
    made = {}
    with onto:
        for n in names:
            made[n] = type(n, (owlready2.Thing,), {})
        for pn, dom, rng in oprops:
            p = type(pn, (owlready2.ObjectProperty,), {})
            p.domain = [made[dom]]; p.range = [made[rng]]; made[pn] = p
        for pn, dom, rng in dprops:
            p = type(pn, (owlready2.DataProperty,), {})
            p.domain = [made[dom]]; p.range = [rng]; made[pn] = p
    for n, ent in made.items():
        ent.label = [n.lower()]
    onto.save(file=str(path), format="rdfxml")
    return path


def _row(s, t, et):
    return [SRC + s, TGT + t, "=", 0.95, et]


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    d = tmp_path_factory.mktemp("ragsmoke")
    srcp = _build_onto(SRC,
        ["Paper", "Person", "Institution", "Topic", "Review"],
        [("authorOf", "Person", "Paper"), ("affiliatedWith", "Person", "Institution"),
         ("hasTopic", "Paper", "Topic")],
        [("title", "Paper", str), ("pageCount", "Paper", int), ("birthDate", "Person", datetime.date)],
        d / "src.owl")
    tgtp = _build_onto(TGT,
        ["Article", "Author", "Organization", "Subject", "Evaluation"],
        [("writes", "Author", "Article"), ("worksAt", "Author", "Organization"),
         ("aboutSubject", "Article", "Subject")],
        [("name", "Article", str), ("numPages", "Article", int), ("dob", "Author", datetime.date)],
        d / "tgt.owl")
    OA_source, OA_target = load_ontologies(str(srcp), str(tgtp), cache_dir=None, vocabulary=get_preset("default"))

    initial = pd.DataFrame([
        _row("Paper", "Article", "CLS"), _row("Person", "Author", "CLS"),
        _row("Institution", "Organization", "CLS"), _row("Topic", "Subject", "CLS"),
        _row("authorOf", "writes", "OPROP"), _row("affiliatedWith", "worksAt", "OPROP"),
        _row("title", "name", "DPROP"), _row("pageCount", "numPages", "DPROP"),
        _row("Review", "Evaluation", "CLS"),        # CLS query (also an initial-alignment row)
        _row("hasTopic", "aboutSubject", "OPROP"),  # OPROP query
        _row("birthDate", "dob", "DPROP"),          # DPROP query (datatype: date)
    ])
    m_ask = pd.DataFrame([
        _row("Review", "Evaluation", "CLS"),
        _row("hasTopic", "aboutSubject", "OPROP"),
        _row("birthDate", "dob", "DPROP"),
    ])
    return OA_source, OA_target, initial, m_ask


#: These tests render real templates, so they need a prompt context like production does.
_CTX = PromptContext()


def _cls_fn():
    return opb.get_oracle_user_prompt_template_function("synonyms_only", _CTX)


def _prop_fn():
    return opb.get_oracle_user_prompt_template_function("prop_domain_range", _CTX)


def _build(loaded, strategy="query-rag", k=2, seed=42, negative_layout="donor-cross-v1"):
    OA_source, OA_target, initial, m_ask = loaded
    return build_query_specific_few_shot(
        mappings=initial, m_ask_df=m_ask, OA_source=OA_source, OA_target=OA_target,
        cls_fn=_cls_fn(), property_fn=_prop_fn(), data_property_fn=None, instance_fn=None,
        strategy=strategy, k=k, seed=seed, bidirectional=False,
        answer_format="true_false", response_mode="structured", prompt_family="synonyms_only",
        negative_layout=negative_layout,
        dataset_sha="smoke",
    )


def _keys(m_ask):
    return {et: SRC + s + "|" + TGT + t for s, t, et in
            [("Review", "Evaluation", "CLS"), ("hasTopic", "aboutSubject", "OPROP"), ("birthDate", "dob", "DPROP")]}


def test_per_query_typed_retrieval_and_pt2_datatype(loaded):
    per_query, traces = _build(loaded, strategy="query-rag", k=2)
    k = _keys(loaded[3])

    # every query got its k examples, typed by the query's entity type
    for et, key in k.items():
        assert traces[key]["entity_type"] == et
        assert len(per_query[key]) == 2, (et, len(per_query[key]))

    cls_examples = " ".join(u for u, _ in per_query[k["CLS"]])
    oprop_examples = " ".join(u for u, _ in per_query[k["OPROP"]])
    dprop_examples = " ".join(u for u, _ in per_query[k["DPROP"]])

    # CLS query -> class template ("The first one is ...")
    assert "The first one is" in cls_examples
    # OPROP query -> object-property phrasing, not datatype phrasing
    assert "connects" in oprop_examples and "has data values of type" not in oprop_examples
    # DPROP query -> datatype phrasing, not object-property "connects ... to"
    assert "has data values of type" in dprop_examples
    assert "connects" not in dprop_examples


def test_determinism_same_seed(loaded):
    a_pq, a_tr = _build(loaded, strategy="query-rag", k=2, seed=42)
    b_pq, b_tr = _build(loaded, strategy="query-rag", k=2, seed=42)
    assert a_pq == b_pq                                   # identical rendered examples
    for key in a_tr:
        assert a_tr[key]["selected"] == b_tr[key]["selected"]   # identical selected ids/ranks


def test_no_cross_query_contamination(loaded):
    per_query, traces = _build(loaded, strategy="query-rag", k=2)
    k = _keys(loaded[3])
    # each M_ask pair (Review/Evaluation, hasTopic/aboutSubject, birthDate/dob) must never appear as an
    # example — they are M_ask, so excluded from the corpus, and each query also excludes itself+reverse
    m_ask_local = [("review", "evaluation"), ("hastopic", "aboutsubject"), ("birthdate", "dob")]
    for key, pairs in per_query.items():
        blob = " ".join(u for u, _ in pairs).lower()
        for s, t in m_ask_local:
            assert not (s in blob and t in blob), (key, s, t)


def test_ablation_modes_and_k(loaded):
    # zero-shot / k=0 -> no examples
    for strat, k in [("zero-shot", 2), ("query-rag", 0)]:
        pq, _ = _build(loaded, strategy=strat, k=k)
        assert all(len(v) == 0 for v in pq.values()), (strat, k)
    # static + query modes at k in {1,2} -> examples produced
    for strat in ["static-random", "static-hard", "query-rag"]:
        for k in (1, 2):
            pq, _ = _build(loaded, strategy=strat, k=k)
            assert all(len(v) == k for v in pq.values()), (strat, k, {kk: len(v) for kk, v in pq.items()})
