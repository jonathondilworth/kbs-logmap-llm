"""
Regression tests for loud-failure and determinism guarantees across the pipeline.
CPU-only except where noted (the property-characteristics tests need owlready2,
which the pipeline env provides).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from logmap_llm.config.schema import FewShotConfig


# --------------------------------------------------------------------------
# Few-shot artifact key coverage (orchestration)
# --------------------------------------------------------------------------

def _load_artifact(tmp_path, payload, expected_keys, prebuilt=None):
    from logmap_llm.pipeline.orchestration import _load_few_shot_artifact

    artifact = tmp_path / "few_shot.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    cfg = FewShotConfig(
        few_shot_k=4, rag_encoder_kind="hashing", prebuilt_few_shot_bundle_path=prebuilt,
    )
    return _load_few_shot_artifact(
        artifact, few_shot_cfg=cfg, oracle_cfg=None, expected_query_keys=expected_keys,
    )


def test_few_shot_artifact_empty_dict_fails_loudly(tmp_path):
    """The {} sentinel from a k=0 run / failed RAG generation must not survive reuse."""
    with pytest.raises(ValueError, match="do not cover the prompt keys"):
        _load_artifact(tmp_path, {}, expected_keys={"a|b"})


def test_few_shot_artifact_partial_dict_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="missing="):
        _load_artifact(tmp_path, {"a|b": [["u", "v"]]}, expected_keys={"a|b", "c|d"})


def test_few_shot_artifact_extra_keys_fail_loudly(tmp_path):
    with pytest.raises(ValueError, match="extra="):
        _load_artifact(
            tmp_path, {"a|b": [["u", "v"]], "stale|key": []}, expected_keys={"a|b"},
        )


def test_few_shot_artifact_exact_coverage_loads(tmp_path):
    loaded = _load_artifact(
        tmp_path, {"a|b": [["u", "v"]], "c|d": []}, expected_keys={"a|b", "c|d"},
    )
    assert loaded == {"a|b": [("u", "v")], "c|d": []}


def test_few_shot_artifact_empty_list_fails_loudly(tmp_path):
    with pytest.raises(ValueError, match="empty list"):
        _load_artifact(tmp_path, [], expected_keys={"a|b"})


def test_few_shot_artifact_legacy_list_loads(tmp_path):
    assert _load_artifact(tmp_path, [["u", "v"]], expected_keys={"a|b"}) == [("u", "v")]


# --------------------------------------------------------------------------
# Non-object JSON answers fall back to the plain parser
# --------------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        choice = SimpleNamespace(
            message=SimpleNamespace(content=self._content),
            finish_reason="stop",
        )
        return SimpleNamespace(choices=[choice], usage=None)


def _create_manager(content):
    from logmap_llm.constants import BinaryOutputFormat, InteractionStyle
    from logmap_llm.oracle.manager import OracleConsultationManager

    mgr = OracleConsultationManager.__new__(OracleConsultationManager)
    mgr.response_format = BinaryOutputFormat
    mgr.model_name = "test-model"
    mgr.temperature = 0.0
    mgr.top_p = 1.0
    mgr.seed = None
    mgr.max_completion_tokens = 16
    mgr.interaction_style = InteractionStyle.LOCAL_VLLM
    mgr.enable_thinking = None
    mgr.supports_chat_template_kwargs = False
    mgr.logprobs = False
    mgr.top_logprobs = None
    mgr._frozen = True
    mgr._frozen_messages = ()
    mgr.client = SimpleNamespace(
        chat=SimpleNamespace(completions=_FakeCompletions(content))
    )
    return mgr


@pytest.mark.parametrize(
    "content,expected",
    [("true", True), ("false", False), ('"Yes"', True), ('["no"]', False)],
)
def test_create_path_bare_json_answers_parse(content, expected):
    """json.loads succeeds on these, but ** splat would TypeError; the plain-text
    fallback must run instead of surfacing an ('error', NaN) consultation."""
    out = _create_manager(content)._consult_via_create("prompt")
    assert out.parsed.answer is expected


def test_create_path_object_json_still_parses_directly():
    out = _create_manager('{"answer": true}')._consult_via_create("prompt")
    assert out.parsed.answer is True


# --------------------------------------------------------------------------
# Paired layouts keep later complete pairs
# --------------------------------------------------------------------------

def _paired_retriever(fallback=None):
    from logmap_llm.oracle.rag import (
        EntityKind, FallbackPolicy, HashingEncoder, Mode, RagConfig, RagRetriever,
        Source, TypedCorpus,
    )

    corpus = TypedCorpus(dataset_sha="pair-drop")
    # P1 = (S, T): its donor negative would need target != {S, T} with src S — the only
    # other donor target is S itself, so no negative exists for P1.
    # P2 = (A, S): P1's target T is an eligible wraparound donor target -> (A, T) is valid.
    corpus.add_pairs(
        [("urn:S", "urn:T"), ("urn:A", "urn:S")],
        kind=EntityKind.CLS, source=Source.ANCHOR,
        embed_text_fn=lambda s, t: f"{s} {t}",
    )
    cfg = RagConfig(
        mode=Mode.QUERY_RAG, k=4, negative_layout="paired-donor-v2",
        fallback=fallback or FallbackPolicy(),
    )
    return RagRetriever(
        corpus=corpus, encoder=HashingEncoder(dim=64),
        render_fn=lambda s, t, payload: f"{s} <=> {t}", config=cfg,
    )


def _paired_query():
    from logmap_llm.oracle.rag import EntityKind, QueryMapping

    return QueryMapping(
        src_iri="urn:Q1", tgt_iri="urn:Q2", kind=EntityKind.CLS, embed_text="urn:S urn:T",
    )


def test_paired_layout_keeps_later_complete_pairs():
    result = _paired_retriever().retrieve(
        _paired_query(), "CLS", "=", "family", k=4, corpus_id="pair-drop",
    )
    # P1 has no eligible negative; the valid (P2, N2) pair behind it must still be emitted.
    assert len(result.examples) == 2
    positive, negative = result.examples
    assert positive.label is True and negative.label is False
    assert negative.src_iri == positive.src_iri  # the D-d pairing invariant


def test_paired_layout_strict_policy_reports_true_pair_count():
    from logmap_llm.oracle.rag import FallbackPolicy, RagRetrievalError

    # on_undersized='error' still fires (2 of 4 pairs); the error must report the true count.
    retriever = _paired_retriever(FallbackPolicy(on_undersized="error"))
    with pytest.raises(RagRetrievalError, match="Only 2 of 4"):
        retriever.retrieve(_paired_query(), "CLS", "=", "family", k=4, corpus_id="x")


# --------------------------------------------------------------------------
# Truncation length is part of the index-cache identity
# --------------------------------------------------------------------------

def test_cls_pool_preprocessing_version_encodes_max_length():
    from logmap_llm.oracle.rag.encoder import cls_pool_preprocessing_version

    assert cls_pool_preprocessing_version(64) == "cls-pool-max64-v1"
    assert cls_pool_preprocessing_version(64) != cls_pool_preprocessing_version(32)


# --------------------------------------------------------------------------
# Pipeline adapter fails loudly on bad entity types / missing training data
# --------------------------------------------------------------------------

def test_unknown_entity_type_raises_instead_of_cls():
    from logmap_llm.oracle.rag import query_mappings_from_m_ask

    m_ask = pd.DataFrame([["http://a#X", "http://b#X", "=", 0.8, "UNKNO"]])
    with pytest.raises(ValueError, match="UNKNO"):
        query_mappings_from_m_ask(m_ask, lambda s, t, k: f"{s} {t}")


def test_missing_train_tsv_raises():
    from logmap_llm.oracle.rag import build_typed_corpus_from_anchors

    init = pd.DataFrame([["http://a#C", "http://b#C", "=", 0.9, "CLS"]])
    m_ask = pd.DataFrame(columns=range(5))
    with pytest.raises(FileNotFoundError, match="training alignment"):
        build_typed_corpus_from_anchors(
            init, m_ask, lambda s, t, k: f"{s} {t}",
            train_tsv_path="/nonexistent/train.tsv",
        )


# --------------------------------------------------------------------------
# Property characteristics come from prop.is_a (needs owlready2)
# --------------------------------------------------------------------------

def test_property_characteristics_read_from_is_a():
    owlready2 = pytest.importorskip("owlready2")
    from logmap_llm.ontology.object import PropertyEntity

    world = owlready2.World()
    try:
        onto = world.get_ontology("http://example.org/chars-test#")
        with onto:
            class relatesTo(owlready2.ObjectProperty, owlready2.TransitiveProperty,
                            owlready2.SymmetricProperty):
                pass

            class plainRel(owlready2.ObjectProperty):
                pass

        declared = PropertyEntity.__new__(PropertyEntity)
        declared.prop = relatesTo
        assert declared.get_characteristics() == ["transitive", "symmetric"]

        plain = PropertyEntity.__new__(PropertyEntity)
        plain.prop = plainRel
        assert plain.get_characteristics() == []
    finally:
        world.close()


def test_undeclared_property_has_no_characteristics():
    pytest.importorskip("owlready2")
    from logmap_llm.ontology.object import PropertyEntity, _URIBackedUndeclaredProperty

    entity = PropertyEntity.__new__(PropertyEntity)
    entity.prop = _URIBackedUndeclaredProperty("http://a#p", "p")
    assert entity.get_characteristics() == []


# --------------------------------------------------------------------------
# The injected PYTHONPATH source root can actually import logmap_llm
# --------------------------------------------------------------------------

def test_child_environment_pythonpath_contains_package():
    from logmap_llm.experiments.run import _child_environment

    env = _child_environment()
    first_entry = env["PYTHONPATH"].split(":")[0]
    assert (Path(first_entry) / "logmap_llm" / "__init__.py").is_file()


# --------------------------------------------------------------------------
# Bidirectional prompt sets must be symmetric
# --------------------------------------------------------------------------

def test_reverse_coverage_asymmetry_raises():
    from logmap_llm.oracle.consultation import _require_reverse_prompt_coverage

    with pytest.raises(ValueError, match="forward-without-reverse"):
        _require_reverse_prompt_coverage({"a|b": "p", "a|b|REVERSE": "q", "c|d": "r"})
    with pytest.raises(ValueError, match="reverse-without-forward"):
        _require_reverse_prompt_coverage({"a|b": "p", "a|b|REVERSE": "q", "c|d|REVERSE": "r"})


def test_reverse_coverage_symmetric_passes():
    from logmap_llm.oracle.consultation import _require_reverse_prompt_coverage

    _require_reverse_prompt_coverage({"a|b": "p", "a|b|REVERSE": "q"})
    _require_reverse_prompt_coverage({})


# --------------------------------------------------------------------------
# Unknown token usage stays None, never a literal 0
# --------------------------------------------------------------------------

def test_sum_optional_tokens():
    from logmap_llm.oracle.consultation import _sum_optional_tokens

    assert _sum_optional_tokens(None, None) is None
    assert _sum_optional_tokens(None, 5) == 5
    assert _sum_optional_tokens(3, None) == 3
    assert _sum_optional_tokens(3, 5) == 8


# --------------------------------------------------------------------------
# A worker that raises is recorded as an 'error' for its key
# --------------------------------------------------------------------------

def test_raised_future_recorded_as_error(monkeypatch):
    import logmap_llm.oracle.consultation as oc

    def _boom(key, prompt, llm_oracle, developer_override=None, few_shot_examples=None):
        raise RuntimeError("worker bug")

    monkeypatch.setattr(oc, "consult_oracle_for_mapping", _boom)
    results = oc._run_consultations(
        llm_oracle=SimpleNamespace(), m_ask_prompts={"a|b": "p", "c|d": "q"},
        pair_entity_types={}, developer_prompt_map=None,
        max_workers=2, failure_tolerance=100, desc="test",
    )
    # both keys must be recorded as 'error', not absent (absent reads as 'skipped' downstream)
    assert results is not None
    assert results["a|b"][0] == "error" and results["c|d"][0] == "error"


# --------------------------------------------------------------------------
# on_budget_overflow is enforced; paired blocks never truncate mid-pair
# --------------------------------------------------------------------------

def _budget_retriever(budget, policy):
    from logmap_llm.oracle.rag import (
        EntityKind, FallbackPolicy, HashingEncoder, Mode, RagConfig, RagRetriever,
        Source, TypedCorpus,
    )

    corpus = TypedCorpus(dataset_sha="budget")
    corpus.add_pairs(
        [("urn:A1", "urn:B1"), ("urn:A2", "urn:B2"), ("urn:A3", "urn:B3")],
        kind=EntityKind.CLS, source=Source.ANCHOR,
        embed_text_fn=lambda s, t: f"{s} {t}",
    )
    cfg = RagConfig(
        mode=Mode.QUERY_RAG, k=4, negative_layout="paired-donor-v2",
        token_budget=budget, fallback=FallbackPolicy(on_budget_overflow=policy),
    )
    return RagRetriever(
        corpus=corpus, encoder=HashingEncoder(dim=64),
        render_fn=lambda s, t, payload: f"{s} <=> {t} padding padding padding", config=cfg,
    )


def _budget_query():
    from logmap_llm.oracle.rag import EntityKind, QueryMapping

    return QueryMapping(
        src_iri="urn:Q1", tgt_iri="urn:Q2", kind=EntityKind.CLS,
        embed_text="urn:A1 urn:B1",
    )


def test_budget_overflow_error_policy_raises():
    from logmap_llm.oracle.rag import RagRetrievalError

    with pytest.raises(RagRetrievalError, match="budget"):
        _budget_retriever(12, "error").retrieve(
            _budget_query(), "CLS", "=", "family", k=4, corpus_id="b")


def test_budget_overflow_zero_shot_policy_drops_examples():
    result = _budget_retriever(12, "zero_shot").retrieve(
        _budget_query(), "CLS", "=", "family", k=4, corpus_id="b")
    assert result.examples == ()
    assert result.trace.effective_mode == "zero_shot"
    assert "zero_shot" in (result.trace.fallback_reason or "")


def test_budget_reduce_k_truncates_at_pair_boundary():
    result = _budget_retriever(25, "reduce_k").retrieve(
        _budget_query(), "CLS", "=", "family", k=4, corpus_id="b")
    labels = [e.label for e in result.examples]
    # even count, strictly alternating (P, N) — never a dangling positive
    assert len(labels) % 2 == 0
    assert labels == [True, False] * (len(labels) // 2)


# --------------------------------------------------------------------------
# 'local' + structured mode is a config error, not per-row consult errors
# --------------------------------------------------------------------------

def test_local_generic_structured_rejected_at_construction():
    from logmap_llm.constants import BinaryOutputFormat
    from logmap_llm.oracle.manager import OracleConsultationManager

    with pytest.raises(ValueError, match="interaction_style='local'"):
        OracleConsultationManager(
            api_key="k", model_name="m", interaction_style="local",
            base_url="http://localhost:1", temperature=0.0, top_p=1.0,
            reasoning_effort=None, max_completion_tokens=8, enable_thinking=None,
            response_format=BinaryOutputFormat,
        )


def test_local_generic_plain_mode_still_constructs():
    from logmap_llm.constants import RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE
    from logmap_llm.oracle.manager import OracleConsultationManager

    mgr = OracleConsultationManager(
        api_key="k", model_name="m", interaction_style="local",
        base_url="http://localhost:1", temperature=0.0, top_p=1.0,
        reasoning_effort=None, max_completion_tokens=8, enable_thinking=None,
        response_format=RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE,
    )
    assert mgr.interaction_style.value == "local"


# --------------------------------------------------------------------------
# Typed literals cannot poison the annotation name sets
# --------------------------------------------------------------------------

def test_literal_text_coercion():
    rdflib = pytest.importorskip("rdflib")
    from logmap_llm.ontology.access import _literal_text

    xsd = rdflib.namespace.XSD
    assert _literal_text(rdflib.Literal("plain label")) == "plain label"
    assert _literal_text(rdflib.Literal("5", datatype=xsd.integer)) == "5"
    # ill-typed literal: .value is None; the lexical form survives
    assert _literal_text(rdflib.Literal("abc", datatype=xsd.integer)) == "abc"
    # everything is str, so min()-based deterministic selection cannot TypeError
    values = {_literal_text(rdflib.Literal("z")),
              _literal_text(rdflib.Literal("5", datatype=xsd.integer))}
    assert min(values) == "5"


# --------------------------------------------------------------------------
# enable_thinking=None resolves a response format
# --------------------------------------------------------------------------

def test_enable_thinking_none_resolves_response_format():
    from logmap_llm.config.schema import OracleConfig
    from logmap_llm.constants import BinaryOutputFormat

    cfg = OracleConfig(model_name="test-model", enable_thinking=None)
    assert cfg.response_format is BinaryOutputFormat


# --------------------------------------------------------------------------
# Degenerate stratification is tagged as a fallback, not as stratified
# --------------------------------------------------------------------------

def test_unstratified_fallback_is_tagged(tmp_path):
    from logmap_llm.utils.misc import compute_conference_m1_m2_stratified

    sysf = tmp_path / "sys.tsv"
    sysf.write_text("http://a#C1\thttp://b#C2\n")
    ref = tmp_path / "ref.tsv"
    ref.write_text("http://a#C1\thttp://b#C2\n")
    (tmp_path / "ref_class.tsv").write_text("http://a#C1\thttp://b#C2\n")

    r = compute_conference_m1_m2_stratified(sysf, ref, initial_alignment_path=None)
    assert r["m1_class"]["source"] == "conference_unstratified_fallback"
    assert "stratification" in r["m1_class"]["metric_notes"]


def test_stratified_with_typing_keeps_stratified_tag(tmp_path):
    from logmap_llm.utils.misc import compute_conference_m1_m2_stratified

    sysf = tmp_path / "sys.tsv"
    sysf.write_text("http://a#C1\thttp://b#C2\n")
    ref = tmp_path / "ref.tsv"
    ref.write_text("http://a#C1\thttp://b#C2\n")
    (tmp_path / "ref_class.tsv").write_text("http://a#C1\thttp://b#C2\n")
    full = tmp_path / "full.txt"
    full.write_text("http://a#C1|http://b#C2|=|1.0|CLS\n")

    r = compute_conference_m1_m2_stratified(sysf, ref, initial_alignment_path=full)
    assert r["m1_class"]["source"] == "conference_stratified"


# --------------------------------------------------------------------------
# A wrong-separator alignment file cannot score as empty
# --------------------------------------------------------------------------

def test_load_mapping_pairs_wrong_separator_raises(tmp_path):
    from logmap_llm.evaluation.io import load_mapping_pairs

    pipe_file = tmp_path / "logmap.txt"
    pipe_file.write_text("http://a#X|http://b#X|=|1.0|CLS\nhttp://a#Y|http://b#Y|=|1.0|CLS\n")
    with pytest.raises(ValueError, match="no .* pairs parsed|refusing"):
        load_mapping_pairs(pipe_file)  # tab default vs pipe content
    # the explicit separator parses the same file fine
    assert load_mapping_pairs(pipe_file, sep="|") == {
        ("http://a#X", "http://b#X"), ("http://a#Y", "http://b#Y"),
    }


def test_load_mapping_pairs_empty_file_is_still_empty(tmp_path):
    from logmap_llm.evaluation.io import load_mapping_pairs

    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    assert load_mapping_pairs(empty) == set()


# --------------------------------------------------------------------------
# The Java boundary emits canonically sorted DataFrames
# --------------------------------------------------------------------------

class _FakeJavaMapping:
    def __init__(self, src, tgt, relation=-2, conf=0.9, etype=0):
        self._src, self._tgt = src, tgt
        self._rel, self._conf, self._etype = relation, conf, etype

    def getIRIStrEnt1(self):
        return self._src

    def getIRIStrEnt2(self):
        return self._tgt

    def getMappingDirection(self):
        return self._rel

    def getConfidence(self):
        return self._conf

    def getTypeOfMapping(self):
        return self._etype


class _FakeJavaHashSet:
    def __init__(self, items):
        self._items = list(items)

    def toArray(self):
        return self._items


def test_java_mappings_sorted_regardless_of_hashset_order():
    from logmap_llm.bridging import java_mappings_2_python

    mappings = [
        _FakeJavaMapping("http://a#Z", "http://b#Z"),
        _FakeJavaMapping("http://a#A", "http://b#A"),
        _FakeJavaMapping("http://a#M", "http://b#M"),
    ]
    df_one = java_mappings_2_python(_FakeJavaHashSet(mappings))
    df_two = java_mappings_2_python(_FakeJavaHashSet(reversed(mappings)))
    assert df_one.equals(df_two)
    assert list(df_one.iloc[:, 0]) == ["http://a#A", "http://a#M", "http://a#Z"]


def test_java_mappings_dedupe_lane_choice_is_deterministic():
    from logmap_llm.bridging import java_mappings_2_python

    # same URI pair emitted in both property lanes (etype 1=DPROP, 2=OPROP);
    # keep-first after the canonical sort always keeps DPROP, whatever the input order
    both = [
        _FakeJavaMapping("http://a#p", "http://b#p", etype=2),
        _FakeJavaMapping("http://a#p", "http://b#p", etype=1),
    ]
    for ordering in (both, list(reversed(both))):
        df = java_mappings_2_python(_FakeJavaHashSet(ordering), dedupe_uri_pairs=True)
        assert len(df) == 1
        assert df.iloc[0]["entityType"] == "DPROP"


# --------------------------------------------------------------------------
# Instance-index build failures raise and are memoised
# --------------------------------------------------------------------------

def test_instance_index_failure_raises_and_memoises(monkeypatch):
    import logmap_llm.ontology.sibling_retrieval as sr

    calls = {"n": 0}

    def _failing_build(onto):
        calls["n"] += 1
        raise RuntimeError("graph walk failed")

    monkeypatch.setattr(sr, "build_instance_type_index", _failing_build)
    selector = sr.SiblingSelector.__new__(sr.SiblingSelector)
    selector._instance_indexes = {}
    onto = object()
    with pytest.raises(RuntimeError, match="graph walk failed"):
        selector._instance_index(onto)
    with pytest.raises(RuntimeError, match="graph walk failed"):
        selector._instance_index(onto)
    assert calls["n"] == 1  # the failed build is memoised, not retried per query


# --------------------------------------------------------------------------
# Atomic text artifacts — no truncated file at the final path
# --------------------------------------------------------------------------

def test_atomic_write_text_strict_publishes_and_cleans_up(tmp_path):
    from logmap_llm.utils.io import atomic_write_text_strict

    target = tmp_path / "out.csv"
    atomic_write_text_strict(target, lambda fp: fp.write("a,b\n1,2\n"))
    assert target.read_text() == "a,b\n1,2\n"
    assert list(tmp_path.iterdir()) == [target]  # no leftover temp files


def test_atomic_write_text_strict_failure_leaves_no_artifact(tmp_path):
    from logmap_llm.utils.io import atomic_write_text_strict

    target = tmp_path / "out.csv"

    def _boom(fp):
        fp.write("partial")
        raise RuntimeError("writer died")

    with pytest.raises(RuntimeError, match="writer died"):
        atomic_write_text_strict(target, _boom)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------
# Parent/children name lists are deterministically ordered
# --------------------------------------------------------------------------

def test_attribute_relatives_names_sorted():
    pytest.importorskip("owlready2")
    from logmap_llm.ontology.object import ClassEntity

    class _Relative:
        def __init__(self, uri, names):
            self.annotation = {"uri": uri}
            self._names = names

        def _get_entry_names(self, name_type):
            return set(self._names)

    entity = ClassEntity.__new__(ClassEntity)
    entity.get_parents = lambda: {
        _Relative("urn:b", ["zeta", "alpha"]),
        _Relative("urn:a", ["mid"]),
        _Relative("urn:c", []),  # nameless -> falls back to its IRI
    }
    assert entity.get_attribute_relatives_names("parents", "preferred_names") == [
        ["mid"], ["alpha", "zeta"], ["urn:c"],
    ]
