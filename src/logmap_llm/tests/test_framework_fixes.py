"""
Regression tests for pipeline correctness fixes: M_ask dedup, atomic cache writes,
content-addressed run ids, few-shot config validation, null-with-reason oracle metrics,
full-alignment typing for Conference M1/M2, index-safe predictions CSV round-trips,
consultation failure-abort, per-query RAG few-shot dispatch, stratified-global
evaluation, and DPROP prompt routing/rendering.

CPU-only; deps: pandas, numpy, pydantic, rdflib, openai, tqdm (no owlready2/JVM/torch).
"""
from __future__ import annotations

import json
import threading

import pandas as pd
import pytest

from logmap_llm.evaluation.metrics import compute_oracle_metrics
from logmap_llm.utils.misc import compute_conference_m1_m2_stratified
from logmap_llm.oracle.consultation import _run_consultations
from logmap_llm.constants import BinaryOutputFormat, TokensUsage
from logmap_llm.config.schema import FewShotConfig
from logmap_llm.utils.data import dedupe_m_ask_by_uri_pair


# --------------------------------------------------------------------------
# kg-abox 'both'-policy artifact — M_ask deduped by (source,target) URI pair
# --------------------------------------------------------------------------

def _m_ask(rows):
    return pd.DataFrame(rows, columns=["source_entity_uri", "target_entity_uri",
                                       "relation", "confidence", "entityType"])


def test_dedupe_collapses_mixed_predicate_both_lanes():
    # a mixed predicate emitted in both lanes -> same (src,tgt), different entityType
    df = _m_ask([
        ["a#father", "b#father", "=", 0.8, "DPROP"],
        ["a#father", "b#father", "=", 0.8, "OPROP"],   # duplicate pair (both-lane)
        ["b#father", "a#father", "=", 0.8, "DPROP"],   # reverse pair -> preserved
        ["a#mother", "b#mother", "=", 0.9, "DPROP"],   # distinct
    ])
    out = dedupe_m_ask_by_uri_pair(df)
    assert len(out) == 3
    # first occurrence kept (DPROP), duplicate OPROP dropped
    father_fwd = out[(out.source_entity_uri == "a#father") & (out.target_entity_uri == "b#father")]
    assert len(father_fwd) == 1 and father_fwd.iloc[0]["entityType"] == "DPROP"
    # reverse pair preserved (directional dedup)
    assert ((out.source_entity_uri == "b#father") & (out.target_entity_uri == "a#father")).any()


def test_dedupe_noop_when_unique():
    df = _m_ask([["a#x", "b#x", "=", 0.8, "DPROP"], ["a#y", "b#y", "=", 0.8, "OPROP"]])
    assert len(dedupe_m_ask_by_uri_pair(df)) == 2


def test_dedupe_empty():
    df = _m_ask([])
    assert len(dedupe_m_ask_by_uri_pair(df)) == 0


# --------------------------------------------------------------------------
# atomic_json_write is safe under concurrent writers
# --------------------------------------------------------------------------

def test_cr03_atomic_write_concurrent_no_corruption(tmp_path):
    from logmap_llm.utils.io import atomic_json_write
    target = tmp_path / "cache.json"
    payload = list(range(200))
    N = 24

    def w(i):
        atomic_json_write(target, {"writer": i, "data": payload})

    threads = [threading.Thread(target=w, args=(i,)) for i in range(N)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # the final file is always valid json (never a corrupted/partial interleave) and a whole payload
    obj = json.loads(target.read_text())
    assert 0 <= obj["writer"] < N and obj["data"] == payload
    # unique temp files were all consumed/cleaned — none left behind
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.name.startswith("cache.json.") and p.name.endswith(".tmp")]
    assert leftovers == []


# --------------------------------------------------------------------------
# content-addressed run id isolates grid runs
# --------------------------------------------------------------------------

def _mock_cfg(model="Qwen3-32B", k=0, cls_prompt="synonyms_only", out="/out"):
    from types import SimpleNamespace as NS
    return NS(
        alignmentTask=NS(task_name="cmt-conference", onto_source_filepath="a.owl", onto_target_filepath="b.owl"),
        oracle=NS(model_name=model, temperature=0.0, top_p=1.0, answer_format="true_false",
                  response_mode="structured", enable_thinking=False),
        prompts=NS(cls_usr_prompt_template_name=cls_prompt, prop_usr_prompt_template_name=None,
                   inst_usr_prompt_template_name=None, sibling_strategy=None),
        few_shot=NS(few_shot_k=k, few_shot_negative_strategy="hard", few_shot_seed=42),
        outputs=NS(logmapllm_output_dirpath=out, logmap_initial_alignment_output_dirpath=out,
                   logmap_refined_alignment_output_dirpath=out),
    )


def test_run_id_deterministic_and_distinguishing():
    from logmap_llm.pipeline.paths import compute_run_id
    a1 = compute_run_id(_mock_cfg(model="Qwen3-32B", k=0))
    a2 = compute_run_id(_mock_cfg(model="Qwen3-32B", k=0))
    b = compute_run_id(_mock_cfg(model="DeepSeek-R1-70B", k=0))   # different model
    c = compute_run_id(_mock_cfg(model="Qwen3-32B", k=8))          # different few-shot k
    assert a1 == a2                      # deterministic
    assert len({a1, b, c}) == 3          # model and k each change the id
    assert len(a1) == 16 and all(ch in "0123456789abcdef" for ch in a1)
    # the 'extra' identity (matcher jar / engine / repeat) also changes the id
    assert compute_run_id(_mock_cfg(), extra={"jar": "x"}) != compute_run_id(_mock_cfg(), extra={"jar": "y"})


def test_isolate_run_namespaces_paths_no_collision(tmp_path):
    from logmap_llm.pipeline.paths import PipelinePaths
    p_a = PipelinePaths.from_config(_mock_cfg(model="Qwen3-32B", out=str(tmp_path)), isolate_run=True)
    p_b = PipelinePaths.from_config(_mock_cfg(model="DeepSeek-R1-70B", out=str(tmp_path)), isolate_run=True)
    # distinct configs -> distinct run-id subdirs -> the same logical artifact resolves to different files
    assert p_a.eval_json() != p_b.eval_json()
    assert p_a.logmap_m_ask() != p_b.logmap_m_ask()
    assert p_a.run_id in str(p_a.eval_json()) and p_b.run_id in str(p_b.eval_json())
    # isolate_run=False keeps the legacy (shared) layout -> collision
    p_c = PipelinePaths.from_config(_mock_cfg(model="Qwen3-32B", out=str(tmp_path)), isolate_run=False)
    p_d = PipelinePaths.from_config(_mock_cfg(model="DeepSeek-R1-70B", out=str(tmp_path)), isolate_run=False)
    assert p_c.eval_json() == p_d.eval_json()   # why the parallel scheduler must set isolate_run=True


# --------------------------------------------------------------------------
# few_shot_negative_strategy is validated (no silent downgrade to random)
# --------------------------------------------------------------------------

def test_cr1_valid_strategies_accepted():
    for s in ["hard", "random", "hard-similar", "query-rag", "static-hard", "static-random", "zero-shot"]:
        assert FewShotConfig(few_shot_negative_strategy=s).few_shot_negative_strategy == s


def test_cr1_unknown_strategy_rejected():
    with pytest.raises(Exception):
        FewShotConfig(few_shot_negative_strategy="haard")  # typo must fail loudly


def test_cr1_negative_k_rejected():
    with pytest.raises(Exception):
        FewShotConfig(few_shot_k=-1)


# --------------------------------------------------------------------------
# undefined oracle metrics are null-with-reason, not 0.0
# --------------------------------------------------------------------------

def test_ea3_specificity_null_when_no_negatives():
    # only a positive prediction that is correct -> tp=1, fp=fn=tn=0 -> specificity undefined
    preds = [{"source": "A", "target": "B", "prediction": True}]
    m = compute_oracle_metrics(preds, reference_alignment={("A", "B")})
    assert m["sensitivity"] == 1.0
    assert m["specificity"] is None
    assert m["youdens_j"] is None
    assert "specificity" in m["metric_notes"] and "youdens_j" in m["metric_notes"]


def test_ea3_sensitivity_null_when_no_positives():
    # a correct rejection -> tn=1, tp=fn=fp=0 -> sensitivity undefined (perfect rejector != worthless)
    preds = [{"source": "A", "target": "C", "prediction": False}]
    m = compute_oracle_metrics(preds, reference_alignment={("A", "B")})
    assert m["specificity"] == 1.0
    assert m["sensitivity"] is None
    assert m["youdens_j"] is None
    assert "sensitivity" in m["metric_notes"]


def test_ea3_defined_case_is_numeric():
    preds = [
        {"source": "A", "target": "B", "prediction": True},   # tp
        {"source": "A", "target": "D", "prediction": True},   # fp
        {"source": "A", "target": "C", "prediction": False},  # tn
        {"source": "A", "target": "E", "prediction": False},  # fn
    ]
    m = compute_oracle_metrics(preds, reference_alignment={("A", "B"), ("A", "E")})
    assert m["sensitivity"] == 0.5 and m["specificity"] == 0.5
    assert m["youdens_j"] == pytest.approx(0.0)
    assert m["metric_notes"] == {}


# --------------------------------------------------------------------------
# Conference M1/M2 typing comes from the full alignment, not M_ask
# --------------------------------------------------------------------------

def test_ea1_full_alignment_typing_retains_confident_mappings(tmp_path):
    sysf = tmp_path / "sys.tsv"
    sysf.write_text("http://a#C1\thttp://b#C2\nhttp://a#P1\thttp://b#P2\n")
    ref = tmp_path / "ref.tsv"
    ref.write_text("http://a#C1\thttp://b#C2\nhttp://a#P1\thttp://b#P2\n")
    (tmp_path / "ref_class.tsv").write_text("http://a#C1\thttp://b#C2\n")
    (tmp_path / "ref_property.tsv").write_text("http://a#P1\thttp://b#P2\n")
    # full initial alignment types all four URIs (pipe format: src|tgt|rel|conf|entityType)
    full = tmp_path / "full.txt"
    full.write_text("http://a#C1|http://b#C2|=|1.0|CLS\nhttp://a#P1|http://b#P2|=|1.0|OPROP\n")
    # M_ask-only alignment omits the *confident* class mapping (so C1/C2 are untyped -> 'other')
    mask = tmp_path / "mask.txt"
    mask.write_text("http://a#P1|http://b#P2|=|0.8|OPROP\n")

    r_full = compute_conference_m1_m2_stratified(sysf, ref, initial_alignment_path=full)
    r_mask = compute_conference_m1_m2_stratified(sysf, ref, initial_alignment_path=mask)

    # full-alignment typing: the confident class mapping is bucketed and found -> recall 1.0
    assert r_full["m1_class"]["recall"] == 1.0
    # M_ask-only typing: C1/C2 -> 'other' -> dropped -> recall understated to 0.0
    assert r_mask["m1_class"]["recall"] == 0.0
    assert r_full["m1_class"]["recall"] > r_mask["m1_class"]["recall"]


# --------------------------------------------------------------------------
# predictions CSV round-trip is index-safe
# --------------------------------------------------------------------------

def test_refine_bridge_index_false_roundtrip(tmp_path):
    df = pd.DataFrame({
        "source_entity_uri": ["s1", "s2"], "target_entity_uri": ["t1", "t2"],
        "relation": ["=", "="], "confidence": [0.8, 0.9], "entityType": ["INST", "INST"],
        "Oracle_prediction": [True, False],
    })
    path = tmp_path / "predictions.csv"
    df.to_csv(path, na_rep="nan", index=False)
    back = pd.read_csv(path)
    back = back.loc[:, ~back.columns.str.startswith("Unnamed:")]
    assert list(back.columns)[0] == "source_entity_uri"
    # positional access (used by _kg_refine_in_python) is aligned
    assert (str(back.iloc[0, 0]), str(back.iloc[0, 1])) == ("s1", "t1")


def test_refine_bridge_legacy_indexful_csv_is_recovered(tmp_path):
    # simulate a legacy CSV with the phantom index column
    df = pd.DataFrame({"source_entity_uri": ["s1"], "target_entity_uri": ["t1"],
                       "relation": ["="], "confidence": [0.8], "entityType": ["INST"]})
    path = tmp_path / "legacy.csv"
    df.to_csv(path, na_rep="nan")  # default index=True -> phantom column
    raw = pd.read_csv(path)
    assert any(c.startswith("Unnamed:") for c in raw.columns)  # phantom column present
    fixed = raw.loc[:, ~raw.columns.str.startswith("Unnamed:")]  # the belt-and-braces drop
    assert (str(fixed.iloc[0, 0]), str(fixed.iloc[0, 1])) == ("s1", "t1")


# --------------------------------------------------------------------------
# consultation failure-abort fires on repeated errors
# --------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, answer: bool):
        self.parsed = BinaryOutputFormat(answer=answer)
        self.logprobs = []
        self.usage = TokensUsage(input_tokens=1, output_tokens=1)


class _GoodOracle:
    def consult_oracle(self, prompt, developer_override=None, few_shot_examples=None):
        return _FakeResp(True)


class _BadOracle:
    def consult_oracle(self, prompt, developer_override=None, few_shot_examples=None):
        raise RuntimeError("endpoint down")


def test_oc1_abort_fires_on_total_outage():
    prompts = {f"s{i}|t{i}": f"prompt {i}" for i in range(12)}
    # every consultation fails -> cumulative errors reach the tolerance -> abort -> None
    res = _run_consultations(_BadOracle(), prompts, pair_entity_types={}, developer_prompt_map=None,
                             max_workers=2, failure_tolerance=5, desc="oc1-bad")
    assert res is None, "failure-abort did not fire on a total outage (OC-1 regression)"


def test_oc1_healthy_run_returns_all_results():
    prompts = {f"s{i}|t{i}": f"prompt {i}" for i in range(12)}
    res = _run_consultations(_GoodOracle(), prompts, pair_entity_types={}, developer_prompt_map=None,
                             max_workers=4, failure_tolerance=5, desc="oc1-good")
    assert res is not None and len(res) == 12
    assert all(pred is True for pred, _conf, _usage in res.values())


# --------------------------------------------------------------------------
# per-query few-shot dispatch + immutable per-request assembly
# --------------------------------------------------------------------------

class _RecordingOracle:
    """Stub oracle that records the few-shot examples each consultation received (by prompt)."""
    def __init__(self):
        self.calls = {}
    def consult_oracle(self, prompt, developer_override=None, few_shot_examples=None):
        self.calls[prompt] = few_shot_examples
        return _FakeResp(True)


def test_run_consultations_per_query_dispatch_no_contamination():
    # each M_ask key must receive its own few-shot examples — never another key's
    prompts = {"k1|x": "PROMPT-1", "k2|y": "PROMPT-2", "k3|z": "PROMPT-3"}
    per_query = {"k1|x": [("u1", "a1")], "k2|y": [("u2", "a2")], "k3|z": []}
    oracle = _RecordingOracle()
    res = _run_consultations(oracle, prompts, pair_entity_types={}, developer_prompt_map=None,
                             max_workers=3, failure_tolerance=5, desc="rag-dispatch",
                             per_query_examples=per_query)
    assert res is not None and len(res) == 3
    assert oracle.calls["PROMPT-1"] == [("u1", "a1")]
    assert oracle.calls["PROMPT-2"] == [("u2", "a2")]
    assert oracle.calls["PROMPT-3"] == []          # empty pool -> zero-shot for this key, not another key's examples


def _stub_manager():
    from logmap_llm.oracle.manager import OracleConsultationManager
    m = OracleConsultationManager(
        api_key="EMPTY", model_name="stub", interaction_style="vllm",
        base_url="http://localhost:8000/v1", temperature=0.0, top_p=1.0,
        reasoning_effort="minimal", max_completion_tokens=8, enable_thinking=False)
    m.add_developer_message("DEV")
    m.freeze_messages()
    return m


def test_manager_per_query_assembly_is_local_and_immutable():
    m = _stub_manager()
    kwargs = m._build_base_kwargs("QUERY", few_shot_examples=[("u1", "a1"), ("u2", "a2")])
    roles = [(msg["role"], msg["content"]) for msg in kwargs["messages"]]
    # developer, then the per-query few-shot turns, then the target query — assembled in order
    assert roles == [
        ("system", "DEV"),
        ("user", "u1"), ("assistant", "a1"),
        ("user", "u2"), ("assistant", "a2"),
        ("user", "QUERY"),
    ]
    # the shared frozen prefix was not mutated (concurrency guarantee)
    assert m._frozen_messages == ({"role": "system", "content": "DEV"},)
    # a second call with different examples does not leak the first call's examples
    kwargs2 = m._build_base_kwargs("QUERY2", few_shot_examples=[("z", "b")])
    assert [msg["content"] for msg in kwargs2["messages"]] == ["DEV", "z", "b", "QUERY2"]


def test_manager_no_examples_is_legacy_shape():
    m = _stub_manager()
    kwargs = m._build_base_kwargs("QUERY", few_shot_examples=None)
    assert [msg["content"] for msg in kwargs["messages"]] == ["DEV", "QUERY"]


def test_rag_fewshot_json_dict_roundtrip_to_consult(tmp_path):
    # closes the loop: stage_two writes a per-query dict to few_shot_json; orchestration loads it
    # (dict branch) and consult dispatches per-key examples.
    per_query_written = {"k1|x": [["u1", "a1"]], "k2|y": [["u2", "a2"], ["u3", "a3"]]}
    fp = tmp_path / "few_shot.json"
    fp.write_text(json.dumps(per_query_written))

    loaded = json.loads(fp.read_text())
    assert isinstance(loaded, dict)                         # orchestration takes the dict branch
    per_query = {k: [tuple(p) for p in pairs] for k, pairs in loaded.items()}

    oracle = _RecordingOracle()
    res = _run_consultations(oracle, {"k1|x": "P1", "k2|y": "P2"}, pair_entity_types={},
                             developer_prompt_map=None, max_workers=2, failure_tolerance=5,
                             desc="roundtrip", per_query_examples=per_query)
    assert res is not None
    assert oracle.calls["P1"] == [("u1", "a1")]
    assert oracle.calls["P2"] == [("u2", "a2"), ("u3", "a3")]


# --------------------------------------------------------------------------
# stratified-global evaluation: KG class/property/instance strata
# --------------------------------------------------------------------------

from logmap_llm.evaluation.engines import CustomEvaluationEngine, PartialReferenceEvaluationEngine

# DBkWik-style KG URIs (the '/class/','/property/','/resource/' convention classify_mapping_pair keys on)
_C = "http://dbkwik.webdatacommons.org/a/class/{}"
_P = "http://dbkwik.webdatacommons.org/a/property/{}"
_R = "http://dbkwik.webdatacommons.org/a/resource/{}"
_C2 = "http://dbkwik.webdatacommons.org/b/class/{}"
_P2 = "http://dbkwik.webdatacommons.org/b/property/{}"
_R2 = "http://dbkwik.webdatacommons.org/b/resource/{}"


def _write_tsv(path, pairs):
    path.write_text("".join(f"{s}\t{t}\n" for s, t in pairs))
    return path


def test_ea2_engines_now_support_stratified_global():
    # both engines must report support, otherwise the harness skips the whole stratified block
    assert CustomEvaluationEngine().supports("stratified_global") is True
    assert PartialReferenceEvaluationEngine().supports("stratified_global") is True


def test_ea2_stratified_global_by_uri_convention(tmp_path):
    # system: 1 correct class, 1 wrong property, 1 correct instance
    system = _write_tsv(tmp_path / "sys.tsv", [
        (_C.format("Jedi"), _C2.format("Jedi")),       # class TP
        (_P.format("homeworld"), _P2.format("planet")), # property FP (not in ref)
        (_R.format("Yoda"), _R2.format("Yoda")),        # instance TP
    ])
    # reference: the correct class, the correct property (missed), the correct instance
    reference = _write_tsv(tmp_path / "ref.tsv", [
        (_C.format("Jedi"), _C2.format("Jedi")),
        (_P.format("homeworld"), _P2.format("homeworld")),  # system missed this -> property FN
        (_R.format("Yoda"), _R2.format("Yoda")),
    ])
    strat = CustomEvaluationEngine().compute_stratified_global(system, reference)
    assert set(strat) == {"class", "property", "instance"}
    assert strat["class"]["precision"] == 1.0 and strat["class"]["recall"] == 1.0
    assert strat["instance"]["f1"] == 1.0
    # property stratum: system predicted 1 (wrong), reference had 1 -> P=0, R=0
    assert strat["property"]["true_positives"] == 0
    assert strat["property"]["false_positives"] == 1 and strat["property"]["false_negatives"] == 1
    assert strat["property"]["source"] == "custom_stratified_property"


def test_ea2_explicit_per_type_refs_are_used(tmp_path):
    system = _write_tsv(tmp_path / "sys.tsv", [(_P.format("x"), _P2.format("x"))])
    reference = _write_tsv(tmp_path / "ref.tsv", [(_P.format("x"), _P2.format("x"))])
    # an explicit property ref that disagrees with the full reference -> proves the explicit file wins
    prop_ref = _write_tsv(tmp_path / "reference_property.tsv", [(_P.format("y"), _P2.format("y"))])
    strat = CustomEvaluationEngine().compute_stratified_global(
        system, reference, stratified_refs={"property": prop_ref})
    # scored against the explicit property ref (x vs y) -> no overlap
    assert strat["property"]["true_positives"] == 0 and strat["property"]["false_negatives"] == 1


def test_ea2_partial_reference_stratum_uses_partial_semantics(tmp_path):
    # a system property mapping whose entities are outside the property reference -> 'ignored' (partial GS)
    system = _write_tsv(tmp_path / "sys.tsv", [
        (_P.format("known"), _P2.format("known")),        # in ref -> TP
        (_P.format("offref"), _P2.format("offref")),      # entities not in ref sources/targets -> ignored
    ])
    reference = _write_tsv(tmp_path / "ref.tsv", [(_P.format("known"), _P2.format("known"))])
    strat = PartialReferenceEvaluationEngine().compute_stratified_global(system, reference)
    assert strat["property"]["true_positives"] == 1
    assert strat["property"]["ignored"] == 1          # partial-GS ignored count present
    assert strat["property"]["source"] == "partial_reference_stratified_property"


def test_ea2_empty_stratum_is_omitted_not_zeroed(tmp_path):
    # only class mappings present -> property/instance strata are undefined, must be omitted (not 0.0)
    system = _write_tsv(tmp_path / "sys.tsv", [(_C.format("A"), _C2.format("A"))])
    reference = _write_tsv(tmp_path / "ref.tsv", [(_C.format("A"), _C2.format("A"))])
    strat = CustomEvaluationEngine().compute_stratified_global(system, reference)
    assert set(strat) == {"class"}
    assert "property" not in strat and "instance" not in strat


# --------------------------------------------------------------------------
# DPROP routing and datatype-range rendering (CPU-testable core; the owlready2
# end-to-end DPROP prompt is covered by the RAG integration smoke test)
# --------------------------------------------------------------------------

from logmap_llm.oracle.prompts.routing import mask_row_entity_type, resolve_pair_lane
from logmap_llm.oracle.prompts.formatting import format_domain_range_clause


class _Row:
    """Minimal M_ask-row stand-in (mimics a pandas Series' iloc + len)."""
    def __init__(self, cells):
        self._cells = cells
    def __len__(self):
        return len(self._cells)
    @property
    def iloc(self):
        return self._cells


def test_pt2_mask_row_entity_type_reads_authoritative_column():
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "DPROP"])) == "DPROP"
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "OPROP"])) == "OPROP"
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "CLS"])) == "CLS"
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "INST"])) == "INST"
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "dprop"])) == "DPROP"   # case-insensitive
    assert mask_row_entity_type(_Row(["s", "t", "=", 0.8, "UNKNO"])) is None       # non-tag -> None
    assert mask_row_entity_type(_Row(["s", "t"])) is None                          # column absent -> None


def test_pt2_authoritative_type_beats_derived_type():
    # an OWL2-punned URI can resolve as 'class' on one side; the DPROP tag must still win
    assert resolve_pair_lane("DPROP", "class", "property") == "property"
    assert resolve_pair_lane("OPROP", "class", "class") == "property"
    assert resolve_pair_lane("CLS", "property", "property") == "class"
    assert resolve_pair_lane("INST", "class", "class") == "instance"


def test_pt2_falls_back_to_derived_type_when_untagged():
    assert resolve_pair_lane(None, "property", "property") == "property"
    assert resolve_pair_lane(None, "instance", "instance") == "instance"
    assert resolve_pair_lane(None, "class", "class") == "class"


def test_pt2_data_property_clause_renders_datatype_not_something():
    # a DPROP with range xsd:date must render datatype phrasing, not '... connects "Person" to something'
    clause = format_domain_range_clause("birthDate", {"Person"}, {"date"}, is_data_property=True)
    assert "which has data values of type" in clause and '"date"' in clause
    assert "connects" not in clause          # no object-property phrasing
    assert 'on "Person"' in clause           # domain still surfaced


def test_pt2_data_property_empty_range_is_not_object_phrased():
    # even with an unknown datatype, a DPROP must not be described as an object property
    clause = format_domain_range_clause("comment", {"Thing"}, set(), is_data_property=True)
    assert "connects" not in clause
    assert "has data values of type" in clause


def test_pt2_object_property_rendering_is_unchanged():
    # the OPROP path (is_data_property=False, the default) renders the exact object-property clause
    clause = format_domain_range_clause("authorOf", {"Person"}, {"Paper"})
    assert clause == '"authorOf" which connects "Person" to "Paper"'
