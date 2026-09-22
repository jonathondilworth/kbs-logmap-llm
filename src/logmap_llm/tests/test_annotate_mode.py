"""
Annotate mode (WP5, LOCAL_CHANGES.md §7): `pipeline.align_ontologies = "external"`,
`pipeline.stop_after_consultation`, and the annotated M_ask files written after every
consultation (`pipeline/annotate.py`).

The golden check regenerates the two composed-mapping annotation files delivered on
17 Sep 2026 (`tests/data/annotate/`) from their M_ask and the recorded oracle predictions:
the `.txt` byte-identical, the `.tsv` identical apart from the added `LLM_confidence` column.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd
import pytest

from logmap_llm.config.schema import validate_config
from logmap_llm.pipeline import annotate
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.orchestration import align
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.utils.data import normalise_prediction_column

DATA = Path(__file__).parent / "data" / "annotate"


def _config(tmp_path: Path, **overrides) -> tuple:
    root = tmp_path / "run"
    pipeline = {"align_ontologies": "external", "build_oracle_prompts": "bypass", "consult_oracle": "reuse",
                "refine_alignment": "refine", "stop_after_consultation": True}
    pipeline.update(overrides.pop("pipeline", {}))
    task = {"task_name": "mouse-human", "onto_source_filepath": "s.owl", "onto_target_filepath": "t.owl",
            "external_mappings_filepath": str(DATA / "default" / "m_ask.txt")}
    task.update(overrides.pop("alignmentTask", {}))
    document = {
        "alignmentTask": task,
        "oracle": {"model_name": "recorded", "interaction_style": "vllm", "response_mode": "plain",
                   "base_url": "http://127.0.0.1:1/v1"},
        "prompts": {"cls_usr_prompt_template_name": "one_level_of_parents_and_synonyms"},
        "outputs": {"logmapllm_output_dirpath": str(root / "out"),
                    "logmap_initial_alignment_output_dirpath": str(root / "init"),
                    "logmap_refined_alignment_output_dirpath": str(root / "ref")},
        "pipeline": pipeline,
    }
    document.update(overrides)
    cfg = validate_config(document)
    paths = PipelinePaths.from_config(cfg)
    paths.create_base_dirs()
    return cfg, paths


# ---------------------------------------------------------------- golden regeneration

@pytest.mark.parametrize("case,template,bidirectional", [
    ("default", "one_level_of_parents_and_synonyms", False),
    ("mutualsub", "sub_parents_synonyms", True),
])
def test_annotated_files_regenerate_the_delivered_ones(tmp_path, case, template, bidirectional):
    fixture = DATA / case
    paths = PipelinePaths(tmp_path / "out", tmp_path / "init", tmp_path / "ref", "mouse-human", template)
    paths.create_base_dirs()
    shutil.copyfile(fixture / "m_ask.txt", paths.logmap_m_ask())
    predictions = normalise_prediction_column(pd.read_csv(fixture / "predictions.csv"))
    txt, tsv = annotate.write_annotated_files(paths, predictions, bidirectional)
    assert txt.name == f"mouse-human-{template}-annotated.txt" and tsv.name == f"mouse-human-{template}-annotated.tsv"
    assert txt.read_bytes() == (fixture / "expected.annotated.txt").read_bytes()
    produced = tsv.read_text().splitlines()
    expected = (fixture / "expected.annotated.tsv").read_text().splitlines()
    assert len(produced) == len(expected)
    assert produced[0] == expected[0] + "\tLLM_confidence"
    for got, want in zip(produced[1:], expected[1:]):
        columns = got.split("\t")
        assert "\t".join(columns[:-1]) == want
        assert columns[-1] == "nan" or 0.0 <= float(columns[-1]) <= 1.0
    # the confidence column is the run's Oracle_confidence, row for row
    confidences = [line.split("\t")[-1] for line in produced[1:]]
    recorded = [annotate._render_confidence(v) for v in predictions["Oracle_confidence"]]
    assert confidences == recorded


def test_verdict_rendering_and_missing_candidates(tmp_path):
    paths = PipelinePaths(tmp_path / "out", tmp_path / "init", tmp_path / "ref", "t", "tpl")
    paths.create_base_dirs()
    paths.logmap_m_ask().write_text(
        "http://a#1|http://b#1|=|0.9|CLS\n"
        "http://a#2|http://b#2|=|0.8|CLS\n"
        "http://a#3|http://b#3|<|0.7|OPROP\n"
        "http://a#4|http://b#4|=|0.6|CLS\n"
        "http://a#1|http://b#1|=|0.9|CLS\n"      # duplicate: collapsed
        "http://a#5|http://b#5\n"                # short row: padded
    )
    predictions = pd.DataFrame({
        "source_entity_uri": ["http://a#1", "http://a#2", "http://a#3"],
        "target_entity_uri": ["http://b#1", "http://b#2", "http://b#3"],
        "relation": ["=", "=", "<"], "confidence": [0.9, 0.8, 0.7], "entityType": ["CLS", "CLS", "OPROP"],
        "Oracle_prediction": [True, "error", False],
        "Oracle_confidence": [0.75, float("nan"), 0.5],
        "Oracle_fwd_prediction": [True, "error", "n/a"],
        "Oracle_rev_prediction": [True, "skipped", "n/a"],
    })
    txt, tsv = annotate.write_annotated_files(paths, predictions, bidirectional=True)
    assert txt.read_text().splitlines() == [
        "http://a#1|http://b#1|=|0.9|CLS|True|True|True",
        "http://a#2|http://b#2|=|0.8|CLS|ERROR|ERROR|SKIPPED",
        "http://a#3|http://b#3|<|0.7|OPROP|False|n/a|n/a",
        "http://a#4|http://b#4|=|0.6|CLS|SKIPPED|SKIPPED|SKIPPED",
        "http://a#5|http://b#5|=|1.0|CLS|SKIPPED|SKIPPED|SKIPPED",
    ]
    lines = tsv.read_text().splitlines()
    assert lines[0] == "source\ttarget\trelation\tconfidence\ttype\tLLM_annotation\tLLM_forward_subsumption\tLLM_reverse_subsumption\tLLM_confidence"
    assert lines[1].split("\t")[-1] == "0.75" and lines[2].split("\t")[-1] == "nan" and lines[4].split("\t")[-1] == "nan"


# ---------------------------------------------------------------- external mappings

def test_load_external_mappings_formats(tmp_path):
    pipe = DATA / "default" / "m_ask.txt"
    rows = annotate.load_external_mappings(pipe)
    assert len(rows) == 423 and rows[0] == ("http://mouse.owl#MA_0000322", "http://human.owl#NCI_C32461", "=", "0.62", "CLS")
    deeponto = tmp_path / "m.tsv"
    deeponto.write_text("SrcEntity\tTgtEntity\tScore\nhttp://a#1\thttp://b#1\t0.5\nhttp://a#2\thttp://b#2\t<\t0.4\tOPROP\n")
    assert annotate.load_external_mappings(deeponto) == [
        ("http://a#1", "http://b#1", "=", "0.5", "CLS"), ("http://a#2", "http://b#2", "<", "0.4", "OPROP"),
    ]
    rdf = tmp_path / "m.rdf"
    rdf.write_text(
        '<?xml version="1.0"?><rdf:RDF xmlns="http://knowledgeweb.semanticweb.org/heterogeneity/alignment" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"><Alignment><map><Cell>'
        '<entity1 rdf:resource="http://a#1"/><entity2 rdf:resource="http://b#1"/><measure>0.83</measure>'
        '<relation>&gt;</relation></Cell></map><map><Cell><entity1 rdf:resource="http://a#2"/>'
        '<entity2 rdf:resource="http://b#2"/></Cell></map></Alignment></rdf:RDF>'
    )
    assert annotate.load_external_mappings(rdf) == [
        ("http://a#1", "http://b#1", ">", "0.83", "CLS"), ("http://a#2", "http://b#2", "=", "1.0", "CLS"),
    ]
    with pytest.raises(ValueError):
        annotate.load_external_mappings(_write(tmp_path / "bad.txt", "only-one-field\n"))


def _write(path: Path, text: str) -> Path:
    path.write_text(text)
    return path


def test_align_external_publishes_m_ask_files(tmp_path):
    cfg, paths = _config(tmp_path)
    result = align(PipelineContext(cfg, paths, logmap=None))
    assert result.n_m_ask == 423 and result.n_mappings == 423
    assert list(result.m_ask_df.columns) == ["source_entity_uri", "target_entity_uri", "relation", "confidence", "entityType"]
    assert result.m_ask_df["confidence"].dtype.kind == "f"
    assert paths.logmap_m_ask().read_bytes() == (DATA / "default" / "m_ask.txt").read_bytes()
    assert paths.logmap_mappings().read_bytes() == paths.logmap_m_ask().read_bytes()
    assert paths.logmap_mappings_tsv().read_text().splitlines()[0] == "http://mouse.owl#MA_0000322\thttp://human.owl#NCI_C32461\t=\t0.62\tCLS"


def test_align_external_collapses_duplicates_and_rejects_missing_file(tmp_path):
    duplicated = _write(tmp_path / "dup.txt", "http://a#1|http://b#1|=|0.9|CLS\nhttp://a#1|http://b#1|=|0.9|CLS\nhttp://a#2|http://b#2\n")
    cfg, paths = _config(tmp_path, alignmentTask={"external_mappings_filepath": str(duplicated)})
    result = align(PipelineContext(cfg, paths, logmap=None))
    assert result.n_m_ask == 2 and paths.logmap_m_ask().read_text().count("\n") == 2
    cfg, paths = _config(tmp_path / "second", alignmentTask={"external_mappings_filepath": str(tmp_path / "absent.txt")})
    with pytest.raises(FileNotFoundError):
        align(PipelineContext(cfg, paths, logmap=None))


# ---------------------------------------------------------------- schema and harness rules

def test_schema_rules_for_external_and_stop(tmp_path):
    cfg, _ = _config(tmp_path)
    assert cfg.pipeline.align_ontologies.value == "external" and cfg.pipeline.stop_after_consultation
    with pytest.raises(ValueError):       # external needs the file
        _config(tmp_path / "a", alignmentTask={"external_mappings_filepath": None})
    with pytest.raises(ValueError):       # the file is only read in external mode
        _config(tmp_path / "b", pipeline={"align_ontologies": "reuse"})
    with pytest.raises(ValueError):       # nothing to stop after
        _config(tmp_path / "c", pipeline={"consult_oracle": "bypass", "build_oracle_prompts": "bypass"})
    with pytest.raises(ValueError):       # a stopped run cannot claim an evaluation
        _config(tmp_path / "d", evaluation={"evaluate": True, "reference_alignment_path": "r.tsv"})
    with pytest.raises(ValueError):       # reuse prompts still needs reuse alignment
        _config(tmp_path / "e", pipeline={"build_oracle_prompts": "reuse"})
    cfg, _ = _config(tmp_path / "f", pipeline={"stop_after_consultation": False, "build_oracle_prompts": "build",
                                               "consult_oracle": "consult"})
    assert not cfg.pipeline.stop_after_consultation


def test_stopped_run_artifacts_and_result(tmp_path):
    from logmap_llm.pipeline.reporting import validate_run_artifacts, write_results_file
    from logmap_llm.pipeline.contracts import EvaluationResult, OracleResult, PromptBuildResult, TimingRecord
    from logmap_llm.experiments.run import _required_job_artifacts
    import json

    cfg, paths = _config(tmp_path)
    align(PipelineContext(cfg, paths, logmap=None))
    predictions = normalise_prediction_column(pd.read_csv(DATA / "default" / "predictions.csv"))
    predictions.to_csv(paths.predictions_csv(), index=False)
    with pytest.raises(FileNotFoundError):     # the annotated files are owed
        validate_run_artifacts(cfg, paths, stopped_after="consultation")
    annotate.write_annotated_files(paths, predictions, False)
    validate_run_artifacts(cfg, paths, stopped_after="consultation")   # no refined alignment, no evaluation
    write_results_file(paths.run_result(), cfg, TimingRecord(), EvaluationResult(),
                       OracleResult(predictions=predictions), PromptBuildResult(prompts={"k": "p"}), paths,
                       stopped_after="consultation")
    payload = json.loads(paths.run_result().read_text())
    assert payload["stopped_after"] == "consultation" and payload["status"] == "succeeded"
    assert {a["role"] for a in payload["artifacts"]} >= {"annotated_txt", "annotated_tsv", "predictions", "m_ask"}
    assert payload["counts"] == {"prompts": 1, "candidates": 423, "true": 214, "false": 209, "error": 0, "skipped": 0}
    required = {p.name for p in _required_job_artifacts(cfg, tmp_path / "root")}
    assert "mouse-human-logmap_mappings.tsv" not in required and "evaluation_results.json" not in required


def test_planner_treats_external_mode_without_alignment_prerequisite():
    from logmap_llm.experiments.plan import _alignment_id, _nested_get, _nested_set

    assert _alignment_id("t", {"pipeline": {"align_ontologies": "external"}, "alignmentTask": {}}, {}, "sha") is None
    document = {"evaluation": {"bioml": {"split_path": "s.tsv"}, "reference_alignment_path": "r.tsv"}}
    assert _nested_get(document, ("evaluation", "bioml", "split_path")) == "s.tsv"
    assert _nested_get(document, ("evaluation", "logmap_oaei", "reference_path")) is None
    _nested_set(document, ("evaluation", "bioml", "split_path"), "/abs/s.tsv")
    assert document["evaluation"]["bioml"]["split_path"] == "/abs/s.tsv"
    _nested_set(document, ("evaluation", "bioml", "missing"), "x")
    assert "missing" not in document["evaluation"]["bioml"]
