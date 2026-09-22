"""
Automatic model selection (config/schema.ModelSelectionConfig, pipeline/model_selection.py,
orchestration.select_model). Self-supervised: LogMap's anchors are the labels and no reference
alignment is read. CPU-only; the oracle is replaced by scripted verdicts.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from logmap_llm.config.schema import validate_config
from logmap_llm.constants import M_ASK_COLUMNS
from logmap_llm.experiments.plan import PlanError, _validate_native
from logmap_llm.pipeline import orchestration
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.contracts import PromptBuildResult
from logmap_llm.pipeline.model_selection import (
    RANKING_LABEL_COLUMN,
    build_ranking_set,
    load_ranking_artifact,
    rank_candidates,
    score_candidate,
    write_ranking_artifact,
)
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.pipeline.reporting import validate_run_artifacts


# --------------------------------------------------------------------------
# ranking set
# --------------------------------------------------------------------------

MAPPINGS = [
    ["a1", "b1", "=", 1.0, "CLS"],
    ["a2", "b2", "=", 1.0, "CLS"],
    ["a3", "b3", "=", 1.0, "CLS"],
    ["a4", "b4", "<", 1.0, "CLS"],     # subsumption: never an anchor
    ["a5", "b5", "=", 1.0, "UNKNO"],   # unknown type: never an anchor
    ["p1", "q1", "=", 1.0, "OPROP"],
    ["m1", "n1", "=", 0.6, "CLS"],     # escalated to M_ask: never an anchor
]
M_ASK = [["m1", "n1", "=", 0.6, "CLS"]]


def _mappings(rows=MAPPINGS):
    return pd.DataFrame(rows)  # positional columns, as read from LogMap's file


def _m_ask(rows=M_ASK):
    frame = pd.DataFrame(rows, columns=range(5))
    frame.columns = list(M_ASK_COLUMNS)
    return frame


def test_ranking_set_uses_only_equivalence_anchors_outside_m_ask():
    ranking = build_ranking_set(_mappings(), _m_ask(), max_anchors=10, seed=1)
    positives = ranking[ranking[RANKING_LABEL_COLUMN]]
    assert sorted(zip(positives.iloc[:, 0], positives.iloc[:, 1])) == [
        ("a1", "b1"), ("a2", "b2"), ("a3", "b3"), ("p1", "q1")]
    assert list(ranking.columns) == [*M_ASK_COLUMNS, RANKING_LABEL_COLUMN]
    assert not set(ranking.iloc[:, 0]) & {"a4", "a5", "m1"}


def test_negatives_swap_targets_within_kind_and_never_a_proposed_pair():
    ranking = build_ranking_set(_mappings(), _m_ask(), max_anchors=10, seed=1)
    proposed = {frozenset(row[:2]) for row in MAPPINGS + M_ASK}
    anchors = {(r[0], r[1], r[4]) for r in MAPPINGS if r[2] == "=" and r[4] != "UNKNO"} - {("m1", "n1", "CLS")}
    negatives = ranking[~ranking[RANKING_LABEL_COLUMN]]
    assert len(negatives) == 3  # the lone OPROP anchor has no same-kind donor
    for src, tgt, _rel, _conf, kind, _label in negatives.itertuples(index=False):
        assert any(a[0] == src and a[2] == kind for a in anchors)
        assert any(a[1] == tgt and a[2] == kind and a[0] != src for a in anchors)
        assert frozenset({src, tgt}) not in proposed
    # every negative directly follows its anchor
    assert ranking[RANKING_LABEL_COLUMN].tolist() == [True, False, True, False, True, False, True]


def test_ranking_set_is_capped_and_deterministic():
    first = build_ranking_set(_mappings(), _m_ask(), max_anchors=2, seed=7)
    second = build_ranking_set(_mappings(), _m_ask(), max_anchors=2, seed=7)
    assert int(first[RANKING_LABEL_COLUMN].sum()) == 2
    pd.testing.assert_frame_equal(first, second)


def test_ranking_set_is_empty_without_anchors():
    ranking = build_ranking_set(_mappings(M_ASK), _m_ask(), max_anchors=10, seed=1)
    assert ranking.empty and list(ranking.columns) == [*M_ASK_COLUMNS, RANKING_LABEL_COLUMN]


def test_ranking_artifact_round_trip_keeps_only_rendered_questions(tmp_path):
    ranking = build_ranking_set(_mappings(), _m_ask(), max_anchors=10, seed=1)
    keys = (ranking.iloc[:, 0] + "|" + ranking.iloc[:, 1]).tolist()
    prompts = {key: f"prompt for {key}" for key in keys[:3]}  # the rest were unresolvable
    path = tmp_path / "ranking.json"
    kept = write_ranking_artifact(path, ranking, prompts, max_anchors=10, seed=1)
    assert len(kept) == 3
    loaded, loaded_prompts = load_ranking_artifact(path)
    assert loaded_prompts == prompts
    assert loaded.values.tolist() == kept.values.tolist()
    assert json.loads(path.read_text())["max_anchors"] == 10


# --------------------------------------------------------------------------
# scoring and ranking
# --------------------------------------------------------------------------

def _ranking_df(labels):
    rows = [[f"s{i}", f"t{i}", "=", 1.0, "CLS", label] for i, label in enumerate(labels)]
    return pd.DataFrame(rows, columns=[*M_ASK_COLUMNS, RANKING_LABEL_COLUMN])


def _oracle(model_name, base_url="https://openrouter.ai/api/v1"):
    return validate_config(_config_dict(oracle={"model_name": model_name, "base_url": base_url})).oracle


def test_score_candidate_counts_correct_and_unanswered():
    ranking = _ranking_df([True, False, True])
    predictions = ranking.copy()
    predictions["Oracle_prediction"] = [True, True, "error"]
    record = score_candidate(0, _oracle("m"), ranking, predictions)
    assert (record["asked"], record["correct"], record["errors"], record["status"]) == (3, 1, 1, "scored")
    aborted = score_candidate(1, _oracle("m"), ranking, None)
    assert (aborted["correct"], aborted["errors"], aborted["status"]) == (0, 3, "aborted")


def test_rank_candidates_orders_by_correct_then_errors_then_index():
    records = [
        {"index": 0, "correct": 2, "errors": 1},
        {"index": 1, "correct": 3, "errors": 0},
        {"index": 2, "correct": 2, "errors": 0},
        {"index": 3, "correct": 2, "errors": 0},
    ]
    assert [r["index"] for r in rank_candidates(records)] == [1, 2, 3, 0]


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def _config_dict(**sections):
    base = {
        "alignmentTask": {"task_name": "t", "onto_source_filepath": "s.owl",
                          "onto_target_filepath": "t.owl"},
        "oracle": {"model_name": "base/model", "api_key": "ENV:KEY"},
        "outputs": {"logmapllm_output_dirpath": "o", "logmap_initial_alignment_output_dirpath": "i",
                    "logmap_refined_alignment_output_dirpath": "r"},
        "pipeline": {},
    }
    base.update(sections)
    return base


CANDIDATES = [
    {"model_name": "m1"},
    {"model_name": "m2", "base_url": "http://localhost:8000/v1", "api_key": "EMPTY"},
]


def test_selection_is_off_by_default_and_absent_from_dumps():
    cfg = validate_config(_config_dict())
    assert cfg.model_selection is None and cfg.automatic_model_selection is False
    assert "model_selection" not in cfg.model_dump(exclude_none=True)
    off = validate_config(_config_dict(model_selection={"candidates": CANDIDATES}))
    assert off.automatic_model_selection is False


def test_candidates_merge_over_base_oracle():
    cfg = validate_config(_config_dict(model_selection={"automatic": True, "candidates": CANDIDATES}))
    assert cfg.automatic_model_selection is True
    first, second = cfg.candidate_oracle_configs()
    assert (first.model_name, first.api_key, first.base_url) == ("m1", "ENV:KEY", cfg.oracle.base_url)
    assert (second.model_name, second.api_key, second.base_url) == ("m2", "EMPTY", "http://localhost:8000/v1")
    assert first.temperature == cfg.oracle.temperature


@pytest.mark.parametrize("section, message", [
    ({"automatic": True, "candidates": CANDIDATES[:1]}, "at least two"),
    ({"automatic": True, "candidates": [CANDIDATES[0], {"base_url": "x"}]}, "must name a model_name"),
    ({"automatic": True, "candidates": [CANDIDATES[0], {"model_name": "m2", "typo": 1}]},
     "not a valid \\[oracle\\] table"),
    ({"automatic": True, "max_anchors": 0, "candidates": CANDIDATES}, "max_anchors"),
])
def test_selection_rejects_bad_sections(section, message):
    with pytest.raises(ValueError, match=message):
        validate_config(_config_dict(model_selection=section))


def test_selection_requires_consultation():
    with pytest.raises(ValueError, match="consult_oracle='consult'"):
        validate_config(_config_dict(
            model_selection={"automatic": True, "candidates": CANDIDATES},
            pipeline={"consult_oracle": "bypass"},
        ))


def test_batch_planner_rejects_automatic_selection():
    config = _config_dict(model_selection={"automatic": True, "candidates": CANDIDATES})
    config["oracle"]["api_key"] = "EMPTY"
    with pytest.raises(PlanError, match="single-run feature"):
        _validate_native(config, "defaults")


# --------------------------------------------------------------------------
# the phase
# --------------------------------------------------------------------------

def _paths(tmp_path):
    paths = PipelinePaths(tmp_path / "out", tmp_path / "init", tmp_path / "ref", "t", "synonyms_only")
    paths.create_base_dirs()
    return paths


def _phase_fixture(tmp_path, monkeypatch):
    cfg = validate_config(_config_dict(model_selection={
        "automatic": True,
        "candidates": [*CANDIDATES, {"model_name": "m3"}],
    }))
    paths = _paths(tmp_path)
    ranking = _ranking_df([True, False, True])
    prompts = {f"s{i}|t{i}": f"prompt {i}" for i in range(3)}
    write_ranking_artifact(paths.model_ranking_json(), ranking, prompts, max_anchors=10, seed=42)
    monkeypatch.setattr(orchestration, "_developer_prompts", lambda cfg: ("dev", None))

    def fake_consult(oracle_cfg, asked_prompts, candidates_df, bidirectional, *_args, **_kwargs):
        assert asked_prompts == prompts and bidirectional is False
        labels = candidates_df[RANKING_LABEL_COLUMN].tolist()
        verdicts = {"m1": [True, True, True], "m2": labels, "m3": None}[oracle_cfg.model_name]
        if verdicts is None:
            return None  # aborted under the failure tolerance
        frame = candidates_df.copy()
        frame["Oracle_prediction"] = verdicts
        return frame

    monkeypatch.setattr(orchestration, "_consult", fake_consult)
    return PipelineContext(cfg, paths, None), prompts


def test_select_model_applies_the_winner_and_records_the_ranking(tmp_path, monkeypatch):
    ctx, prompts = _phase_fixture(tmp_path, monkeypatch)
    result = orchestration.select_model(ctx, PromptBuildResult(prompts={"m|n": "p"}))
    assert result.performed and result.questions == 3
    assert [r["model_name"] for r in result.ranking] == ["m2", "m1", "m3"]
    assert [r["status"] for r in result.ranking] == ["scored", "scored", "aborted"]
    # the winner (with its own endpoint) now drives consultation and reporting
    assert (ctx.cfg.oracle.model_name, ctx.cfg.oracle.base_url, ctx.cfg.oracle.api_key) == (
        "m2", "http://localhost:8000/v1", "EMPTY")
    written = json.loads(ctx.run_paths.model_selection_json().read_text())
    assert written["status"] == "selected" and written["selected"]["model_name"] == "m2"
    assert written["selected"]["correct"] == 3


def test_select_model_skips_when_m_ask_is_empty(tmp_path, monkeypatch):
    ctx, _ = _phase_fixture(tmp_path, monkeypatch)
    result = orchestration.select_model(ctx, PromptBuildResult(prompts={}))
    assert not result.performed
    assert ctx.cfg.oracle.model_name == "base/model"
    assert json.loads(ctx.run_paths.model_selection_json().read_text())["status"] == "skipped"


def test_select_model_needs_the_ranking_artifact(tmp_path, monkeypatch):
    ctx, _ = _phase_fixture(tmp_path, monkeypatch)
    ctx.run_paths.model_ranking_json().unlink()
    with pytest.raises(FileNotFoundError):
        orchestration.select_model(ctx, PromptBuildResult(prompts={"m|n": "p"}))


def test_run_artifacts_require_the_selection_record(tmp_path):
    cfg = validate_config(_config_dict(
        model_selection={"automatic": True, "candidates": CANDIDATES},
        pipeline={"align_ontologies": "reuse", "build_oracle_prompts": "reuse"},
    ))
    paths = _paths(tmp_path)
    for path in (paths.logmap_mappings(), paths.logmap_m_ask(), paths.prompts_json(),
                 paths.predictions_csv(), paths.annotated_txt(), paths.annotated_tsv(),
                 paths.refined_mappings_tsv()):
        path.write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="model_selection"):
        validate_run_artifacts(cfg, paths)
    paths.model_selection_json().write_text("{}", encoding="utf-8")
    validate_run_artifacts(cfg, paths)
