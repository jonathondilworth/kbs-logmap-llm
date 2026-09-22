"""
Track-faithful evaluation engines (`evaluation/engines/{logmap_oaei,bioml}.py`), the
engine list of the harness, the extended contract and the config schema.

* Golden Anatomy (`tests/data/anatomy/`): the EACL 2025 / OAEI 2025 LogMapLLM alignment
  against the Anatomy reference. LogMap's evaluator gives TP/FP/FN 1276/49/240 and prints
  0.963/0.842/0.898 (EACL Table 3, OM 2025 Table 1); the plain engine 0.9630/0.8417/0.8983;
  MELT 3.3 (`ConfusionMatrixMetric`, recorded on ngpu) 1325 cells, P 0.963019, R 0.841689,
  F1 0.898275. MELT is not a dependency; its numbers are.
* Synthetic Bio-ML pair (`tests/data/bioml/`, hand-worked values in the docstrings below).
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import pytest

from logmap_llm.config.schema import EvaluationConfig
from logmap_llm.evaluation import conventions as C
from logmap_llm.evaluation.contract import EvaluationContractError, validate_evaluation_payload
from logmap_llm.evaluation.engines import (
    BioMLEvaluationEngine,
    CustomEvaluationEngine,
    LogMapOAEIEvaluationEngine,
    build_engine,
)
from logmap_llm.evaluation.harness import evaluate_alignment, select_engine

DATA = Path(__file__).parent / "data"
ANATOMY = DATA / "anatomy"
BIOML = DATA / "bioml"
S = "http://example.org/src#"
T = "http://example.org/tgt#"

MELT_33 = {"cells": 1325, "precision": 0.963019, "recall": 0.841689, "f1": 0.898275}


def _r(block: dict, nd: int = 3) -> tuple:
    return tuple(None if block[k] is None else round(block[k], nd) for k in ("precision", "recall", "f1"))


@pytest.fixture(scope="module")
def anatomy_tsvs(tmp_path_factory):
    """Tab-separated copies of the golden files for the plain engine (it reads TSV only)."""
    root = tmp_path_factory.mktemp("anatomy")
    system = root / "system.tsv"
    system.write_text("".join(f"{s}\t{t}\t{r}\n" for s, t, r in C.load_alignment_cells(ANATOMY / "anatomy-logmap_mappings.txt")))
    reference = root / "reference.tsv"
    reference.write_text("SrcEntity\tTgtEntity\tScore\n" + "".join(
        f"{s}\t{t}\t1.0\n" for s, t, r in C.load_reference_cells(ANATOMY / "reference.rdf") if r == "="
    ))
    return system, reference


# ---------------------------------------------------------------- golden Anatomy

def test_golden_anatomy_logmap_oaei_matches_eacl_table_3():
    engine = LogMapOAEIEvaluationEngine(reference_path=ANATOMY / "reference.rdf")
    block = engine.compute_global(ANATOMY / "anatomy-logmap_mappings.txt", ANATOMY / "reference.rdf")
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (1276, 49, 240)
    assert block["printed_3dp"] == {"precision": 0.963, "recall": 0.842, "f1": 0.898}
    assert block["system_cells"] == 1325 and block["reference_cells"] == 1516
    assert block["reference_flagged"] == 0 and block["system_flagged_discounted"] == 0
    assert block["protocol"] == "logmap_oaei" and block["source"] == "logmap_oaei"


def test_golden_anatomy_plain_engine_agrees_with_melt(anatomy_tsvs):
    system, reference = anatomy_tsvs
    block = CustomEvaluationEngine().compute_global(system, reference)
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (1276, 49, 240)
    assert block["system_size"] == MELT_33["cells"]
    assert _r(block, 4) == (0.9630, 0.8417, 0.8983)
    for key in ("precision", "recall", "f1"):
        assert math.isclose(block[key], MELT_33[key], abs_tol=5e-7), key
    # the logmap_oaei engine on the same TSV pair (relation column '=' only) agrees exactly
    hashed = LogMapOAEIEvaluationEngine().compute_global(system, reference)
    assert (hashed["true_positives"], hashed["false_positives"], hashed["false_negatives"]) == (1276, 49, 240)


def test_logmap_oaei_rejects_train_reference(anatomy_tsvs):
    system, reference = anatomy_tsvs
    with pytest.raises(ValueError):
        LogMapOAEIEvaluationEngine().compute_global(system, reference, train_reference_path=reference)


# ---------------------------------------------------------------- harness with engines

def _write_predictions(path: Path, rows: list[tuple[str, str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_entity_uri", "target_entity_uri", "Oracle_prediction", "Oracle_confidence"])
        writer.writerows([[s, t, p, "1.0"] for s, t, p in rows])


def test_harness_engine_list_adds_blocks_and_keeps_plain_ones(anatomy_tsvs, tmp_path):
    system, reference = anatomy_tsvs
    cells = C.load_alignment_cells(ANATOMY / "anatomy-logmap_mappings.txt")
    ref_pairs = C.cells_to_pairs(C.load_reference_cells(ANATOMY / "reference.rdf"))
    in_ref = [c for c in cells if (c[0], c[1]) in ref_pairs][:4]
    not_in_ref = [c for c in cells if (c[0], c[1]) not in ref_pairs][:2]
    predictions = tmp_path / "predictions.csv"
    _write_predictions(predictions, [
        (in_ref[0][0], in_ref[0][1], "True"),          # TP under both conventions
        (in_ref[1][1], in_ref[1][0], "True"),          # reversed: FP for plain, TP for logmap_oaei
        (in_ref[2][0], in_ref[2][1], "False"),         # FN
        (in_ref[3][1], in_ref[3][0], "False"),         # reversed: TN for plain, FN for logmap_oaei
        (not_in_ref[0][0], not_in_ref[0][1], "True"),  # FP
        (not_in_ref[1][0], not_in_ref[1][1], "error"), # error
    ])
    out = tmp_path / "evaluation_results.json"
    plain = evaluate_alignment(system, reference, oracle_predictions_path=predictions, task_name="mouse-human",
                               metrics=["global", "oracle"], force_custom=True)
    results = evaluate_alignment(
        system, reference, oracle_predictions_path=predictions, task_name="mouse-human",
        metrics=["global", "oracle"], output_json_path=out, force_custom=True,
        engines=["custom", "logmap_oaei"],
        engine_options={"logmap_oaei": {"reference_path": str(ANATOMY / "reference.rdf"), "rounded": True}},
    )
    payload = json.loads(out.read_text())
    validate_evaluation_payload(payload, ["global", "oracle"], task_name="mouse-human")
    # the plain blocks are exactly what the historical path produces
    assert results["global"] == plain["global"]
    assert {k: v for k, v in results["oracle"].items() if k != "false_mappings"} == {
        k: v for k, v in plain["oracle"].items() if k != "false_mappings"}
    assert "engines" not in plain and payload["engines"] == ["custom", "logmap_oaei"]
    assert payload["engine"] == "custom"
    hashed = payload["global_logmap_oaei"]
    assert _r(hashed) == (0.963, 0.842, 0.898) and hashed["rounded_3dp"] is True
    assert (hashed["true_positives"], hashed["false_positives"], hashed["false_negatives"]) == (1276, 49, 240)
    # oracle: orientation-insensitive labels
    assert (payload["oracle"]["tp"], payload["oracle"]["fp"], payload["oracle"]["tn"], payload["oracle"]["fn"]) == (1, 2, 1, 1)
    o = payload["oracle_logmap_oaei"]
    assert (o["tp"], o["fp"], o["tn"], o["fn"], o["errors"]) == (2, 1, 0, 2, 1)
    assert o["orientation_insensitive"] is True and "false_mappings" not in o
    assert o["total_candidates"] == 6


def test_harness_rejects_primary_engine_in_extra_position(anatomy_tsvs):
    system, reference = anatomy_tsvs
    with pytest.raises(ValueError):
        evaluate_alignment(system, reference, metrics=["global"], engines=["custom", "partial_reference"])
    with pytest.raises(ValueError):
        evaluate_alignment(system, reference, metrics=["global"], engines=["logmap_oaei"])


def test_select_engine_primary_overrides_flags():
    assert select_engine(force_custom=False, primary="custom").name() == "custom"
    assert select_engine(partial_reference=False, primary="partial_reference").name() == "partial_reference"
    assert select_engine(partial_reference=True).name() == "partial_reference"
    with pytest.raises(ValueError):
        select_engine(primary="bioml")


def test_build_engine_registry():
    assert build_engine("logmap_oaei").name() == "logmap_oaei"
    assert build_engine("bioml", {"edition": 2026, "setting": "unsupervised"}).edition == 2026
    assert build_engine("custom").name() == "custom"
    with pytest.raises(ValueError):
        build_engine("melt")
    with pytest.raises(ValueError):
        build_engine("bioml", {"edition": 2021})
    with pytest.raises(ValueError):
        build_engine("bioml", {"edition": 2025, "setting": "codabench"})


# ---------------------------------------------------------------- Bio-ML editions on the synthetic pair
#
# system = S1T1, S2T2, S3T3, S5T5, S6T6, S4T1, S3T4 (7); full = S1T1, S2T2, S3T3, S4T4; train = S1T1;
# ignored (use_in_alignment=false) = S5, T6; deprecated = T5; repaired flags S3T3; split: S1T1 train,
# S2T2 valid, S3T3 + S4T4 test.
#   2022 unsupervised: TP 3 (S1T1 S2T2 S3T3) FP 4 FN 1                    -> 3/7, 3/4
#   2022 semi_supervised: S1T1 removed from both sides: TP 2 FP 4 FN 1     -> 1/3, 2/3
#   2025 unsupervised: S5T5, S6T6 dropped: TP 3 FP 2 FN 1                  -> 0.6, 0.75
#   2025 semi_supervised: TP 2 FP 2 FN 1                                   -> 0.5, 2/3
#   2026 standard (S5T5 dropped, T5 deprecated): TP 3 FP 3 FN 1            -> 0.5, 0.75
#   2026 repaired: R+ = S1T1 S2T2 S4T4, S3T3 out of both sides: TP 2 FP 3 FN 1 -> 0.4, 2/3
#   2026 codabench_standard: test slice S3T3 S4T4, S1T1 S2T2 masked: TP 1 FP 3 FN 1 -> 0.25, 0.5
#   2026 codabench_repaired: positives S4T4 only, scored S6T6 S4T1 S3T4: TP 0 FP 3 FN 1 -> 0.0, 0.0, F1 undefined

def _bioml(**options) -> dict:
    options.setdefault("reference_path", BIOML / "full.tsv")
    return BioMLEvaluationEngine(**options).compute_global(BIOML / "system.tsv", BIOML / "full.tsv")


def _counts(block: dict) -> tuple[int, int, int]:
    return block["true_positives"], block["false_positives"], block["false_negatives"]


def test_bioml_2022():
    block = _bioml(edition=2022, setting="unsupervised", train_alignment_path=BIOML / "train.tsv",
                   test_reference_path=BIOML / "test.tsv")
    assert block["headline_view"] == "unsupervised" and block["edition"] == 2022
    assert _counts(block) == (3, 4, 1) and math.isclose(block["precision"], 3 / 7) and block["recall"] == 0.75
    assert block["ignored_classes"] == 0 and block["system_cells_ignored"] == 0
    semi = block["views"]["semi_supervised"]
    assert _counts(semi) == (2, 4, 1) and math.isclose(semi["precision"], 1 / 3) and math.isclose(semi["recall"], 2 / 3)
    assert semi["protocol"] == "null_reference" and semi["null_size"] == 1
    # the ignored list is not consulted before 2023
    same = _bioml(edition=2022, ignored_classes_path=BIOML / "ignored_classes.txt")
    assert _counts(same) == (3, 4, 1)


def test_bioml_2025_official_and_semi_supervised():
    block = _bioml(edition=2025, setting="unsupervised", ignored_classes_path=BIOML / "ignored_classes.txt",
                   train_alignment_path=BIOML / "train.tsv", test_reference_path=BIOML / "test.tsv")
    assert _counts(block) == (3, 2, 1) and block["precision"] == 0.6 and block["recall"] == 0.75
    assert block["ignored_classes"] == 2 and block["system_cells_ignored"] == 2 and block["system_cells"] == 7
    semi = block["views"]["semi_supervised"]
    assert _counts(semi) == (2, 2, 1) and semi["precision"] == 0.5 and math.isclose(semi["recall"], 2 / 3)
    headline_semi = _bioml(edition=2025, setting="semi_supervised", ignored_classes_path=BIOML / "ignored_classes.txt",
                           train_alignment_path=BIOML / "train.tsv")   # test = full - train
    assert headline_semi["headline_view"] == "semi_supervised" and _counts(headline_semi) == (2, 2, 1)
    assert _counts(headline_semi["views"]["unsupervised"]) == (3, 2, 1)
    with pytest.raises(ValueError):
        _bioml(edition=2025, setting="semi_supervised")   # no train split configured


def test_bioml_2026_views():
    block = _bioml(edition=2026, setting="unsupervised", reference_path=BIOML / "reference.rdf",
                   reference_repaired_path=BIOML / "reference_repaired.rdf",
                   deprecated_classes_path=BIOML / "deprecated_classes.txt", split_path=BIOML / "split.tsv")
    assert block["headline_view"] == "repaired" and block["deprecated_classes"] == 1 and block["system_cells_ignored"] == 1
    views = block["views"]
    assert set(views) == {"standard", "repaired", "codabench_standard", "codabench_repaired"}
    assert _counts(views["standard"]) == (3, 3, 1) and views["standard"]["precision"] == 0.5 and views["standard"]["recall"] == 0.75
    assert _counts(views["repaired"]) == (2, 3, 1) and views["repaired"]["precision"] == 0.4
    assert views["repaired"]["reference_flagged"] == 1 and views["repaired"]["predicted_flagged"] == 1
    assert _counts(views["codabench_standard"]) == (1, 3, 1) and views["codabench_standard"]["precision"] == 0.25
    cb = views["codabench_repaired"]
    assert _counts(cb) == (0, 3, 1) and cb["precision"] == 0.0 and cb["recall"] == 0.0 and cb["f1"] is None
    assert "f1" in cb["metric_notes"] and cb["system_masked_removed"] == 2
    codabench = _bioml(edition=2026, setting="codabench", reference_path=BIOML / "reference.rdf",
                       reference_repaired_path=BIOML / "reference_repaired.rdf", split_path=BIOML / "split.tsv")
    assert codabench["headline_view"] == "codabench_repaired" and codabench["deprecated_classes"] == 0
    # without the deprecated list S5T5 stays a (false positive) prediction
    assert _counts(codabench["views"]["standard"]) == (3, 4, 1)
    with pytest.raises(ValueError):
        _bioml(edition=2026, setting="codabench", reference_path=BIOML / "reference.rdf",
               reference_repaired_path=BIOML / "reference_repaired.rdf")   # no split


def test_bioml_oracle_reference_pairs_follow_the_setting():
    engine = BioMLEvaluationEngine(edition=2025, setting="semi_supervised", reference_path=BIOML / "full.tsv",
                                   train_alignment_path=BIOML / "train.tsv")
    assert engine.oracle_reference_pairs(None) == {(S + "S2", T + "T2"), (S + "S3", T + "T3"), (S + "S4", T + "T4")}
    engine = BioMLEvaluationEngine(edition=2026, setting="codabench", reference_path=BIOML / "reference.rdf",
                                   reference_repaired_path=BIOML / "reference_repaired.rdf", split_path=BIOML / "split.tsv")
    assert engine.oracle_reference_pairs(None) == {(S + "S4", T + "T4")}
    metrics = engine.compute_oracle(
        [{"source": S + "S4", "target": T + "T4", "prediction": True, "confidence": 1.0},
         {"source": S + "S3", "target": T + "T3", "prediction": True, "confidence": 1.0}],
        engine.oracle_reference_pairs(None),
    )
    assert (metrics["tp"], metrics["fp"]) == (1, 1) and metrics["source"] == "bioml"


def test_bioml_blocks_pass_the_contract(tmp_path):
    out = tmp_path / "evaluation_results.json"
    evaluate_alignment(
        BIOML / "system.tsv", BIOML / "full.tsv", task_name="synthetic", metrics=["global"], output_json_path=out,
        force_custom=True, engines=["custom", "bioml"],
        engine_options={"bioml": {"edition": 2026, "setting": "codabench", "reference_path": str(BIOML / "reference.rdf"),
                                  "reference_repaired_path": str(BIOML / "reference_repaired.rdf"),
                                  "deprecated_classes_path": str(BIOML / "deprecated_classes.txt"),
                                  "split_path": str(BIOML / "split.tsv")}},
    )
    payload = json.loads(out.read_text())
    validate_evaluation_payload(payload, ["global"], task_name="synthetic")
    assert payload["global_bioml"]["headline_view"] == "codabench_repaired"
    assert payload["global_bioml"]["f1"] is None and "f1" in payload["global_bioml"]["metric_notes"]


# ---------------------------------------------------------------- contract

def _payload() -> dict:
    cells = C.load_alignment_cells(DATA / "conventions" / "system.txt")
    reference = C.load_reference_cells(DATA / "conventions" / "reference.rdf")
    return {
        "schema_version": 1, "task_name": "t", "engine": "custom", "metrics": ["global"],
        "engines": ["custom", "logmap_oaei"],
        "global": C.prf_plain(C.cells_to_pairs(cells), C.cells_to_pairs(reference)),
        "global_logmap_oaei": C.prf_logmap_oaei(cells, reference, rounded=True),
    }


def test_contract_accepts_rounded_engine_block_and_rejects_tampering():
    payload = _payload()
    validate_evaluation_payload(payload, ["global"], task_name="t")
    tampered = copy.deepcopy(payload)
    tampered["global_logmap_oaei"]["precision"] = 0.501
    with pytest.raises(EvaluationContractError):
        validate_evaluation_payload(tampered, ["global"], task_name="t")
    unrounded = copy.deepcopy(payload)
    unrounded["global_logmap_oaei"]["rounded_3dp"] = False
    with pytest.raises(EvaluationContractError):       # 0.556 is not 5/9 exactly
        validate_evaluation_payload(unrounded, ["global"], task_name="t")


def test_contract_requires_protocol_blocks_and_consistent_engine_list():
    payload = _payload()
    missing = copy.deepcopy(payload)
    del missing["global_logmap_oaei"]["protocol"]
    with pytest.raises(EvaluationContractError):
        validate_evaluation_payload(missing, ["global"], task_name="t")
    absent = copy.deepcopy(payload)
    del absent["global_logmap_oaei"]
    with pytest.raises(EvaluationContractError):
        validate_evaluation_payload(absent, ["global"], task_name="t")
    wrong_order = copy.deepcopy(payload)
    wrong_order["engines"] = ["logmap_oaei", "custom"]
    with pytest.raises(EvaluationContractError):
        validate_evaluation_payload(wrong_order, ["global"], task_name="t")
    bad_view = copy.deepcopy(payload)
    bad_view["global_logmap_oaei"]["views"] = {"x": {"precision": 1.0}}
    with pytest.raises(EvaluationContractError):
        validate_evaluation_payload(bad_view, ["global"], task_name="t")


# ---------------------------------------------------------------- config schema

def test_schema_engine_list_rules():
    cfg = EvaluationConfig(evaluate=True, reference_alignment_path="r.tsv",
                           engines=["custom", "logmap_oaei", "bioml"], bioml={"edition": 2025})
    assert cfg.engines == ["custom", "logmap_oaei", "bioml"]
    assert cfg.engine_options() == {"logmap_oaei": {}, "bioml": {"edition": 2025, "setting": "unsupervised"}}
    assert EvaluationConfig().engines is None and EvaluationConfig().engine_options() == {"logmap_oaei": {}, "bioml": {}}
    for bad in (
        {"engines": ["logmap_oaei"]},
        {"engines": ["custom", "custom"]},
        {"engines": ["custom", "melt"]},
        {"engines": ["custom"], "partial_reference": True},
        {"engines": ["partial_reference"]},
        {"engines": ["custom", "bioml"], "bioml": {"edition": 2026, "setting": "codabench", "reference_repaired_path": "x"}},
        {"engines": ["custom", "bioml"], "bioml": {"edition": 2026}},
        {"engines": ["custom", "bioml"], "bioml": {"setting": "semi_supervised"}},
        {"engines": ["custom", "bioml"], "bioml": {"edition": 2025, "setting": "codabench", "split_path": "s"}},
    ):
        with pytest.raises(ValueError):
            EvaluationConfig(evaluate=True, reference_alignment_path="r.tsv", **bad)
    ok = EvaluationConfig(evaluate=True, reference_alignment_path="r.tsv", train_alignment_path="train.tsv",
                          engines=["custom", "bioml"], bioml={"setting": "semi_supervised"})
    assert ok.bioml.setting == "semi_supervised"
    kg = EvaluationConfig(evaluate=True, reference_alignment_path="r.tsv", partial_reference=True,
                          engines=["partial_reference", "logmap_oaei"])
    assert kg.engines[0] == "partial_reference"


def test_schema_default_dump_is_unchanged():
    # unset engine tables are omitted, so frozen configs without them stay byte-identical
    dumped = EvaluationConfig().model_dump(mode="json", exclude_none=True)
    assert "engines" not in dumped and "logmap_oaei" not in dumped and "bioml" not in dumped
