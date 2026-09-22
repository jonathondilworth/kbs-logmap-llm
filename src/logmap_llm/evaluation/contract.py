"""Validation for the versioned evaluation artifact consumed by experiments."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from logmap_llm.constants import EXTRA_EVALUATION_ENGINES, PRIMARY_EVALUATION_ENGINES
from logmap_llm.evaluation.conventions import printed_measures


class EvaluationContractError(ValueError):
    """An evaluation artifact is incomplete or scientifically inconsistent."""


def _table(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EvaluationContractError(f"{label} must be a JSON object")
    return value


def _count(table: Mapping[str, Any], key: str, label: str) -> int:
    value = table.get(key)
    if type(value) is not int or value < 0:
        raise EvaluationContractError(f"{label}.{key} must be a non-negative integer")
    return value


def _number(
    table: Mapping[str, Any],
    key: str,
    label: str,
    *,
    minimum: float,
    maximum: float,
    nullable: bool = False,
) -> float | None:
    value = table.get(key)
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvaluationContractError(f"{label}.{key} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise EvaluationContractError(
            f"{label}.{key} must be between {minimum} and {maximum}"
        )
    return result


def _expected_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _same(actual: float, expected: float, label: str) -> None:
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise EvaluationContractError(
            f"{label} is inconsistent with the declared confusion counts"
        )


def _metric_notes(
    table: Mapping[str, Any], label: str, *, required: bool = True
) -> Mapping[str, Any]:
    if "metric_notes" not in table and not required:
        return {}
    notes = _table(table.get("metric_notes"), f"{label}.metric_notes")
    if any(
        not isinstance(key, str) or not isinstance(value, str) or not value.strip()
        for key, value in notes.items()
    ):
        raise EvaluationContractError(f"{label}.metric_notes must contain non-empty reasons")
    return notes


def _validate_global(
    value: Any,
    *,
    label: str = "evaluation.global",
    expected_source: str | None = None,
) -> None:
    table = _table(value, label)
    tp = _count(table, "true_positives", label)
    fp = _count(table, "false_positives", label)
    fn = _count(table, "false_negatives", label)
    system_size = _count(table, "system_size", label)
    reference_size = _count(table, "reference_size", label)
    source = table.get("source")
    if not isinstance(source, str) or not source:
        raise EvaluationContractError(f"{label}.source must be a non-empty string")
    if expected_source is not None and source != expected_source:
        raise EvaluationContractError(
            f"{label}.source must be {expected_source!r} for the selected engine"
        )
    if reference_size != tp + fn:
        raise EvaluationContractError(f"{label}.reference_size is inconsistent")
    partial_fields = {"ignored", "evaluated_size"} & set(table)
    if partial_fields:
        if partial_fields != {"ignored", "evaluated_size"}:
            raise EvaluationContractError(
                f"{label}.ignored and evaluated_size must be declared together"
            )
        ignored = _count(table, "ignored", label)
        evaluated = _count(table, "evaluated_size", label)
        if evaluated != tp + fp or system_size != evaluated + ignored:
            raise EvaluationContractError(f"{label} partial-reference sizes are inconsistent")
    elif system_size != tp + fp:
        raise EvaluationContractError(f"{label}.system_size is inconsistent")
    precision = _number(
        table, "precision", label, minimum=0.0, maximum=1.0, nullable=True
    )
    recall = _number(table, "recall", label, minimum=0.0, maximum=1.0, nullable=True)
    f1 = _number(table, "f1", label, minimum=0.0, maximum=1.0, nullable=True)
    # metric_notes may be omitted when every metric is defined, but any null
    # metric below still requires an explicit per-metric reason.
    notes = _metric_notes(table, label, required=False)

    # A logmap_oaei block with rounded_3dp reports the values LogMap prints (P and R
    # rounded to three decimals with Java Math.round, F derived from the rounded values);
    # it reconciles against those instead of the exact ratios.
    rounded = table.get("rounded_3dp", False)
    if not isinstance(rounded, bool):
        raise EvaluationContractError(f"{label}.rounded_3dp must be a boolean")
    printed = printed_measures(tp, fp, fn) if rounded else None

    if tp + fp:
        if precision is None:
            raise EvaluationContractError(f"{label}.precision is unexpectedly null")
        _same(
            precision,
            printed["precision"] if printed else _expected_ratio(tp, tp + fp),
            f"{label}.precision",
        )
    elif precision is not None:
        raise EvaluationContractError(f"{label}.precision must be null when undefined")

    if tp + fn:
        if recall is None:
            raise EvaluationContractError(f"{label}.recall is unexpectedly null")
        _same(
            recall,
            printed["recall"] if printed else _expected_ratio(tp, tp + fn),
            f"{label}.recall",
        )
    elif recall is not None:
        raise EvaluationContractError(f"{label}.recall must be null when undefined")

    if precision is None or recall is None or precision + recall == 0:
        if f1 is not None:
            raise EvaluationContractError(f"{label}.f1 must be null when undefined")
    else:
        if f1 is None:
            raise EvaluationContractError(f"{label}.f1 is unexpectedly null")
        _same(
            f1,
            printed["f1"] if printed else 2 * precision * recall / (precision + recall),
            f"{label}.f1",
        )

    for key, value in (("precision", precision), ("recall", recall), ("f1", f1)):
        if value is None and key not in notes:
            raise EvaluationContractError(f"{label}.{key} is null without a metric note")


def _validate_engine_global(value: Any, *, label: str) -> None:
    """A `global_<engine>` block of a track-faithful engine: the usual reconciliation, a
    declared `protocol`, and every entry of an optional `views` table reconciled the
    same way (the Bio-ML engine reports its other settings there)."""
    _validate_global(value, label=label)
    table = _table(value, label)
    protocol = table.get("protocol")
    if not isinstance(protocol, str) or not protocol:
        raise EvaluationContractError(f"{label}.protocol must be a non-empty string")
    views = table.get("views")
    if views is not None:
        views = _table(views, f"{label}.views")
        for view_name, view in views.items():
            view_label = f"{label}.views.{view_name}"
            _validate_global(view, label=view_label)
            if not isinstance(_table(view, view_label).get("protocol"), str):
                raise EvaluationContractError(f"{view_label}.protocol must be a string")
        headline = table.get("headline_view")
        if headline is not None and headline not in views:
            raise EvaluationContractError(f"{label}.headline_view is not one of its views")


def _validate_oracle(value: Any, label: str = "evaluation.oracle") -> None:
    table = _table(value, label)
    if "false_mappings" in table:
        raise EvaluationContractError(
            f"{label}.false_mappings must be kept in its separate diagnostic artifact"
        )
    counts = {
        key: _count(table, key, label)
        for key in (
            "tp", "fp", "tn", "fn", "errors", "partial_scope_excluded",
            "oracle_excluded", "total_candidates",
        )
    }
    if counts["oracle_excluded"] != counts["errors"] + counts["partial_scope_excluded"]:
        raise EvaluationContractError(f"{label}.oracle_excluded is inconsistent")
    classified = sum(counts[key] for key in ("tp", "fp", "tn", "fn"))
    if counts["total_candidates"] != classified + counts["oracle_excluded"]:
        raise EvaluationContractError(f"{label}.total_candidates is inconsistent")

    precision = _number(table, "oracle_precision", label, minimum=0.0, maximum=1.0)
    recall = _number(table, "oracle_recall", label, minimum=0.0, maximum=1.0)
    f1 = _number(table, "oracle_f1", label, minimum=0.0, maximum=1.0)
    assert precision is not None and recall is not None and f1 is not None
    _same(
        precision,
        _expected_ratio(counts["tp"], counts["tp"] + counts["fp"]),
        f"{label}.oracle_precision",
    )
    _same(
        recall,
        _expected_ratio(counts["tp"], counts["tp"] + counts["fn"]),
        f"{label}.oracle_recall",
    )
    _same(f1, _expected_ratio(2 * precision * recall, precision + recall), f"{label}.oracle_f1")

    sensitivity = _number(
        table, "sensitivity", label, minimum=0.0, maximum=1.0, nullable=True
    )
    specificity = _number(
        table, "specificity", label, minimum=0.0, maximum=1.0, nullable=True
    )
    youdens = _number(table, "youdens_j", label, minimum=-1.0, maximum=1.0, nullable=True)
    if counts["tp"] + counts["fn"]:
        if sensitivity is None:
            raise EvaluationContractError(f"{label}.sensitivity is unexpectedly null")
        _same(
            sensitivity,
            _expected_ratio(counts["tp"], counts["tp"] + counts["fn"]),
            f"{label}.sensitivity",
        )
    elif sensitivity is not None:
        raise EvaluationContractError(f"{label}.sensitivity must be null when undefined")
    if counts["tn"] + counts["fp"]:
        if specificity is None:
            raise EvaluationContractError(f"{label}.specificity is unexpectedly null")
        _same(
            specificity,
            _expected_ratio(counts["tn"], counts["tn"] + counts["fp"]),
            f"{label}.specificity",
        )
    elif specificity is not None:
        raise EvaluationContractError(f"{label}.specificity must be null when undefined")
    if sensitivity is None or specificity is None:
        if youdens is not None:
            raise EvaluationContractError(f"{label}.youdens_j must be null when undefined")
    else:
        if youdens is None:
            raise EvaluationContractError(f"{label}.youdens_j is unexpectedly null")
        _same(youdens, sensitivity + specificity - 1.0, f"{label}.youdens_j")
    notes = _metric_notes(table, label)
    for key, value in (
        ("sensitivity", sensitivity),
        ("specificity", specificity),
        ("youdens_j", youdens),
    ):
        if value is None and key not in notes:
            raise EvaluationContractError(f"{label}.{key} is null without a metric note")


def validate_evaluation_payload(
    payload: Any,
    metrics: Sequence[str],
    *,
    task_name: str | None = None,
) -> None:
    """Validate requested metric blocks, primitive types, ranges, and raw counts."""
    result = _table(payload, "evaluation")
    if result.get("schema_version") != 1:
        raise EvaluationContractError("evaluation.schema_version must be 1")
    if task_name is not None and result.get("task_name") != task_name:
        raise EvaluationContractError("evaluation.task_name does not match the frozen config")
    engine = result.get("engine")
    source_for_engine = {
        "custom": "custom",
        "partial_reference": "kg_partial",
        "deeponto": "deeponto",
    }
    if engine not in source_for_engine:
        raise EvaluationContractError(
            "evaluation.engine must be custom, partial_reference, or deeponto"
        )
    # Optional engine list (present when [evaluation] engines was set): the primary engine
    # first, then the track-faithful engines whose blocks must all be present.
    extra_engines: list[str] = []
    if "engines" in result:
        engines = result["engines"]
        if (
            not isinstance(engines, list)
            or not engines
            or any(not isinstance(name, str) for name in engines)
            or engines[0] != engine
            or any(name in PRIMARY_EVALUATION_ENGINES for name in engines[1:])
            or any(name not in EXTRA_EVALUATION_ENGINES for name in engines[1:])
            or len(set(engines)) != len(engines)
        ):
            raise EvaluationContractError(
                "evaluation.engines must start with the primary engine followed by distinct "
                f"track-faithful engines {list(EXTRA_EVALUATION_ENGINES)}"
            )
        extra_engines = list(engines[1:])
    declared = result.get("metrics")
    if not isinstance(declared, list) or declared != list(metrics):
        raise EvaluationContractError("evaluation.metrics does not match the frozen config")
    recognised = {key for key in ("global", "oracle") if key in result}
    if recognised != set(metrics):
        raise EvaluationContractError(
            "evaluation metric blocks do not match the frozen config"
        )
    for metric in metrics:
        if metric == "global":
            _validate_global(
                result.get("global"), expected_source=source_for_engine[str(engine)]
            )
        elif metric == "oracle":
            _validate_oracle(result.get("oracle"))
        else:  # The config schema should make this unreachable.
            raise EvaluationContractError(f"unsupported evaluation metric: {metric}")
    for name in extra_engines:
        if "global" in metrics and f"global_{name}" not in result:
            raise EvaluationContractError(f"evaluation.global_{name} is missing for engine {name!r}")
        if "oracle" in metrics and f"oracle_{name}" not in result:
            raise EvaluationContractError(f"evaluation.oracle_{name} is missing for engine {name!r}")
    for key, value in result.items():
        if key.startswith("global_"):
            if key[len("global_"):] in EXTRA_EVALUATION_ENGINES:
                _validate_engine_global(value, label=f"evaluation.{key}")
            else:
                _validate_global(value, label=f"evaluation.{key}")
        elif key.startswith("oracle_"):
            _validate_oracle(value, label=f"evaluation.{key}")


__all__ = ["EvaluationContractError", "validate_evaluation_payload"]
