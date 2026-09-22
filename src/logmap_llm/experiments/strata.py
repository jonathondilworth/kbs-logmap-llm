"""Pure classification of sealed global outcomes into additive strata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


Pair = tuple[str, str]


class StrataError(ValueError):
    """Inputs cannot support an exact, reconcilable partition."""


@dataclass(frozen=True)
class OracleCandidate:
    pair: Pair
    prediction: bool | None
    lane: str


_LANES = {
    "conference": ("m1_class", "m2_property", "unknown"),
    "knowledge_graph": ("class", "property", "instance", "unknown"),
}


def lanes(mode: str) -> tuple[str, ...]:
    try:
        return _LANES[mode]
    except KeyError as exc:
        raise StrataError(f"unsupported stratification mode: {mode!r}") from exc


def lane_for_type(mode: str, raw_type: str) -> str:
    """Map LogMap's fifth-column tag to one reporting lane."""
    entity_type = raw_type.strip().upper()
    if mode == "conference":
        if entity_type == "CLS":
            return "m1_class"
        if entity_type in {"OPROP", "DPROP"}:
            return "m2_property"
    elif mode == "knowledge_graph":
        if entity_type == "CLS":
            return "class"
        if entity_type in {"OPROP", "DPROP"}:
            return "property"
        if entity_type == "INST":
            return "instance"
    else:
        lanes(mode)
    return "unknown"


def _nullable_prf(
    tp: int, fp: int, fn: int
) -> tuple[float | None, float | None, float | None, dict[str, str]]:
    notes: dict[str, str] = {}
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if precision is None:
        notes["precision"] = "undefined: no system mappings (tp+fp=0)"
    if recall is None:
        notes["recall"] = "undefined: no reference mappings (tp+fn=0)"
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
        notes["f1"] = "undefined: precision/recall undefined or both zero"
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1, notes


def _alignment_metrics(
    counts: Mapping[str, int], *, partial: bool
) -> dict[str, Any]:
    tp, fp, fn = (
        counts[name]
        for name in ("true_positives", "false_positives", "false_negatives")
    )
    precision, recall, f1, notes = _nullable_prf(tp, fp, fn)
    result: dict[str, Any] = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "system_size": counts["system_size"],
        "reference_size": counts["reference_size"],
        "metric_notes": notes,
        "source": "aggregate_posthoc_partial" if partial else "aggregate_posthoc_complete",
    }
    if partial:
        result["ignored"] = counts["ignored"]
        result["evaluated_size"] = counts["evaluated_size"]
    return result


def _alignment_strata(
    mode: str,
    system_lanes: Mapping[Pair, str],
    reference: set[Pair],
    reference_lanes: Mapping[Pair, str],
    *,
    partial: bool,
) -> dict[str, dict[str, Any]]:
    fields = [
        "true_positives",
        "false_positives",
        "false_negatives",
        "system_size",
        "reference_size",
    ]
    if partial:
        fields.extend(("ignored", "evaluated_size"))
    counts = {lane: {field: 0 for field in fields} for lane in lanes(mode)}
    reference_sources = {source for source, _target in reference}
    reference_targets = {target for _source, target in reference}

    for pair, system_lane in system_lanes.items():
        if pair in reference:
            lane, outcome = reference_lanes[pair], "true_positives"
        elif partial and pair[0] not in reference_sources and pair[1] not in reference_targets:
            lane, outcome = system_lane, "ignored"
        else:
            lane, outcome = system_lane, "false_positives"
        counts[lane][outcome] += 1
        counts[lane]["system_size"] += 1
        if partial and outcome != "ignored":
            counts[lane]["evaluated_size"] += 1

    for pair in reference:
        lane = reference_lanes[pair]
        counts[lane]["reference_size"] += 1
        if pair not in system_lanes:
            counts[lane]["false_negatives"] += 1

    return {
        f"global_{lane}": _alignment_metrics(values, partial=partial)
        for lane, values in counts.items()
    }


def _oracle_metrics(counts: Mapping[str, int]) -> dict[str, Any]:
    tp, fp, tn, fn = (counts[name] for name in ("tp", "fp", "tn", "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall > 0
        else None
    )
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    notes: dict[str, str] = {}
    if precision is None:
        notes["oracle_precision"] = "undefined: no positive oracle predictions (tp+fp=0)"
    if recall is None:
        notes["oracle_recall"] = "undefined: no positive reference candidates (tp+fn=0)"
    if f1 is None:
        notes["oracle_f1"] = "undefined: precision/recall undefined or both zero"
    if sensitivity is None:
        notes["sensitivity"] = "undefined: no positive reference candidates (tp+fn=0)"
    if specificity is None:
        notes["specificity"] = "undefined: no negative reference candidates (tn+fp=0)"
    if sensitivity is None or specificity is None:
        youdens = None
        notes["youdens_j"] = "undefined: sensitivity or specificity undefined"
    else:
        youdens = sensitivity + specificity - 1.0
    return {
        **dict(counts),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youdens_j": youdens,
        "oracle_precision": precision,
        "oracle_recall": recall,
        "oracle_f1": f1,
        "metric_notes": notes,
        "source": "aggregate_posthoc_oracle",
    }


def _oracle_strata(
    mode: str,
    predictions: Sequence[OracleCandidate],
    reference: set[Pair],
    reference_lanes: Mapping[Pair, str],
    *,
    partial: bool,
) -> dict[str, dict[str, Any]]:
    fields = (
        "tp",
        "fp",
        "tn",
        "fn",
        "errors",
        "partial_scope_excluded",
        "oracle_excluded",
        "total_candidates",
    )
    counts = {lane: {field: 0 for field in fields} for lane in lanes(mode)}
    reference_sources = {source for source, _target in reference}
    reference_targets = {target for _source, target in reference}

    for candidate in predictions:
        pair, prediction = candidate.pair, candidate.prediction
        # Oracle strata describe the population presented to each prompt/model
        # lane.  The sealed M_ask tag therefore remains authoritative even if a
        # positive pair's reference sidecar has a different type.
        lane = candidate.lane
        counts[lane]["total_candidates"] += 1
        if prediction is None:
            counts[lane]["errors"] += 1
            counts[lane]["oracle_excluded"] += 1
        elif partial and pair[0] not in reference_sources and pair[1] not in reference_targets:
            counts[lane]["partial_scope_excluded"] += 1
            counts[lane]["oracle_excluded"] += 1
        elif prediction and pair in reference:
            counts[lane]["tp"] += 1
        elif prediction:
            counts[lane]["fp"] += 1
        elif pair in reference:
            counts[lane]["fn"] += 1
        else:
            counts[lane]["tn"] += 1

    return {
        f"oracle_{lane}": _oracle_metrics(values)
        for lane, values in counts.items()
    }


def _count(block: Mapping[str, Any], key: str, label: str) -> int:
    value = block.get(key)
    if type(value) is not int or value < 0:
        raise StrataError(f"sealed {label}.{key} is not a non-negative integer")
    return value


def _reconcile(
    metrics: Mapping[str, Mapping[str, Any]],
    prefix: str,
    global_block: Mapping[str, Any],
    fields: Sequence[str],
) -> None:
    blocks = {name: block for name, block in metrics.items() if name.startswith(prefix)}
    if not blocks:
        raise StrataError(f"no post-hoc blocks produced for {prefix!r}")
    for field in fields:
        observed = sum(_count(block, field, name) for name, block in blocks.items())
        expected = _count(global_block, field, "global")
        if observed != expected:
            raise StrataError(
                f"post-hoc {prefix.rstrip('_')} strata do not reconcile for {field}: "
                f"sum={observed}, sealed_global={expected}"
            )


def derive_strata(
    *,
    mode: str,
    system_lanes: Mapping[Pair, str] | None,
    reference: set[Pair],
    reference_lanes: Mapping[Pair, str],
    global_metrics: Mapping[str, Any] | None,
    partial_reference: bool,
    predictions: Sequence[OracleCandidate] | None = None,
    oracle_metrics: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Classify each sealed outcome once and require exact count conservation."""
    lanes(mode)
    missing_reference_lanes = reference - set(reference_lanes)
    if missing_reference_lanes:
        raise StrataError(
            f"{len(missing_reference_lanes)} reference pair(s) have no stratum"
        )
    valid_lanes = set(lanes(mode))
    if (
        system_lanes is not None
        and set(system_lanes.values()) - valid_lanes
    ) or set(reference_lanes.values()) - valid_lanes:
        raise StrataError("an input contains an unsupported stratum label")

    if (system_lanes is None) != (global_metrics is None):
        raise StrataError(
            "alignment stratification requires system mappings and sealed global metrics together"
        )
    metrics: dict[str, dict[str, Any]] = {}
    if system_lanes is not None and global_metrics is not None:
        metrics.update(
            _alignment_strata(
                mode,
                system_lanes,
                reference,
                reference_lanes,
                partial=partial_reference,
            )
        )
        alignment_fields = [
            "true_positives",
            "false_positives",
            "false_negatives",
            "system_size",
            "reference_size",
        ]
        if partial_reference:
            alignment_fields.extend(("ignored", "evaluated_size"))
        _reconcile(metrics, "global_", global_metrics, alignment_fields)

    if predictions is not None or oracle_metrics is not None:
        if predictions is None or oracle_metrics is None:
            raise StrataError(
                "oracle stratification requires predictions and sealed oracle metrics together"
            )
        if any(candidate.lane not in valid_lanes for candidate in predictions):
            raise StrataError("an oracle candidate contains an unsupported stratum label")
        oracle_blocks = _oracle_strata(
            mode,
            predictions,
            reference,
            reference_lanes,
            partial=partial_reference,
        )
        metrics.update(oracle_blocks)
        _reconcile(
            oracle_blocks,
            "oracle_",
            oracle_metrics,
            (
                "tp",
                "fp",
                "tn",
                "fn",
                "errors",
                "partial_scope_excluded",
                "oracle_excluded",
                "total_candidates",
            ),
        )
    return metrics
