"""Adapter from validated batch artifacts to the pure strata computation."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.experiments.strata import (
    OracleCandidate,
    Pair,
    StrataError,
    derive_strata,
    lane_for_type,
    lanes,
)


@dataclass(frozen=True)
class ArtifactStrataResult:
    metrics: dict[str, dict[str, Any]]
    provenance: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_provenance(path: Path, *, display_path: str | None = None) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise StrataError(f"stratification input is missing: {source}")
    return {
        "path": display_path if display_path is not None else str(source.resolve()),
        "sha256": _sha256(source),
        "bytes": source.stat().st_size,
    }


def sidecar_paths(mode: str, reference_path: Path) -> dict[str, Path]:
    reference = Path(reference_path)
    if mode == "conference":
        return {
            "m1_class": reference.with_name(f"{reference.stem}_class.tsv"),
            "m2_property": reference.with_name(f"{reference.stem}_property.tsv"),
        }
    if mode == "knowledge_graph":
        return {
            lane: reference.parent / f"reference_{lane}.tsv"
            for lane in ("class", "property", "instance")
        }
    lanes(mode)
    raise AssertionError("unreachable")


def _read_pair_rows(path: Path, delimiter: str) -> list[tuple[Pair, str]]:
    rows: list[tuple[Pair, str]] = []
    first_content = True
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for line_number, fields in enumerate(csv.reader(stream, delimiter=delimiter), 1):
            if not fields or not any(field.strip() for field in fields):
                continue
            first = fields[0].strip().lower()
            if first_content and first in {
                "source", "sourceentity", "srcentity", "source_entity_uri", "entity1"
            }:
                first_content = False
                continue
            first_content = False
            if len(fields) < 2 or not fields[0].strip() or not fields[1].strip():
                raise StrataError(f"malformed mapping row at {path}:{line_number}")
            pair = (fields[0].strip(), fields[1].strip())
            rows.append((pair, fields[4].strip() if len(fields) >= 5 else ""))
    return rows


def _typed_pairs(mode: str, path: Path, delimiter: str) -> tuple[dict[Pair, str], int]:
    rows = _read_pair_rows(path, delimiter)
    observed: dict[Pair, set[str]] = {}
    for pair, raw_type in rows:
        observed.setdefault(pair, set()).add(lane_for_type(mode, raw_type))
    typed = {
        pair: next(iter(pair_lanes)) if len(pair_lanes) == 1 else "unknown"
        for pair, pair_lanes in observed.items()
    }
    return typed, len(rows)


def _reference_partition(
    mode: str, reference_path: Path, sidecars: Mapping[str, Path]
) -> tuple[set[Pair], dict[Pair, str], dict[str, set[Pair]]]:
    reference = {pair for pair, _raw_type in _read_pair_rows(reference_path, "\t")}
    expected = set(lanes(mode)) - {"unknown"}
    if set(sidecars) != expected:
        raise StrataError(
            f"{mode} sidecars must be exactly {sorted(expected)}, got {sorted(sidecars)}"
        )
    partitions: dict[str, set[Pair]] = {}
    owner: dict[Pair, str] = {}
    for lane in lanes(mode):
        if lane == "unknown":
            continue
        path = Path(sidecars[lane])
        if not path.is_file():
            raise StrataError(f"required {lane} reference sidecar is missing: {path}")
        pairs = {pair for pair, _raw_type in _read_pair_rows(path, "\t")}
        outside = pairs - reference
        if outside:
            raise StrataError(
                f"{lane} sidecar contains {len(outside)} pair(s) absent from the full "
                f"reference; first={sorted(outside)[0]!r}"
            )
        overlap = set(owner) & pairs
        if overlap:
            pair = sorted(overlap)[0]
            raise StrataError(
                f"reference sidecars overlap at {pair!r}: {owner[pair]} and {lane}"
            )
        partitions[lane] = pairs
        owner.update({pair: lane for pair in pairs})
    partitions["unknown"] = reference - set(owner)
    owner.update({pair: "unknown" for pair in partitions["unknown"]})
    return reference, owner, partitions


def _prediction_value(value: str) -> bool | None:
    normalized = value.strip().lower()
    if normalized in {"true", "yes"}:
        return True
    if normalized in {"false", "no"}:
        return False
    return None


def _prediction_rows(
    mode: str, predictions_path: Path, candidate_path: Path
) -> tuple[list[OracleCandidate], dict[str, int]]:
    candidate_lanes, candidate_rows = _typed_pairs(mode, candidate_path, "|")
    if candidate_rows != len(candidate_lanes):
        raise StrataError(
            "sealed M_ask contains duplicate URI pairs: "
            f"rows={candidate_rows}, unique={len(candidate_lanes)}"
        )
    predictions: list[OracleCandidate] = []
    seen_predictions: set[Pair] = set()
    with Path(predictions_path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "source_entity_uri", "target_entity_uri", "entityType", "Oracle_prediction"
        }
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise StrataError(
                f"predictions CSV lacks required columns {sorted(required)}: {predictions_path}"
            )
        for line_number, row in enumerate(reader, 2):
            pair = (
                str(row.get("source_entity_uri", "")).strip(),
                str(row.get("target_entity_uri", "")).strip(),
            )
            if not all(pair):
                raise StrataError(f"blank prediction pair at {predictions_path}:{line_number}")
            if pair not in candidate_lanes:
                raise StrataError(
                    f"prediction pair is absent from sealed M_ask at "
                    f"{predictions_path}:{line_number}: {pair!r}"
                )
            if pair in seen_predictions:
                raise StrataError(
                    f"duplicate prediction pair at {predictions_path}:{line_number}: {pair!r}"
                )
            seen_predictions.add(pair)
            prediction_lane = lane_for_type(mode, str(row.get("entityType", "")))
            candidate_lane = candidate_lanes[pair]
            if prediction_lane != candidate_lane:
                raise StrataError(
                    f"prediction/M_ask entityType mismatch for {pair!r}: "
                    f"{prediction_lane} != {candidate_lane}"
                )
            predictions.append(
                OracleCandidate(
                    pair,
                    _prediction_value(str(row.get("Oracle_prediction", ""))),
                    candidate_lane,
                )
            )
    prediction_pairs = {candidate.pair for candidate in predictions}
    if prediction_pairs != set(candidate_lanes):
        raise StrataError(
            "sealed predictions do not cover the unique M_ask candidate set: "
            f"missing={len(set(candidate_lanes) - prediction_pairs)}, "
            f"extra={len(prediction_pairs - set(candidate_lanes))}"
        )
    return predictions, {
        "candidate_rows": candidate_rows,
        "candidate_unique_pairs": len(candidate_lanes),
        "prediction_rows": len(predictions),
    }


def derive_artifact_strata(
    *,
    mode: str,
    system_path: Path | None,
    reference_path: Path,
    reference_sidecars: Mapping[str, Path],
    global_metrics: Mapping[str, Any] | None,
    partial_reference: bool,
    candidate_path: Path | None = None,
    predictions_path: Path | None = None,
    oracle_metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> ArtifactStrataResult:
    """Load immutable files, call the pure classifier, and record provenance."""
    reference, reference_lanes, partitions = _reference_partition(
        mode, Path(reference_path), reference_sidecars
    )
    system_lanes = (
        _typed_pairs(mode, Path(system_path), "\t")[0]
        if system_path is not None
        else None
    )
    predictions: list[OracleCandidate] | None = None
    coverage: dict[str, int] = {}
    if predictions_path is not None or oracle_metrics is not None:
        if predictions_path is None or candidate_path is None or oracle_metrics is None:
            raise StrataError(
                "oracle stratification requires predictions, M_ask, and sealed oracle metrics together"
            )
        predictions, coverage = _prediction_rows(
            mode, Path(predictions_path), Path(candidate_path)
        )
        coverage["candidate_reference_lane_disagreements"] = sum(
            candidate.pair in reference
            and candidate.lane != reference_lanes[candidate.pair]
            for candidate in predictions
        )
    metrics = derive_strata(
        mode=mode,
        system_lanes=system_lanes,
        reference=reference,
        reference_lanes=reference_lanes,
        global_metrics=global_metrics,
        partial_reference=partial_reference,
        predictions=predictions,
        oracle_metrics=oracle_metrics,
    )
    details: dict[str, Any] = {
        "schema": 1,
        "mode": mode,
        "policy": "sealed-global-outcomes-v2",
        "reconciled": True,
        "partial_reference": partial_reference,
        "reference": {
            "all": file_provenance(Path(reference_path)),
            **{
                lane: file_provenance(Path(reference_sidecars[lane]))
                for lane in lanes(mode)
                if lane != "unknown"
            },
            "pairs": len(reference),
            **{f"{lane}_pairs": len(pairs) for lane, pairs in partitions.items()},
        },
        **coverage,
    }
    if provenance:
        details["artifact"] = dict(provenance)
    return ArtifactStrataResult(metrics, details)


def _mode(config: LogMapLLMConfig) -> str | None:
    if (
        config.evaluation.stratified_class_property
        and config.evaluation.stratified_by_entity_type
    ):
        raise StrataError("evaluation stratification modes are mutually exclusive")
    if config.evaluation.stratified_class_property:
        return "conference"
    if config.evaluation.stratified_by_entity_type:
        return "knowledge_graph"
    return None


def stale_metric_prefixes(config: LogMapLLMConfig) -> tuple[str, ...]:
    """Return evaluator-side stratum blocks replaced by exhaustive post-hoc data."""
    mode = _mode(config)
    if mode == "conference":
        lanes = ("m1_class", "m2_property", "unknown")
    elif mode == "knowledge_graph":
        lanes = ("class", "property", "instance", "unknown")
    else:
        return ()
    return tuple(
        f"metric.{kind}_{lane}."
        for lane in lanes
        for kind in ("global", "oracle")
    )


def _declared_artifact(
    batch_dir: Path,
    complete: Mapping[str, Any],
    *,
    filename: str | None = None,
    suffix: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    matches = []
    for artifact in complete.get("artifacts", []):
        name = Path(str(artifact.get("path", ""))).name
        if (filename is not None and name == filename) or (
            suffix is not None and name.endswith(suffix)
        ):
            matches.append(artifact)
    if len(matches) != 1:
        description = filename if filename is not None else f"*{suffix}"
        raise StrataError(
            f"completion must declare exactly one {description} artifact; found {len(matches)}"
        )
    artifact = matches[0]
    value = Path(str(artifact["path"]))
    path = value if value.is_absolute() else batch_dir / value
    return path, {
        "path": str(artifact["path"]),
        "sha256": str(artifact["sha256"]),
        "bytes": int(artifact["bytes"]),
    }


def _input_fingerprint(
    expected: Mapping[str, Any], name: str
) -> Mapping[str, Any] | None:
    records = expected.get("input_fingerprints", {})
    if isinstance(records, Mapping):
        value = records.get(name)
        return value if isinstance(value, Mapping) else None
    if isinstance(records, list):
        for value in records:
            if isinstance(value, Mapping) and value.get("name") == name:
                return value
    return None


def derive_job_strata(
    batch_dir: Path,
    expected: Mapping[str, Any],
    complete: Mapping[str, Any],
    config: LogMapLLMConfig,
    evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read sealed inputs, derive exact strata, and attach audit provenance."""
    mode = _mode(config)
    if mode is None:
        return {}, {}
    task = config.alignmentTask.task_name
    global_metrics = evaluation.get("global")
    if global_metrics is not None and not isinstance(global_metrics, Mapping):
        raise StrataError("sealed global metric block is malformed")
    system_path: Path | None = None
    artifact_provenance: dict[str, Any] = {}
    if isinstance(global_metrics, Mapping):
        system_path, system_record = _declared_artifact(
            batch_dir, complete, filename=f"{task}-logmap_mappings.tsv"
        )
        artifact_provenance["system"] = system_record
    reference_path = Path(config.evaluation.reference_alignment_path).resolve()
    reference_record = _input_fingerprint(
        expected, "evaluation.reference_alignment_path"
    )
    if reference_record is None:
        raise StrataError("manifest lacks the full-reference input fingerprint")
    if reference_record.get("kind") != "file":
        raise StrataError("manifest full-reference fingerprint is not a file")
    planned_reference = Path(str(reference_record.get("path", "")))
    if not planned_reference.is_absolute():
        planned_reference = batch_dir / planned_reference
    if planned_reference.resolve() != reference_path:
        raise StrataError(
            "frozen config and manifest identify different full references: "
            f"{reference_path} != {planned_reference.resolve()}"
        )
    actual_reference = file_provenance(reference_path)
    if (
        str(reference_record.get("sha256")) != actual_reference["sha256"]
        or reference_record.get("bytes") != actual_reference["bytes"]
    ):
        raise StrataError(f"full reference changed after batch generation: {reference_path}")

    oracle = evaluation.get("oracle")
    predictions_path: Path | None = None
    candidate_path: Path | None = None
    if oracle is not None:
        if not isinstance(oracle, Mapping):
            raise StrataError("sealed oracle metric block is malformed")
        candidate_path, candidate_record = _declared_artifact(
            batch_dir,
            complete,
            filename=f"{task}-logmap_mappings_to_ask_oracle_user_llm.txt",
        )
        artifact_provenance["m_ask"] = candidate_record
        predictions_path, predictions_record = _declared_artifact(
            batch_dir,
            complete,
            suffix="mappings_to_ask_with_oracle_predictions.csv",
        )
        artifact_provenance["predictions"] = predictions_record
    if global_metrics is None and oracle is None:
        raise StrataError(
            "post-hoc stratification requires a sealed global or oracle metric block"
        )

    result = derive_artifact_strata(
        mode=mode,
        system_path=system_path,
        reference_path=reference_path,
        reference_sidecars=sidecar_paths(mode, reference_path),
        global_metrics=global_metrics,
        partial_reference=config.evaluation.partial_reference,
        candidate_path=candidate_path,
        predictions_path=predictions_path,
        oracle_metrics=oracle if isinstance(oracle, Mapping) else None,
        provenance=artifact_provenance,
    )
    result.provenance["reference"]["all"]["planned_sha256"] = str(
        reference_record["sha256"]
    )
    return result.metrics, result.provenance
