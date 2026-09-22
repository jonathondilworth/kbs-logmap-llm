"""Aggregate one frozen batch without hiding failed or missing jobs."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.experiments.aggregate_strata import (
    derive_job_strata,
    stale_metric_prefixes,
)
from logmap_llm.experiments.plan import load_manifest, load_toml, sha256_file
from logmap_llm.experiments.run import validate_completion, verify_batch_identity


class IncompleteBatchError(RuntimeError):
    """Aggregation completed, but the expected matrix was incomplete."""


_IDENTITY_COLUMNS = [
    "id",
    "task",
    "model",
    "condition",
    "repeat",
    "condition_id",
    "execution_hash",
    "core_sha256",
    "status",
    "attempt",
    "error",
]


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    value = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return value


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, dict):
            flat.update(_flatten(item, name))
        elif isinstance(item, (str, int, float, bool)) or item is None:
            flat[name] = item
    return flat


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(columns), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _latest_attempt(
    batch_dir: Path,
    job_id: str,
    config: LogMapLLMConfig,
) -> tuple[Path | None, dict[str, Any] | None]:
    attempts = batch_dir / "jobs" / job_id / "attempts"
    if not attempts.is_dir():
        return None, None
    directories = sorted(
        (path for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()),
        reverse=True,
    )
    if not directories:
        return None, None
    attempt = directories[0]
    return attempt, validate_completion(batch_dir, attempt / "complete.json", config)


def _attempt_status(attempt: Path | None) -> tuple[str, str]:
    if attempt is None:
        return "not_run", ""
    path = attempt / "status.json"
    if not path.is_file():
        return "incomplete", "attempt has no status or valid completion record"
    try:
        status = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return "incomplete", str(exc)
    state = str(status.get("status", "incomplete"))
    if state in {"finished", "success", "degraded"}:
        return "invalid_artifacts", "completion record or declared artifacts are invalid"
    return state, str(status.get("error", ""))


def _artifact_path(batch_dir: Path, complete: dict[str, Any], filename: str) -> Path | None:
    for artifact in complete.get("artifacts", []):
        value = artifact.get("path", "")
        if Path(value).name == filename:
            path = Path(value)
            return path if path.is_absolute() else batch_dir / path
    return None


def _job_row(
    batch_dir: Path, expected: dict[str, Any], core_sha256: str
) -> dict[str, Any]:
    expected = dict(expected, core_sha256=core_sha256)
    row = {column: expected.get(column, "") for column in _IDENTITY_COLUMNS}
    try:
        relative_config = Path(expected["config_path"])
        if relative_config.is_absolute():
            raise ValueError("config_path must be relative")
        config_path = (batch_dir / relative_config).resolve()
        config_path.relative_to(batch_dir.resolve())
        if not config_path.is_file():
            raise ValueError(f"missing frozen config: {config_path}")
        if sha256_file(config_path) != expected.get("config_sha256"):
            raise ValueError(f"frozen config hash changed: {config_path}")
        config = LogMapLLMConfig.model_validate(load_toml(config_path))
    except (KeyError, OSError, ValueError) as exc:
        row["status"] = "invalid_plan"
        row["error"] = str(exc)
        return row
    attempt, complete = _latest_attempt(batch_dir, expected["id"], config)
    row["attempt"] = attempt.name if attempt else ""
    if complete is None:
        row["status"], row["error"] = _attempt_status(attempt)
        return row
    identity = {
        key: expected.get(key)
        for key in (
            "id",
            "task",
            "model",
            "condition",
            "repeat",
            "condition_id",
            "execution_hash",
            "config_sha256",
            "alignment_id",
            "core_sha256",
        )
    }
    mismatched = [key for key, value in identity.items() if complete.get(key) != value]
    if mismatched:
        row["status"] = "invalid_artifacts"
        row["error"] = "completion identity mismatch: " + ", ".join(mismatched)
        return row
    row["status"] = complete["status"]
    row["error"] = ""
    evaluation = _artifact_path(batch_dir, complete, "evaluation_results.json")
    result = _artifact_path(batch_dir, complete, "run_result.json")
    try:
        if result is not None:
            payload = _read_json(result)
            row.update({f"run.{key}": value for key, value in _flatten(payload).items()})
        if evaluation is not None:
            payload = _read_json(evaluation)
            row.update({f"metric.{key}": value for key, value in _flatten(payload).items()})
            prefixes = stale_metric_prefixes(config)
            if prefixes:
                for key in tuple(row):
                    if key.startswith(prefixes):
                        del row[key]
                metrics, provenance = derive_job_strata(
                    batch_dir, expected, complete, config, payload
                )
                row.update(
                    {
                        f"metric.{key}": value
                        for key, value in _flatten(metrics).items()
                    }
                )
                row.update(
                    {
                        f"strata.{key}": value
                        for key, value in _flatten(provenance).items()
                    }
                )
    except (OSError, ValueError, csv.Error) as exc:
        row["status"] = "invalid_artifacts"
        row["error"] = str(exc)
    return row


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonnegative_count(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _divide(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _global_blocks(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return validated global/global_* metric blocks present in flattened rows."""
    suffix = ".true_positives"
    blocks = {
        key[len("metric.") : -len(suffix)]
        for row in rows
        for key in row
        if key.startswith("metric.global") and key.endswith(suffix)
    }
    return sorted(blocks, key=lambda block: (block != "global", block))


def _oracle_blocks(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Return validated oracle/oracle_* blocks present in flattened rows."""
    suffix = ".tp"
    blocks = {
        key[len("metric.") : -len(suffix)]
        for row in rows
        for key in row
        if key.startswith("metric.oracle") and key.endswith(suffix)
    }
    return sorted(blocks, key=lambda block: (block != "oracle", block))


def _pooled_global(
    rows: Sequence[dict[str, Any]], block: str = "global"
) -> dict[str, Any]:
    """Pool one alignment metric block from raw confusion counts."""
    prefix = f"metric.{block}"
    keys = {
        "true_positives": f"{prefix}.true_positives",
        "false_positives": f"{prefix}.false_positives",
        "false_negatives": f"{prefix}.false_negatives",
    }
    eligible = [
        row
        for row in rows
        if all(_nonnegative_count(row.get(source)) is not None for source in keys.values())
    ]
    if not eligible:
        return {}
    counts = {
        name: sum(_nonnegative_count(row.get(source)) or 0 for row in eligible)
        for name, source in keys.items()
    }
    tp = counts["true_positives"]
    fp = counts["false_positives"]
    fn = counts["false_negatives"]
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    f1 = (
        _ratio(2 * precision * recall, precision + recall)
        if precision is not None and recall is not None
        else None
    )
    output = {
        f"pooled.{block}.jobs": len(eligible),
        f"pooled.{block}.expected_jobs": len(rows),
        f"pooled.{block}.true_positives": tp,
        f"pooled.{block}.false_positives": fp,
        f"pooled.{block}.false_negatives": fn,
        f"pooled.{block}.precision": precision,
        f"pooled.{block}.recall": recall,
        f"pooled.{block}.f1": f1,
    }
    if precision is None:
        output[f"pooled.{block}.metric_notes.precision"] = (
            "undefined: no pooled system mappings (tp+fp=0)"
        )
    if recall is None:
        output[f"pooled.{block}.metric_notes.recall"] = (
            "undefined: no pooled reference mappings (tp+fn=0)"
        )
    if f1 is None:
        output[f"pooled.{block}.metric_notes.f1"] = (
            "undefined: precision/recall undefined or both zero"
        )
    # Partial-reference sizes are useful for auditing KG recombinations. Only
    # pool an optional count when every contributing job declared it.
    for name in ("ignored", "system_size", "evaluated_size", "reference_size"):
        source = f"{prefix}.{name}"
        values = [_nonnegative_count(row.get(source)) for row in eligible]
        if values and all(value is not None for value in values):
            output[f"pooled.{block}.{name}"] = sum(value or 0 for value in values)
    return output


def _pooled_oracle(
    rows: Sequence[dict[str, Any]], block: str = "oracle"
) -> dict[str, Any]:
    """Pool the candidate-classification confusion matrix, where TN is defined."""
    names = ("tp", "fp", "tn", "fn")
    prefix = f"metric.{block}"
    eligible = [
        row
        for row in rows
        if all(_nonnegative_count(row.get(f"{prefix}.{name}")) is not None for name in names)
    ]
    if not eligible:
        return {}
    counts = {
        name: sum(_nonnegative_count(row.get(f"{prefix}.{name}")) or 0 for row in eligible)
        for name in names
    }
    tp, fp, tn, fn = (counts[name] for name in names)
    precision = _ratio(tp, tp + fp)
    sensitivity = _ratio(tp, tp + fn)
    specificity = _ratio(tn, tn + fp)
    f1 = (
        _ratio(2 * precision * sensitivity, precision + sensitivity)
        if precision is not None
        and sensitivity is not None
        and precision + sensitivity > 0
        else None
    )
    accuracy = _ratio(tp + tn, tp + fp + tn + fn)
    output: dict[str, Any] = {
        f"pooled.{block}.jobs": len(eligible),
        f"pooled.{block}.expected_jobs": len(rows),
        **{f"pooled.{block}.{name}": value for name, value in counts.items()},
        f"pooled.{block}.precision": precision,
        f"pooled.{block}.recall": sensitivity,
        f"pooled.{block}.f1": f1,
        f"pooled.{block}.sensitivity": sensitivity,
        f"pooled.{block}.specificity": specificity,
        f"pooled.{block}.youdens_j": (
            sensitivity + specificity - 1.0
            if sensitivity is not None and specificity is not None
            else None
        ),
        f"pooled.{block}.accuracy": accuracy,
    }
    undefined = {
        "precision": (
            precision,
            "undefined: no positive pooled oracle predictions (tp+fp=0)",
        ),
        "recall": (
            sensitivity,
            "undefined: no positive pooled reference candidates (tp+fn=0)",
        ),
        "f1": (f1, "undefined: pooled oracle precision/recall undefined or both zero"),
        "sensitivity": (
            sensitivity,
            "undefined: no positive pooled reference candidates (tp+fn=0)",
        ),
        "specificity": (
            specificity,
            "undefined: no negative pooled reference candidates (tn+fp=0)",
        ),
        "youdens_j": (
            output[f"pooled.{block}.youdens_j"],
            "undefined: pooled sensitivity or specificity undefined",
        ),
        "accuracy": (accuracy, "undefined: no classified pooled oracle candidates"),
    }
    for metric, (value, reason) in undefined.items():
        if value is None:
            output[f"pooled.{block}.metric_notes.{metric}"] = reason
    for name in ("errors", "partial_scope_excluded", "oracle_excluded", "total_candidates"):
        values = [
            _nonnegative_count(row.get(f"{prefix}.{name}")) for row in eligible
        ]
        if values and all(value is not None for value in values):
            output[f"pooled.{block}.{name}"] = sum(value or 0 for value in values)
    return output


def _group_rows(
    rows: Sequence[dict[str, Any]], group_by: Sequence[str], *, allow_partial: bool
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key, "") for key in group_by)].append(row)
    output: list[dict[str, Any]] = []
    for identity, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        statuses = Counter(str(row["status"]) for row in members)
        completed = [row for row in members if row["status"] in {"success", "degraded"}]
        summary: dict[str, Any] = dict(zip(group_by, identity))
        summary.update(
            {
                "expected": len(members),
                "completed": len(completed),
                "coverage": _divide(len(completed), len(members)),
                "complete": len(completed) == len(members),
                "success": statuses["success"],
                "degraded": statuses["degraded"],
                "failed": sum(
                    count
                    for state, count in statuses.items()
                    if state not in {"success", "degraded", "not_run"}
                ),
                "not_run": statuses["not_run"],
            }
        )
        if len(completed) != len(members) and not allow_partial:
            output.append(summary)
            continue
        numeric_metric_keys = sorted(
            {
                key
                for row in completed
                for key, value in row.items()
                if key.startswith("metric.") and _number(value) is not None
            }
        )
        for key in numeric_metric_keys:
            values = [_number(row.get(key)) for row in completed]
            present = [value for value in values if value is not None]
            summary[f"contributors.{key[7:]}"] = len(present)
            if present:
                summary[f"mean.{key[7:]}"] = statistics.fmean(present)
        for block in _global_blocks(completed):
            summary.update(_pooled_global(completed, block))
        for block in _oracle_blocks(completed):
            summary.update(_pooled_oracle(completed, block))
        output.append(summary)
    return output


def _columns(rows: Sequence[dict[str, Any]], leading: Sequence[str]) -> list[str]:
    extras = sorted({key for row in rows for key in row} - set(leading))
    return [*leading, *extras]


def _markdown(
    batch_name: str,
    rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    group_by: Sequence[str],
) -> str:
    statuses = Counter(str(row["status"]) for row in rows)
    complete = statuses["success"] + statuses["degraded"]
    lines = [
        f"# Results: {batch_name}",
        "",
        f"Coverage: **{complete}/{len(rows)} ({_divide(100 * complete, len(rows)):.1f}%)**",
        "",
        "| Status | Jobs |",
        "|---|---:|",
    ]
    for status, count in sorted(statuses.items()):
        lines.append(f"| {status} | {count} |")
    lines.extend(["", "## Groups", ""])
    headers = [*group_by, "completed", "expected", "coverage"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in summaries:
        values = [str(row.get(key, "")) for key in group_by]
        values.extend(
            [
                str(row["completed"]),
                str(row["expected"]),
                f"{100 * row['coverage']:.1f}%",
            ]
        )
        lines.append("| " + " | ".join(values) + " |")
    preferred_metrics = [
        "pooled.global.precision",
        "pooled.global.recall",
        "pooled.global.f1",
        "mean.global.precision",
        "mean.global.recall",
        "mean.global.f1",
        "mean.oracle.oracle_precision",
        "mean.oracle.oracle_recall",
        "mean.oracle.oracle_f1",
        "mean.oracle.sensitivity",
        "mean.oracle.specificity",
        "mean.oracle.youdens_j",
    ]
    metric_columns = [
        key for key in preferred_metrics if any(key in row for row in summaries)
    ]
    if metric_columns:
        lines.extend(["", "## Primary metrics", ""])
        metric_headers = [*group_by, *metric_columns]
        lines.append("| " + " | ".join(metric_headers) + " |")
        lines.append("|" + "|".join("---" for _ in metric_headers) + "|")
        for row in summaries:
            values = [str(row.get(key, "")) for key in group_by]
            for key in metric_columns:
                value = row.get(key)
                values.append(
                    f"{value:.4f}"
                    if isinstance(value, float)
                    else ("" if value is None else str(value))
                )
            lines.append("| " + " | ".join(values) + " |")
        lines.extend(["", "All metrics and contributor counts are in `summary.csv`."])

    labels = {
        "global": "Global",
        "global_m1_class": "M1/class",
        "global_m2_property": "M2/property",
        "global_class": "CLS",
        "global_property": "PROP",
        "global_instance": "INST",
        "global_unknown": "UNKNOWN",
    }
    blocks = sorted(
        {
            key.removeprefix("pooled.").removesuffix(".true_positives")
            for row in summaries
            for key in row
            if key.startswith("pooled.global") and key.endswith(".true_positives")
        },
        key=lambda block: (block != "global", block),
    )
    if blocks:
        lines.extend(["", "## Pooled alignment strata", ""])
        headers = [
            *group_by,
            "stratum",
            "jobs",
            "expected",
            "TP",
            "FP",
            "FN",
            "precision",
            "recall",
            "f1",
        ]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join("---" for _ in headers) + "|")
        for row in summaries:
            for block in blocks:
                prefix = f"pooled.{block}"
                if f"{prefix}.true_positives" not in row:
                    continue
                values = [str(row.get(key, "")) for key in group_by]
                values.extend(
                    [
                        labels.get(block, block),
                        str(row.get(f"{prefix}.jobs", "")),
                        str(row.get(f"{prefix}.expected_jobs", "")),
                        str(row.get(f"{prefix}.true_positives", "")),
                        str(row.get(f"{prefix}.false_positives", "")),
                        str(row.get(f"{prefix}.false_negatives", "")),
                    ]
                )
                for metric in ("precision", "recall", "f1"):
                    value = row.get(f"{prefix}.{metric}")
                    values.append(f"{value:.4f}" if isinstance(value, float) else "")
                lines.append("| " + " | ".join(values) + " |")
        lines.extend(
            [
                "",
                "TP/FP/FN are additive. Alignment TN is not defined without an explicit candidate universe.",
            ]
        )
    incomplete = [row for row in rows if row["status"] not in {"success", "degraded"}]
    if incomplete:
        lines.extend(["", "## Incomplete jobs", "", "| Job | Status | Detail |", "|---|---|---|"])
        for row in incomplete:
            detail = str(row.get("error", "")).replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {row['id']} | {row['status']} | {detail} |")
    return "\n".join(lines) + "\n"


def aggregate_batch(
    batch_dir: str | os.PathLike[str], *, allow_incomplete: bool = False
) -> dict[str, Any]:
    """Write jobs.csv, summary.csv, and summary.md for exactly one manifest."""
    root = Path(batch_dir).resolve()
    manifest = load_manifest(root)
    verify_batch_identity(root, manifest, compare_current_core=False)
    core_sha256 = manifest["source"]["core_sha256"]
    rows = [_job_row(root, expected, core_sha256) for expected in manifest["jobs"]]
    aggregate = manifest.get("aggregate", {})
    group_by = aggregate.get("group_by", ["task", "model", "condition"])
    if not isinstance(group_by, list) or any(
        key not in {"task", "model", "condition", "repeat"} for key in group_by
    ):
        raise ValueError("manifest aggregate.group_by contains an unsupported field")
    require_complete = bool(aggregate.get("require_complete", True))
    summaries = _group_rows(
        rows,
        group_by,
        allow_partial=allow_incomplete or not require_complete,
    )
    output = root / "aggregate"
    _write_csv(output / "jobs.csv", rows, _columns(rows, _IDENTITY_COLUMNS))
    summary_leading = [*group_by, "expected", "completed", "coverage", "success", "degraded", "failed", "not_run"]
    _write_csv(output / "summary.csv", summaries, _columns(summaries, summary_leading))
    _atomic_text(
        output / "summary.md",
        _markdown(manifest.get("batch", {}).get("name", root.name), rows, summaries, group_by),
    )
    incomplete = sum(row["status"] not in {"success", "degraded"} for row in rows)
    result = {
        "expected": len(rows),
        "completed": len(rows) - incomplete,
        "incomplete": incomplete,
        "output_dir": str(output),
    }
    if incomplete and require_complete and not allow_incomplete:
        raise IncompleteBatchError(
            f"wrote coverage-aware results to {output}, but {incomplete}/{len(rows)} jobs are incomplete; "
            "use --allow-incomplete to accept partial coverage"
        )
    return result
