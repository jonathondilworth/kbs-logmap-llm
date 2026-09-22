"""Deterministic planning for small, local LogMapLLM experiment batches.

The planner has one job: turn one readable TOML file into an immutable batch
of ordinary, standalone LogMapLLM configs.  It deliberately knows nothing
about process scheduling or result aggregation.
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import os
import re
import shutil
import tempfile
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from logmap_llm.config.schema import LogMapLLMConfig, validate_config
from logmap_llm.experiments.environment import environment_payload, environments_differ


SCHEMA_VERSION = 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_ENV_RE = re.compile(r"^ENV:[A-Za-z_][A-Za-z0-9_]*$")

_TOP_KEYS = {
    "schema", "batch", "defaults", "tasks", "models", "conditions",
    "matrix", "aggregate",
}
_BATCH_KEYS = {
    "name", "output_root", "jobs", "timeout_seconds", "reuse_alignments",
}
_MATRIX_KEYS = {"tasks", "models", "conditions", "repeats", "exclude"}
_AGGREGATE_KEYS = {"group_by", "require_complete"}
_MODEL_KEYS = {"config", "max_parallel_runs", "verify_endpoint"}
_AXIS_KEYS = {"config"}
_SELECTOR_KEYS = {"id", "task", "model", "condition", "repeat"}

# Native config paths that are inputs or persistent caches.  Generated output
# paths are handled separately and may never be supplied by the author.
_RESOLVED_PATHS = (
    ("alignmentTask", "onto_source_filepath"),
    ("alignmentTask", "onto_target_filepath"),
    ("alignmentTask", "logmap_parameters_dirpath"),
    ("alignmentTask", "external_mappings_filepath"),
    ("oracle", "local_oracle_predictions_dirpath"),
    ("evaluation", "reference_alignment_path"),
    ("evaluation", "train_alignment_path"),
    ("evaluation", "test_cands_path"),
    ("evaluation", "logmap_oaei", "reference_path"),
    ("evaluation", "bioml", "reference_path"),
    ("evaluation", "bioml", "reference_repaired_path"),
    ("evaluation", "bioml", "test_reference_path"),
    ("evaluation", "bioml", "train_alignment_path"),
    ("evaluation", "bioml", "ignored_classes_path"),
    ("evaluation", "bioml", "deprecated_classes_path"),
    ("evaluation", "bioml", "split_path"),
    ("few_shot", "rag_cache_dir"),
    ("few_shot", "prebuilt_few_shot_bundle_path"),
)
_FINGERPRINTED_PATHS = set(_RESOLVED_PATHS) - {("few_shot", "rag_cache_dir")}


class PlanError(ValueError):
    """A concise, user-correctable error in a batch specification."""


def load_toml(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read a TOML document without applying any schema-specific behavior."""
    source = Path(path)
    with source.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise PlanError(f"TOML root must be a table: {source}")
    return value


def _sorted_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sorted_data(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_sorted_data(item) for item in value]
    return value


def _toml_text(data: Mapping[str, Any]) -> str:
    try:
        import tomli_w
    except ImportError as exc:  # pragma: no cover - packaging guarantees it
        raise RuntimeError("generating configs requires the 'tomli-w' package") from exc
    return tomli_w.dumps(_sorted_data(data))


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""
    _atomic_write(Path(path), text)


def atomic_write_json(path: str | os.PathLike[str], value: Any) -> None:
    """Atomically write stable, human-readable JSON."""
    text = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    _atomic_write(Path(path), text)


def write_toml(path: str | os.PathLike[str], value: Mapping[str, Any]) -> None:
    """Atomically write deterministic TOML."""
    _atomic_write(Path(path), _toml_text(value))


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Return a streamed SHA-256 digest for one regular file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    """Hash JSON-compatible data independently of mapping insertion order."""
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_MANIFEST_PAYLOAD_KEYS = {
    "schema", "batch", "source", "alignments", "jobs", "aggregate",
}
#: Keys a manifest may carry. `environment` records the resolved Python dependency set the
#: batch was generated in (see experiments/environment.py). The seal digest is built from
#: the keys actually present, so a manifest without `environment` digests unchanged; a
#: manifest must never acquire `"environment": None`, which would change its digest.
_MANIFEST_OPTIONAL_KEYS = {"environment"}
_MANIFEST_SEAL = {
    "schema": 1,
    "algorithm": "sha256",
    "canonicalization": "json-sort-v1",
}


def _valid_manifest_keys(keys: set[str]) -> bool:
    """Required keys all present, and nothing beyond the optional extras."""
    return _MANIFEST_PAYLOAD_KEYS <= keys <= _MANIFEST_PAYLOAD_KEYS | _MANIFEST_OPTIONAL_KEYS


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seal_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic integrity-sealed execution manifest."""
    if not _valid_manifest_keys(set(payload)):
        raise PlanError("cannot seal a malformed manifest payload")
    sealed = copy.deepcopy(dict(payload))
    sealed["seal"] = {**_MANIFEST_SEAL, "digest": _manifest_digest(payload)}
    return sealed


def verify_manifest_seal(manifest: Mapping[str, Any]) -> None:
    """Fail closed when any declared execution metadata changed after generation."""
    # A presence test, not a key-set identity: an unsealed manifest that also carries
    # `environment` must still get this actionable message rather than "malformed".
    if "seal" not in manifest:
        raise PlanError(
            "unsealed legacy batch manifest; regenerate the batch from batch.toml"
        )
    if not _valid_manifest_keys(set(manifest) - {"seal"}):
        raise PlanError("malformed batch manifest")
    seal = manifest.get("seal")
    if not isinstance(seal, dict) or set(seal) != {*_MANIFEST_SEAL, "digest"}:
        raise PlanError("malformed batch manifest seal")
    if any(seal.get(key) != value for key, value in _MANIFEST_SEAL.items()):
        raise PlanError("unsupported batch manifest seal")
    digest = seal.get("digest")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise PlanError("malformed batch manifest seal digest")
    # Digest the keys actually present, not a fixed set: a fixed set would exclude
    # `environment` from the digest, and a materialised `None` would change old digests.
    payload = {key: value for key, value in manifest.items() if key != "seal"}
    if not hmac.compare_digest(_manifest_digest(payload), digest):
        raise PlanError(
            "manifest seal mismatch; regenerate the batch from batch.toml"
        )


def manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a manifest that must match across hosts: everything but the environment.

    `environment` is a function of the local virtualenv (and `seal` covers it), so any
    cross-host check must compare this rather than raw bytes.
    """
    return {
        key: value for key, value in manifest.items()
        if key not in _MANIFEST_OPTIONAL_KEYS and key != "seal"
    }


def manifests_equivalent(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """True when two manifests agree on everything except the environment they were made in."""
    return manifest_identity(left) == manifest_identity(right)


def load_manifest(batch_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Load and validate the safety-critical shape of a batch manifest."""
    path = Path(batch_dir) / "manifest.json"
    try:
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    except FileNotFoundError as exc:
        raise PlanError(f"batch manifest not found: {path}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA_VERSION:
        raise PlanError(f"unsupported or malformed batch manifest: {path}")
    verify_manifest_seal(manifest)
    if (
        not _valid_manifest_keys(set(manifest) - {"seal"})
        or "seal" not in manifest
        or not isinstance(manifest.get("jobs"), list)
        or not isinstance(manifest.get("alignments"), list)
    ):
        raise PlanError(f"malformed batch manifest: {path}")
    _validate_manifest_rows(manifest, path)
    return manifest


def _safe_internal_path(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value:
        raise PlanError(f"{label} must be a non-empty relative path")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise PlanError(f"{label} must stay inside the batch directory")


def _digest_string(value: Any, label: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise PlanError(f"{label} must be a SHA-256 digest")


def _validate_fingerprint_table(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be a table")
    for field, record in value.items():
        if not isinstance(field, str) or not isinstance(record, dict):
            raise PlanError(f"{label} contains a malformed record")
        kind = record.get("kind")
        allowed = {"path", "kind", "sha256", "bytes"}
        if kind == "directory":
            allowed.add("files")
        if set(record) != allowed or kind not in {"file", "directory"}:
            raise PlanError(f"{label}.{field} has an unsupported shape")
        if not isinstance(record.get("path"), str) or not record["path"]:
            raise PlanError(f"{label}.{field}.path must be non-empty")
        _digest_string(record.get("sha256"), f"{label}.{field}.sha256")
        size = record.get("bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise PlanError(f"{label}.{field}.bytes must be a non-negative integer")
        if kind == "directory":
            files = record.get("files")
            if isinstance(files, bool) or not isinstance(files, int) or files < 0:
                raise PlanError(f"{label}.{field}.files must be a non-negative integer")


def _validate_manifest_rows(manifest: Mapping[str, Any], path: Path) -> None:
    batch = manifest.get("batch")
    source = manifest.get("source")
    aggregate = manifest.get("aggregate")
    if not isinstance(batch, dict) or not isinstance(source, dict) or not isinstance(aggregate, dict):
        raise PlanError(f"malformed batch manifest metadata: {path}")
    _unknown_keys(
        batch,
        {"name", "id", "plan_hash", "jobs", "expected_jobs", "timeout_seconds", "reuse_alignments"},
        "manifest.batch",
    )
    _identifier(batch.get("name"), "manifest.batch.name")
    _identifier(batch.get("id"), "manifest.batch.id")
    _digest_string(batch.get("plan_hash"), "manifest.batch.plan_hash")
    _positive_int(batch.get("jobs"), "manifest.batch.jobs")
    expected_jobs = _positive_int(batch.get("expected_jobs"), "manifest.batch.expected_jobs")
    _positive_int(batch.get("timeout_seconds"), "manifest.batch.timeout_seconds")
    if not isinstance(batch.get("reuse_alignments"), bool):
        raise PlanError("manifest.batch.reuse_alignments must be true or false")
    if set(source) != {"spec_sha256", "core_sha256"}:
        raise PlanError("manifest.source has an unsupported shape")
    _digest_string(source.get("spec_sha256"), "manifest.source.spec_sha256")
    _digest_string(source.get("core_sha256"), "manifest.source.core_sha256")
    if set(aggregate) != {"group_by", "require_complete"}:
        raise PlanError("manifest.aggregate has an unsupported shape")
    group_by = aggregate.get("group_by")
    if (
        not isinstance(group_by, list) or not group_by
        or any(value not in {"task", "model", "condition", "repeat"} for value in group_by)
        or len(set(group_by)) != len(group_by)
        or not isinstance(aggregate.get("require_complete"), bool)
    ):
        raise PlanError("manifest.aggregate is invalid")

    job_keys = {
        "id", "task", "model", "condition", "repeat", "condition_id", "execution_hash",
        "alignment_id", "config_path", "config_sha256", "timeout_seconds",
        "max_parallel_runs", "verify_endpoint", "expected_served_model", "input_fingerprints",
    }
    alignment_keys = {
        "id", "task", "config_path", "config_sha256", "timeout_seconds",
        "input_fingerprints", "job_ids",
    }
    job_ids: set[str] = set()
    for index, row in enumerate(manifest["jobs"]):
        if not isinstance(row, dict) or set(row) != job_keys:
            raise PlanError(f"manifest.jobs[{index}] has an unsupported shape")
        job_id = _identifier(row.get("id"), f"manifest.jobs[{index}].id")
        if job_id in job_ids:
            raise PlanError(f"duplicate job ID in manifest: {job_id}")
        job_ids.add(job_id)
        for axis in ("task", "model", "condition"):
            _identifier(row.get(axis), f"manifest.jobs[{index}].{axis}")
        repeat = row.get("repeat")
        if isinstance(repeat, bool) or not isinstance(repeat, int) or repeat < 0:
            raise PlanError(f"manifest.jobs[{index}].repeat must be a non-negative integer")
        _digest_string(row.get("condition_id"), f"manifest.jobs[{index}].condition_id")
        _digest_string(row.get("execution_hash"), f"manifest.jobs[{index}].execution_hash")
        _digest_string(row.get("config_sha256"), f"manifest.jobs[{index}].config_sha256")
        _safe_internal_path(row.get("config_path"), f"manifest.jobs[{index}].config_path")
        _positive_int(row.get("timeout_seconds"), f"manifest.jobs[{index}].timeout_seconds")
        maximum = row.get("max_parallel_runs")
        if maximum is not None:
            _positive_int(maximum, f"manifest.jobs[{index}].max_parallel_runs")
        if not isinstance(row.get("verify_endpoint"), bool):
            raise PlanError(f"manifest.jobs[{index}].verify_endpoint must be true or false")
        if not isinstance(row.get("expected_served_model"), str) or not row["expected_served_model"]:
            raise PlanError(f"manifest.jobs[{index}].expected_served_model must be non-empty")
        _validate_fingerprint_table(
            row.get("input_fingerprints"), f"manifest.jobs[{index}].input_fingerprints",
        )

    if len(job_ids) != expected_jobs:
        raise PlanError("manifest.batch.expected_jobs does not match the jobs list")

    alignment_ids: set[str] = set()
    for index, row in enumerate(manifest["alignments"]):
        if not isinstance(row, dict) or set(row) != alignment_keys:
            raise PlanError(f"manifest.alignments[{index}] has an unsupported shape")
        alignment_id = _identifier(row.get("id"), f"manifest.alignments[{index}].id")
        if alignment_id in alignment_ids:
            raise PlanError(f"duplicate alignment ID in manifest: {alignment_id}")
        alignment_ids.add(alignment_id)
        _identifier(row.get("task"), f"manifest.alignments[{index}].task")
        _safe_internal_path(row.get("config_path"), f"manifest.alignments[{index}].config_path")
        _digest_string(row.get("config_sha256"), f"manifest.alignments[{index}].config_sha256")
        _positive_int(row.get("timeout_seconds"), f"manifest.alignments[{index}].timeout_seconds")
        _validate_fingerprint_table(
            row.get("input_fingerprints"), f"manifest.alignments[{index}].input_fingerprints",
        )
        members = row.get("job_ids")
        if (
            not isinstance(members, list) or not members
            or any(member not in job_ids for member in members)
            or len(set(members)) != len(members)
        ):
            raise PlanError(f"manifest.alignments[{index}].job_ids is invalid")

    for index, row in enumerate(manifest["jobs"]):
        alignment_id = row.get("alignment_id")
        if alignment_id is not None and alignment_id not in alignment_ids:
            raise PlanError(f"manifest.jobs[{index}] references an unknown alignment")
        if alignment_id is not None:
            alignment = next(item for item in manifest["alignments"] if item["id"] == alignment_id)
            if row["id"] not in alignment["job_ids"]:
                raise PlanError(f"manifest.jobs[{index}] is absent from its alignment membership")


def select_jobs(
    jobs: Sequence[Mapping[str, Any]],
    selectors: Sequence[str] = (),
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Apply simple ``key=a,b`` equality selectors in manifest order."""
    if limit is not None and (isinstance(limit, bool) or limit < 1):
        raise PlanError("--limit must be a positive integer")

    parsed: dict[str, set[str]] = {}
    for selector in selectors:
        key, separator, raw_values = selector.partition("=")
        key = key.strip()
        values = {value.strip() for value in raw_values.split(",") if value.strip()}
        if not separator or key not in _SELECTOR_KEYS or not values:
            allowed = ", ".join(sorted(_SELECTOR_KEYS))
            raise PlanError(f"invalid selector {selector!r}; expected KEY=a,b ({allowed})")
        if key in parsed:
            raise PlanError(f"selector key repeated: {key!r}; use one comma-separated selector")
        parsed[key] = values

    selected = [
        dict(job)
        for job in jobs
        if all(str(job.get(key)) in values for key, values in parsed.items())
    ]
    return selected[:limit] if limit is not None else selected


def _unknown_keys(table: Mapping[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise PlanError(f"unknown key(s) in {label}: {', '.join(unknown)}")


def _table(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PlanError(f"{label} must be a TOML table")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PlanError(f"{label} must be a positive integer")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise PlanError(f"{label} must match [A-Za-z0-9][A-Za-z0-9_-]*")
    return value


def _axis(table: Any, label: str, *, model: bool = False) -> dict[str, dict[str, Any]]:
    entries = _table(table, label)
    if not entries:
        raise PlanError(f"{label} must define at least one entry")
    result: dict[str, dict[str, Any]] = {}
    allowed = _MODEL_KEYS if model else _AXIS_KEYS
    for axis_id, raw_entry in entries.items():
        _identifier(axis_id, f"{label} ID")
        entry = _table(raw_entry, f"{label}.{axis_id}")
        _unknown_keys(entry, allowed, f"{label}.{axis_id}")
        config = _table(entry.get("config", {}), f"{label}.{axis_id}.config")
        if "outputs" in config:
            raise PlanError(f"{label}.{axis_id}.config.outputs is managed by the harness")
        normalized: dict[str, Any] = {"config": copy.deepcopy(config)}
        if model:
            maximum = entry.get("max_parallel_runs")
            if maximum is not None:
                normalized["max_parallel_runs"] = _positive_int(
                    maximum, f"{label}.{axis_id}.max_parallel_runs",
                )
            verify = entry.get("verify_endpoint", False)
            if not isinstance(verify, bool):
                raise PlanError(f"{label}.{axis_id}.verify_endpoint must be true or false")
            normalized["verify_endpoint"] = verify
        result[axis_id] = normalized
    return result


def _matrix_ids(matrix: Mapping[str, Any], key: str, defined: Mapping[str, Any]) -> list[str]:
    values = matrix.get(key)
    if not isinstance(values, list) or not values or any(not isinstance(v, str) for v in values):
        raise PlanError(f"matrix.{key} must be a non-empty array of IDs")
    if len(set(values)) != len(values):
        raise PlanError(f"matrix.{key} contains duplicate IDs")
    missing = sorted(set(values) - set(defined))
    if missing:
        raise PlanError(f"matrix.{key} references unknown ID(s): {', '.join(missing)}")
    return sorted(values)


def _flatten(value: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Any]:
    flat: dict[tuple[str, ...], Any] = {}
    for key, item in value.items():
        path = prefix + (str(key),)
        if isinstance(item, dict) and item:
            flat.update(_flatten(item, path))
        else:
            flat[path] = item
    return flat


def _check_axis_conflicts(overlays: Sequence[tuple[str, Mapping[str, Any]]]) -> None:
    assigned: dict[tuple[str, ...], tuple[str, Any]] = {}
    for owner, overlay in overlays:
        for path, value in _flatten(overlay).items():
            for previous_path, (previous_owner, previous_value) in assigned.items():
                overlaps = (
                    path == previous_path
                    or path[: len(previous_path)] == previous_path
                    or previous_path[: len(path)] == path
                )
                if overlaps and (path != previous_path or value != previous_value):
                    dotted = ".".join(path if len(path) >= len(previous_path) else previous_path)
                    raise PlanError(
                        f"axis conflict at {dotted}: {previous_owner} and {owner} set unequal values"
                    )
            assigned[path] = (owner, value)


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _nested_get(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """The value at a dotted path of nested tables, or None when any level is absent."""
    node: Any = data
    for key in path[:-1]:
        node = node.get(key) if isinstance(node, Mapping) else None
    return node.get(path[-1]) if isinstance(node, Mapping) else None


def _nested_set(data: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    node: Any = data
    for key in path[:-1]:
        node = node.get(key) if isinstance(node, dict) else None
    if isinstance(node, dict) and path[-1] in node:
        node[path[-1]] = value


def _resolve_paths(config: dict[str, Any], source_dir: Path) -> None:
    for path in _RESOLVED_PATHS:
        value = _nested_get(config, path)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise PlanError(f"{'.'.join(path)} must be a filesystem path string")
        if "://" in value:
            raise PlanError(f"{'.'.join(path)} must be a pinned local path, not a URL")
        resolved = Path(value).expanduser()
        if not resolved.is_absolute():
            resolved = source_dir / resolved
        _nested_set(config, path, str(resolved.resolve()))


def directory_digest(path: str | os.PathLike[str]) -> tuple[str, int, int]:
    """Hash a directory from sorted relative paths, file digests, and sizes."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"not a directory: {root}")
    digest = hashlib.sha256()
    total_bytes = 0
    files = 0
    for child in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix()):
        relative = child.relative_to(root).as_posix()
        child_digest = sha256_file(child)
        size = child.stat().st_size
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(child_digest.encode("ascii") + b"\0")
        digest.update(str(size).encode("ascii") + b"\n")
        total_bytes += size
        files += 1
    return digest.hexdigest(), total_bytes, files


def _fingerprint_inputs(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    fingerprints: dict[str, dict[str, Any]] = {}
    for field in sorted(_FINGERPRINTED_PATHS):
        value = _nested_get(config, field)
        if value is None or value == "":
            continue
        path = Path(value)
        if path.is_file():
            fingerprints[".".join(field)] = {
                "path": str(path),
                "kind": "file",
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        elif path.is_dir():
            digest, size, count = directory_digest(path)
            fingerprints[".".join(field)] = {
                "path": str(path),
                "kind": "directory",
                "sha256": digest,
                "bytes": size,
                "files": count,
            }
        else:
            raise PlanError(f"configured input does not exist: {'.'.join(field)}={path}")
    return fingerprints


def source_digest() -> str:
    """Return a deterministic digest of the installed LogMapLLM Python source."""
    package = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    runtime_sources = (
        path
        for path in package.rglob("*.py")
        if "tests" not in path.relative_to(package).parts
        and not path.name.startswith("test_")
    )
    for source in sorted(runtime_sources, key=lambda p: p.as_posix()):
        digest.update(source.relative_to(package).as_posix().encode("utf-8") + b"\0")
        digest.update(sha256_file(source).encode("ascii") + b"\n")
    return digest.hexdigest()


def _inject_outputs(config: dict[str, Any], root: Path) -> None:
    config["outputs"] = {
        "logmapllm_output_dirpath": str(root / "logmapllm-outputs"),
        "logmap_initial_alignment_output_dirpath": str(root / "logmap-initial-alignment"),
        "logmap_refined_alignment_output_dirpath": str(root / "logmap-refined-alignment"),
    }


def _validate_native(config: Mapping[str, Any], label: str) -> dict[str, Any]:
    api_key = config.get("oracle", {}).get("api_key", "EMPTY") if isinstance(config.get("oracle"), dict) else "EMPTY"
    if not isinstance(api_key, str) or (api_key != "EMPTY" and not _ENV_RE.fullmatch(api_key)):
        raise PlanError(f"{label}: oracle.api_key must be 'EMPTY' or ENV:VARIABLE")
    try:
        validated = validate_config(copy.deepcopy(dict(config)))
    except Exception as exc:
        raise PlanError(f"{label}: invalid LogMapLLM config: {exc}") from exc
    if validated.automatic_model_selection:
        raise PlanError(
            f"{label}: model_selection.automatic is a single-run feature; a batch fixes each "
            "job's model through its models axis"
        )
    # A campaign may never be ambiguous about how its pseudo-negatives were built. Enforced
    # at generation rather than in the config schema: frozen per-job configs in completed
    # batches carry few_shot_k > 0 and no layout, and are re-validated by aggregate/import/
    # collector paths, so a schema-level requirement would break completed batches.
    if validated.few_shot.few_shot_k > 0 and validated.few_shot.rag_negative_layout is None:
        raise PlanError(
            f"{label}: few_shot.rag_negative_layout is required when few_shot_k > 0. "
            "Choose 'paired-sibling-v2' (negative = positive with its target replaced by an "
            "ontology sibling) or 'paired-donor-v2' (target of the next ranked donor). "
            "'donor-cross-v1' is FROZEN: it reproduces the completed campaigns and must not "
            "be used for new science."
        )
    logmap_value = validated.alignmentTask.logmap_parameters_dirpath.strip()
    if not logmap_value:
        raise PlanError(
            f"{label}: alignmentTask.logmap_parameters_dirpath must be explicit for batch runs"
        )
    logmap_dir = Path(logmap_value)
    required_file = logmap_dir / "logmap-matcher-4.0.jar"
    required_dependencies = logmap_dir / "java-dependencies"
    required_parameters = logmap_dir / "parameters.txt"
    if not required_file.is_file():
        raise PlanError(f"{label}: LogMap JAR not found: {required_file}")
    if not required_dependencies.is_dir():
        raise PlanError(f"{label}: Java dependencies directory not found: {required_dependencies}")
    if not required_parameters.is_file():
        raise PlanError(f"{label}: LogMap parameters file not found: {required_parameters}")
    return validated.model_dump(mode="json", exclude_none=True)


def _check_authored_secret(config: Mapping[str, Any], label: str) -> None:
    oracle = config.get("oracle")
    if not isinstance(oracle, dict):
        return
    for key in ("api_key", "openrouter_apikey"):
        if key not in oracle:
            continue
        value = oracle[key]
        if not isinstance(value, str) or (value != "EMPTY" and not _ENV_RE.fullmatch(value)):
            raise PlanError(f"{label}.{key} must be 'EMPTY' or ENV:VARIABLE")


def _check_partial_native_config(config: Mapping[str, Any], label: str) -> None:
    """Reject unknown native keys even in axes not selected by the matrix."""
    sections = set(LogMapLLMConfig.model_fields) - {"outputs"}
    _unknown_keys(config, sections, label)
    for section_name, section in config.items():
        if not isinstance(section, dict):
            raise PlanError(f"{label}.{section_name} must be a TOML table")
        model = LogMapLLMConfig.model_fields[section_name].annotation
        fields = set(getattr(model, "model_fields", {}))
        _unknown_keys(section, fields, f"{label}.{section_name}")


def _identity_config(
    config: Mapping[str, Any], fingerprints: Mapping[str, Mapping[str, Any]], *,
    scientific: bool,
) -> dict[str, Any]:
    identity = copy.deepcopy(dict(config))
    identity.pop("outputs", None)
    oracle = identity.get("oracle")
    if isinstance(oracle, dict):
        oracle.pop("api_key", None)
        if scientific:
            oracle.pop("base_url", None)
            oracle.pop("max_workers", None)
            oracle.pop("local_oracle_predictions_dirpath", None)
    few_shot = identity.get("few_shot")
    if scientific and isinstance(few_shot, dict):
        few_shot.pop("rag_cache_dir", None)
    alignment_task = identity.get("alignmentTask")
    if scientific and isinstance(alignment_task, dict):
        alignment_task.pop("logmap_jvm_memory", None)
    for dotted, fingerprint in fingerprints.items():
        _nested_set(
            identity, tuple(dotted.split(".")),
            {"kind": fingerprint["kind"], "sha256": fingerprint["sha256"]},
        )
    return identity


def _alignment_id(
    task: str,
    config: Mapping[str, Any],
    fingerprints: Mapping[str, Mapping[str, Any]],
    source_digest: str,
) -> str | None:
    mode = config["pipeline"]["align_ontologies"]
    if mode in ("bypass", "external"):
        # external: the M_ask comes from a file, there is no LogMap alignment to share
        return None
    if mode != "align":
        raise PlanError(
            "batch alignment reuse requires pipeline.align_ontologies='align' "
            "(or 'bypass' / 'external'); the harness supplies reuse inputs at execution time"
        )
    alignment_inputs = {
        key: {"kind": value["kind"], "sha256": value["sha256"]}
        for key, value in fingerprints.items()
        if key.startswith("alignmentTask.")
    }
    alignment_task = copy.deepcopy(config["alignmentTask"])
    alignment_task.pop("logmap_jvm_memory", None)
    for dotted, fingerprint in alignment_inputs.items():
        alignment_task[dotted.split(".", 1)[1]] = fingerprint
    value = {
        "schema": SCHEMA_VERSION,
        "task": task,
        "alignmentTask": alignment_task,
        "inputs": alignment_inputs,
        "core_sha256": source_digest,
    }
    return canonical_hash(value)[:16]


def _parse_spec(source: Path) -> dict[str, Any]:
    raw = load_toml(source)
    _unknown_keys(raw, _TOP_KEYS, "document root")
    if type(raw.get("schema")) is not int or raw["schema"] != SCHEMA_VERSION:
        raise PlanError(f"schema must be the integer {SCHEMA_VERSION}")

    batch = _table(raw.get("batch"), "batch")
    _unknown_keys(batch, _BATCH_KEYS, "batch")
    name = _identifier(batch.get("name"), "batch.name")
    output_value = batch.get("output_root")
    if not isinstance(output_value, str) or not output_value.strip():
        raise PlanError("batch.output_root must be a non-empty path string")
    output_root = Path(output_value).expanduser()
    if not output_root.is_absolute():
        output_root = source.parent / output_root
    jobs = _positive_int(batch.get("jobs", 1), "batch.jobs")
    timeout = _positive_int(batch.get("timeout_seconds"), "batch.timeout_seconds")
    reuse = batch.get("reuse_alignments", True)
    if not isinstance(reuse, bool):
        raise PlanError("batch.reuse_alignments must be true or false")

    defaults = _table(raw.get("defaults", {}), "defaults")
    if "outputs" in defaults:
        raise PlanError("defaults.outputs is managed by the harness")
    tasks = _axis(raw.get("tasks"), "tasks")
    models = _axis(raw.get("models"), "models", model=True)
    conditions = _axis(raw.get("conditions"), "conditions")
    _check_authored_secret(defaults, "defaults.oracle")
    _check_partial_native_config(defaults, "defaults")
    for axis_name, entries in (("tasks", tasks), ("models", models), ("conditions", conditions)):
        for axis_id, entry in entries.items():
            _check_authored_secret(entry["config"], f"{axis_name}.{axis_id}.config.oracle")
            _check_partial_native_config(entry["config"], f"{axis_name}.{axis_id}.config")

    matrix = _table(raw.get("matrix"), "matrix")
    _unknown_keys(matrix, _MATRIX_KEYS, "matrix")
    task_ids = _matrix_ids(matrix, "tasks", tasks)
    model_ids = _matrix_ids(matrix, "models", models)
    condition_ids = _matrix_ids(matrix, "conditions", conditions)
    repeats = _positive_int(matrix.get("repeats", 1), "matrix.repeats")

    excludes = matrix.get("exclude", [])
    if not isinstance(excludes, list):
        raise PlanError("matrix.exclude must be an array of inline tables")
    normalized_excludes: list[dict[str, Any]] = []
    for index, raw_exclude in enumerate(excludes):
        exclude = _table(raw_exclude, f"matrix.exclude[{index}]")
        _unknown_keys(exclude, {"task", "model", "condition", "repeat"}, f"matrix.exclude[{index}]")
        if not {"task", "model", "condition"}.issubset(exclude):
            raise PlanError(
                f"matrix.exclude[{index}] must name task, model, and condition; repeat is optional"
            )
        for key, defined in (("task", tasks), ("model", models), ("condition", conditions)):
            value = exclude[key]
            if not isinstance(value, str) or value not in defined:
                raise PlanError(f"matrix.exclude[{index}].{key} references an unknown ID")
        if "repeat" in exclude:
            repeat = exclude["repeat"]
            if isinstance(repeat, bool) or not isinstance(repeat, int) or not 0 <= repeat < repeats:
                raise PlanError(f"matrix.exclude[{index}].repeat is outside 0..{repeats - 1}")
        normalized_excludes.append(dict(exclude))

    aggregate = _table(raw.get("aggregate", {}), "aggregate")
    _unknown_keys(aggregate, _AGGREGATE_KEYS, "aggregate")
    group_by = aggregate.get("group_by", ["task", "model", "condition"])
    if (
        not isinstance(group_by, list) or not group_by
        or any(value not in {"task", "model", "condition", "repeat"} for value in group_by)
        or len(set(group_by)) != len(group_by)
    ):
        raise PlanError("aggregate.group_by must contain unique task/model/condition/repeat names")
    require_complete = aggregate.get("require_complete", True)
    if not isinstance(require_complete, bool):
        raise PlanError("aggregate.require_complete must be true or false")

    return {
        "batch": {
            "name": name,
            "output_root": output_root.resolve(),
            "jobs": jobs,
            "timeout_seconds": timeout,
            "reuse_alignments": reuse,
        },
        "defaults": copy.deepcopy(defaults),
        "tasks": tasks,
        "models": models,
        "conditions": conditions,
        "matrix": {
            "tasks": task_ids,
            "models": model_ids,
            "conditions": condition_ids,
            "repeats": repeats,
            "exclude": normalized_excludes,
        },
        "aggregate": {"group_by": list(group_by), "require_complete": require_complete},
    }


def _excluded(cell: Mapping[str, Any], excludes: Iterable[Mapping[str, Any]]) -> bool:
    return any(all(cell[key] == value for key, value in exclusion.items()) for exclusion in excludes)


def _job_id(task: str, model: str, condition: str, repeat: int, execution_hash: str) -> str:
    suffix = execution_hash[:8]
    prefix = f"{task}-{model}-{condition}-r{repeat}"
    maximum_prefix = 96 - len(suffix) - 1
    return f"{prefix[:maximum_prefix].rstrip('-_')}-{suffix}"


def _build(source: Path) -> tuple[Path, dict[str, Any], dict[str, str], dict[str, str]]:
    spec = _parse_spec(source)
    batch = spec["batch"]
    core_digest = source_digest()
    spec_digest = sha256_file(source)
    placeholder = source.parent / ".logmap-llm-plan-placeholder"

    planned: list[dict[str, Any]] = []
    matrix = spec["matrix"]
    for task in matrix["tasks"]:
        for model in matrix["models"]:
            for condition in matrix["conditions"]:
                for repeat in range(matrix["repeats"]):
                    cell = {"task": task, "model": model, "condition": condition, "repeat": repeat}
                    if _excluded(cell, matrix["exclude"]):
                        continue
                    overlays = [
                        (f"tasks.{task}", spec["tasks"][task]["config"]),
                        (f"models.{model}", spec["models"][model]["config"]),
                        (f"conditions.{condition}", spec["conditions"][condition]["config"]),
                    ]
                    _check_axis_conflicts(overlays)
                    config = copy.deepcopy(spec["defaults"])
                    for _owner, overlay in overlays:
                        config = _deep_merge(config, overlay)
                    if "outputs" in config:
                        raise PlanError("outputs are managed by the harness")
                    _resolve_paths(config, source.parent)
                    _inject_outputs(config, placeholder)
                    normalized = _validate_native(config, f"{task}/{model}/{condition}/r{repeat}")
                    fingerprints = _fingerprint_inputs(normalized)

                    condition_identity = canonical_hash({
                        "schema": SCHEMA_VERSION,
                        "task": task,
                        "model": model,
                        "condition": condition,
                        "config": _identity_config(normalized, fingerprints, scientific=True),
                        "core_sha256": core_digest,
                    })
                    model_meta = spec["models"][model]
                    execution_identity = canonical_hash({
                        "schema": SCHEMA_VERSION,
                        "condition_id": condition_identity,
                        "repeat": repeat,
                        "config": _identity_config(normalized, fingerprints, scientific=False),
                        "timeout_seconds": batch["timeout_seconds"],
                        "max_parallel_runs": model_meta.get("max_parallel_runs"),
                        "verify_endpoint": model_meta["verify_endpoint"],
                    })
                    alignment_id = (
                        _alignment_id(task, normalized, fingerprints, core_digest)
                        if batch["reuse_alignments"] else None
                    )
                    if not batch["reuse_alignments"] and normalized["pipeline"]["align_ontologies"] == "reuse":
                        raise PlanError("align_ontologies='reuse' requires batch.reuse_alignments=true")
                    planned.append({
                        **cell,
                        "id": _job_id(task, model, condition, repeat, execution_identity),
                        "condition_id": condition_identity,
                        "execution_hash": execution_identity,
                        "alignment_id": alignment_id,
                        "normalized": normalized,
                        "input_fingerprints": fingerprints,
                        "max_parallel_runs": model_meta.get("max_parallel_runs"),
                        "verify_endpoint": model_meta["verify_endpoint"],
                    })

    if not planned:
        raise PlanError("matrix expansion produced no jobs")
    planned.sort(key=lambda job: (job["task"], job["model"], job["condition"], job["repeat"]))
    plan_hash = canonical_hash({
        "schema": SCHEMA_VERSION,
        "spec_sha256": spec_digest,
        "core_sha256": core_digest,
        "batch": {
            "name": batch["name"], "jobs": batch["jobs"],
            "timeout_seconds": batch["timeout_seconds"],
            "reuse_alignments": batch["reuse_alignments"],
        },
        "jobs": [
            {key: job[key] for key in ("id", "condition_id", "execution_hash", "alignment_id")}
            for job in planned
        ],
        "aggregate": spec["aggregate"],
    })
    batch_id = f"{batch['name']}-{plan_hash[:12]}"
    batch_dir = batch["output_root"] / batch_id

    config_texts: dict[str, str] = {}
    job_rows: list[dict[str, Any]] = []
    for job in planned:
        config = copy.deepcopy(job["normalized"])
        _inject_outputs(config, batch_dir / "jobs" / job["id"] / "standalone")
        config = _validate_native(config, job["id"])
        relative = f"configs/{job['id']}.toml"
        text = _toml_text(config)
        config_texts[relative] = text
        job_rows.append({
            "id": job["id"],
            "task": job["task"],
            "model": job["model"],
            "condition": job["condition"],
            "repeat": job["repeat"],
            "condition_id": job["condition_id"],
            "execution_hash": job["execution_hash"],
            "alignment_id": job["alignment_id"],
            "config_path": relative,
            "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "timeout_seconds": batch["timeout_seconds"],
            "max_parallel_runs": job["max_parallel_runs"],
            "verify_endpoint": job["verify_endpoint"],
            "expected_served_model": config["oracle"]["model_name"],
            "input_fingerprints": job["input_fingerprints"],
        })

    alignment_texts: dict[str, str] = {}
    alignment_rows: list[dict[str, Any]] = []
    for alignment_id in sorted({job["alignment_id"] for job in planned if job["alignment_id"]}):
        members = [job for job in planned if job["alignment_id"] == alignment_id]
        donor = members[0]
        config = copy.deepcopy(donor["normalized"])
        config["pipeline"].update({
            "align_ontologies": "align",
            "build_oracle_prompts": "bypass",
            "consult_oracle": "bypass",
            "refine_alignment": "bypass",
        })
        config["evaluation"]["evaluate"] = False
        _inject_outputs(config, batch_dir / "alignments" / alignment_id / "standalone")
        config = _validate_native(config, f"alignment {alignment_id}")
        relative = f"alignments/{alignment_id}/config.toml"
        text = _toml_text(config)
        alignment_texts[relative] = text
        alignment_rows.append({
            "id": alignment_id,
            "task": donor["task"],
            "config_path": relative,
            "config_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "timeout_seconds": batch["timeout_seconds"],
            "input_fingerprints": {
                key: value for key, value in donor["input_fingerprints"].items()
                if key.startswith("alignmentTask.")
            },
            "job_ids": [job["id"] for job in members],
        })

    manifest = seal_manifest({
        "schema": SCHEMA_VERSION,
        "batch": {
            "name": batch["name"],
            "id": batch_id,
            "plan_hash": plan_hash,
            "jobs": batch["jobs"],
            "expected_jobs": len(job_rows),
            "timeout_seconds": batch["timeout_seconds"],
            "reuse_alignments": batch["reuse_alignments"],
        },
        "source": {"spec_sha256": spec_digest, "core_sha256": core_digest},
        # Sealed, but outside run identity: two hosts running the same campaign must
        # produce identical job ids even though their virtualenvs differ, so cross-host
        # comparisons work modulo `environment` and `seal`.
        "environment": environment_payload(Path(__file__).resolve().parents[3]),
        "alignments": alignment_rows,
        "jobs": job_rows,
        "aggregate": spec["aggregate"],
    })
    return batch_dir, manifest, config_texts, alignment_texts


def _plan_tsv(manifest: Mapping[str, Any]) -> str:
    rows: list[list[Any]] = [[
        "id", "task", "model", "condition", "repeat", "alignment_id", "config_path",
    ]]
    for job in manifest["jobs"]:
        rows.append([
            job["id"], job["task"], job["model"], job["condition"], job["repeat"],
            job["alignment_id"] or "", job["config_path"],
        ])
    output: list[str] = []
    for row in rows:
        output.append("\t".join(str(value) for value in row))
    return "\n".join(output) + "\n"


def _print_plan(batch_dir: Path, manifest: Mapping[str, Any], dry_run: bool) -> None:
    batch = manifest["batch"]
    print(f"Batch: {batch['name']}")
    print(f"Expected jobs: {batch['expected_jobs']}")
    print(f"Alignment prerequisites: {len(manifest['alignments'])}")
    print()
    print(f"{'JOB':<50} {'TASK':<16} {'MODEL':<16} CONDITION")
    for job in manifest["jobs"]:
        print(f"{job['id']:<50} {job['task']:<16} {job['model']:<16} {job['condition']}")
    print()
    print("No files written (--dry-run)." if dry_run else f"Generated: {batch_dir}")


def generate_batch(
    spec_path: str | os.PathLike[str], *, dry_run: bool = False,
) -> Path:
    """Validate, expand, and optionally freeze one batch specification."""
    source = Path(spec_path).expanduser().resolve()
    if not source.is_file():
        raise PlanError(f"batch specification not found: {source}")
    batch_dir, manifest, config_texts, alignment_texts = _build(source)
    if dry_run:
        _print_plan(batch_dir, manifest, True)
        return batch_dir

    if batch_dir.exists():
        existing = load_manifest(batch_dir)
        # Compare modulo `environment` (and `seal`, which covers it) so a `uv sync` between
        # generate runs does not make an identical batch look like a conflict. The original
        # environment record is kept: it says what the batch was generated in.
        if manifests_equivalent(existing, manifest):
            changed = environments_differ(
                existing.get("environment"), manifest.get("environment"),
            )
            if changed:
                print(
                    "note: batch already exists and is unchanged; keeping its original "
                    f"environment record (now differs in: {', '.join(changed)})"
                )
            _print_plan(batch_dir, manifest, False)
            return batch_dir
        raise PlanError(f"refusing to overwrite conflicting batch directory: {batch_dir}")

    batch_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{manifest['batch']['id']}.", dir=batch_dir.parent))
    try:
        atomic_write_text(staging / "batch.toml", source.read_text(encoding="utf-8"))
        atomic_write_json(staging / "manifest.json", manifest)
        atomic_write_text(staging / "plan.tsv", _plan_tsv(manifest))
        for relative, text in {**config_texts, **alignment_texts}.items():
            atomic_write_text(staging / relative, text)
        for job in manifest["jobs"]:
            (staging / "jobs" / job["id"] / "attempts").mkdir(parents=True, exist_ok=True)
        for alignment in manifest["alignments"]:
            (staging / "alignments" / alignment["id"] / "attempts").mkdir(
                parents=True, exist_ok=True,
            )
        os.replace(staging, batch_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _print_plan(batch_dir, manifest, False)
    return batch_dir


__all__ = [
    "PlanError",
    "atomic_write_json",
    "atomic_write_text",
    "canonical_hash",
    "directory_digest",
    "generate_batch",
    "load_manifest",
    "load_toml",
    "manifest_identity",
    "manifests_equivalent",
    "select_jobs",
    "seal_manifest",
    "sha256_file",
    "source_digest",
    "verify_manifest_seal",
    "write_toml",
]
