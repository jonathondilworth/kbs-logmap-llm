"""Import one verified initial alignment into a fresh frozen batch.

This is intentionally a copy operation, not pipeline execution.  A transfer
receipt binds the destination to its exact sealed batch and separately records
the manifest of the system that actually produced the copied files.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from logmap_llm.experiments.plan import atomic_write_json, load_manifest, sha256_file
from logmap_llm.experiments.run import (
    BatchRunError,
    _batch_lock,
    _internal_path,
    _load_config,
    _now,
    _verify_frozen_row,
    validate_completion,
    verify_batch_identity,
)


RECEIPT_KIND = "logmap-llm-alignment-transfer"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_CORE_SUFFIXES = (
    "-logmap_mappings.txt",
    "-logmap_mappings.tsv",
    "-logmap_mappings_to_ask_oracle_user_llm.txt",
)


def _read_receipt(path: Path) -> tuple[dict[str, Any], str, int]:
    if path.is_symlink() or not path.is_file():
        raise BatchRunError(f"receipt must be a regular, non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BatchRunError(f"cannot read a valid alignment receipt from {path}: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "kind",
        "destination",
        "producer",
        "source_task_filename_prefix",
        "artifacts",
    }:
        raise BatchRunError("alignment receipt has an unsupported shape")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise BatchRunError("alignment receipt schema must be the integer 1")
    if value["kind"] != RECEIPT_KIND:
        raise BatchRunError(f"alignment receipt kind must be {RECEIPT_KIND!r}")

    destination = value["destination"]
    destination_keys = {
        "batch_id",
        "manifest_sha256",
        "alignment_id",
        "task",
        "config_sha256",
        "core_sha256",
    }
    if not isinstance(destination, dict) or set(destination) != destination_keys:
        raise BatchRunError("alignment receipt destination has an unsupported shape")
    for key in ("batch_id", "alignment_id", "task"):
        if not isinstance(destination[key], str) or not destination[key]:
            raise BatchRunError(f"alignment receipt destination.{key} must be non-empty")
    for key in ("manifest_sha256", "config_sha256", "core_sha256"):
        if (
            not isinstance(destination[key], str)
            or not _SHA256_RE.fullmatch(destination[key])
        ):
            raise BatchRunError(
                f"alignment receipt destination.{key} must be a SHA-256 digest"
            )

    producer = value["producer"]
    if not isinstance(producer, dict) or set(producer) != {"description", "manifest"}:
        raise BatchRunError("alignment receipt producer has an unsupported shape")
    if not isinstance(producer["description"], str) or not producer["description"].strip():
        raise BatchRunError("alignment receipt producer.description must be non-empty")
    producer_manifest = producer["manifest"]
    if (
        not isinstance(producer_manifest, dict)
        or set(producer_manifest) != {"path", "sha256", "bytes"}
    ):
        raise BatchRunError("alignment receipt producer.manifest has an unsupported shape")
    _safe_relative(producer_manifest["path"], "producer.manifest.path")
    if (
        not isinstance(producer_manifest["sha256"], str)
        or not _SHA256_RE.fullmatch(producer_manifest["sha256"])
        or isinstance(producer_manifest["bytes"], bool)
        or not isinstance(producer_manifest["bytes"], int)
        or producer_manifest["bytes"] < 0
    ):
        raise BatchRunError("alignment receipt producer.manifest hash or size is invalid")

    prefix = value["source_task_filename_prefix"]
    if not isinstance(prefix, str) or _SAFE_PREFIX_RE.fullmatch(prefix) is None:
        raise BatchRunError(
            "alignment receipt source_task_filename_prefix is not a safe filename prefix"
        )

    artifacts = value["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise BatchRunError("alignment receipt artifacts must be a non-empty list")
    seen: set[str] = set()
    for index, record in enumerate(artifacts):
        if not isinstance(record, dict) or set(record) != {"path", "sha256", "bytes"}:
            raise BatchRunError(f"alignment receipt artifacts[{index}] has an unsupported shape")
        relative = _safe_relative(record["path"], f"artifacts[{index}].path")
        normalized = relative.as_posix()
        if normalized in seen:
            raise BatchRunError(f"alignment receipt repeats artifact path {normalized!r}")
        seen.add(normalized)
        if not isinstance(record["sha256"], str) or not _SHA256_RE.fullmatch(record["sha256"]):
            raise BatchRunError(f"alignment receipt artifacts[{index}].sha256 is invalid")
        size = record["bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise BatchRunError(f"alignment receipt artifacts[{index}].bytes is invalid")
    return value, sha256_file(path), path.stat().st_size


def _safe_relative(value: Any, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise BatchRunError(f"alignment receipt {label} must be a portable relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or path == PurePosixPath(".")
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise BatchRunError(f"alignment receipt {label} escapes or is not canonical: {value!r}")
    return path


def _select_alignment(
    manifest: Mapping[str, Any], *, task: str | None, alignment_id: str | None
) -> dict[str, Any]:
    if (task is None) == (alignment_id is None):
        raise BatchRunError("select exactly one alignment with --task or --alignment-id")
    if task is not None:
        matches = [row for row in manifest["alignments"] if row["task"] == task]
        label = f"task {task!r}"
    else:
        matches = [row for row in manifest["alignments"] if row["id"] == alignment_id]
        label = f"alignment ID {alignment_id!r}"
    if len(matches) != 1:
        raise BatchRunError(f"{label} selected {len(matches)} alignments; expected exactly one")
    return dict(matches[0])


def _verify_receipt_identity(
    batch_dir: Path,
    manifest: Mapping[str, Any],
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> None:
    destination = receipt["destination"]
    expected = {
        "batch_id": manifest["batch"]["id"],
        "manifest_sha256": sha256_file(batch_dir / "manifest.json"),
        "alignment_id": row["id"],
        "task": row["task"],
        "config_sha256": row["config_sha256"],
        "core_sha256": manifest["source"]["core_sha256"],
    }
    mismatched = [
        key for key, value in expected.items() if destination.get(key) != value
    ]
    if mismatched:
        raise BatchRunError(
            "alignment receipt does not belong to this frozen alignment: "
            + ", ".join(mismatched)
        )


def validate_import_completion(
    batch_dir: Path,
    attempt: Path,
    complete: Mapping[str, Any],
) -> bool:
    """Check that an imported completion still matches its receipt and batch."""
    provenance = complete.get("import_provenance")
    if complete.get("kind") != "alignment" or not isinstance(provenance, dict):
        return False
    if set(provenance) != {
        "method",
        "receipt",
        "destination",
        "producer",
        "producer_manifest",
        "source_task_filename_prefix",
    } or provenance.get("method") != "verified-copy-v1":
        return False
    prefix = provenance.get("source_task_filename_prefix")
    receipt_record = provenance.get("receipt")
    if (
        not isinstance(prefix, str)
        or _SAFE_PREFIX_RE.fullmatch(prefix) is None
        or not isinstance(receipt_record, dict)
        or set(receipt_record) != {"path", "sha256", "bytes"}
    ):
        return False
    receipt_path = _internal_path(batch_dir, receipt_record["path"])
    if receipt_path.parent != attempt.resolve() or receipt_path.name != "import-receipt.json":
        return False
    receipt, digest, size = _read_receipt(receipt_path)
    if digest != receipt_record.get("sha256") or size != receipt_record.get("bytes"):
        return False
    if (
        receipt["destination"] != provenance.get("destination")
        or receipt["producer"] != provenance.get("producer")
        or receipt["source_task_filename_prefix"] != prefix
    ):
        return False
    if (
        complete.get("config_sha256") != receipt["destination"]["config_sha256"]
        or complete.get("core_sha256") != receipt["destination"]["core_sha256"]
    ):
        return False

    producer_record = provenance.get("producer_manifest")
    if not isinstance(producer_record, dict) or set(producer_record) != {
        "path", "sha256", "bytes",
    }:
        return False
    producer_path = _internal_path(batch_dir, producer_record["path"])
    if (
        producer_path.parent != attempt.resolve()
        or producer_path.name != "producer-manifest.json"
        or producer_path.is_symlink()
        or not producer_path.is_file()
        or producer_path.stat().st_size != producer_record.get("bytes")
        or sha256_file(producer_path) != producer_record.get("sha256")
        or producer_record.get("sha256") != receipt["producer"]["manifest"]["sha256"]
        or producer_record.get("bytes") != receipt["producer"]["manifest"]["bytes"]
    ):
        return False

    manifest = load_manifest(batch_dir)
    rows = [row for row in manifest["alignments"] if row["id"] == complete.get("id")]
    if len(rows) != 1:
        return False
    _verify_receipt_identity(batch_dir, manifest, rows[0], receipt)
    receipt_pairs = sorted(
        (record["sha256"], record["bytes"]) for record in receipt["artifacts"]
    )
    artifacts = complete.get("artifacts")
    if not isinstance(artifacts, list) or any(not isinstance(item, dict) for item in artifacts):
        return False
    complete_pairs = sorted(
        (record.get("sha256"), record.get("bytes")) for record in artifacts
    )
    return receipt_pairs == complete_pairs


def _assert_fresh(batch_dir: Path, row: Mapping[str, Any]) -> Path:
    owner = batch_dir / "alignments" / row["id"]
    attempts = owner / "attempts"
    if not owner.is_dir() or not attempts.is_dir():
        raise BatchRunError(f"alignment owner is not a generated batch directory: {owner}")
    allowed = {"config.toml", "attempts"}
    unexpected = sorted(path.name for path in owner.iterdir() if path.name not in allowed)
    if unexpected or any(attempts.iterdir()):
        detail = ", ".join(unexpected) if unexpected else "existing attempt entries"
        raise BatchRunError(f"alignment owner is not fresh ({detail}): {owner}")

    jobs = {job["id"]: job for job in load_manifest(batch_dir)["jobs"]}
    for job_id in row["job_ids"]:
        job_attempts = batch_dir / "jobs" / job_id / "attempts"
        if job_id not in jobs or not job_attempts.is_dir():
            raise BatchRunError(f"dependent job owner is malformed: {job_id}")
        if any(job_attempts.iterdir()):
            raise BatchRunError(f"dependent job {job_id} already has an attempt")
    return owner


def _source_file(source_dir: Path, relative: PurePosixPath) -> Path:
    current = source_dir
    for index, part in enumerate(relative.parts):
        current = current / part
        try:
            mode = current.lstat().st_mode
        except OSError as exc:
            raise BatchRunError(f"source artifact is missing: {current}") from exc
        if stat.S_ISLNK(mode):
            raise BatchRunError(f"source artifacts may not use symlinks: {current}")
        if index < len(relative.parts) - 1 and not stat.S_ISDIR(mode):
            raise BatchRunError(f"source artifact parent is not a directory: {current}")
    if not stat.S_ISREG(current.lstat().st_mode):
        raise BatchRunError(f"source artifact must be a regular file: {current}")
    return current


def _verified_sources(
    source_dir: Path, receipt: Mapping[str, Any]
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    if source_dir.is_symlink() or not source_dir.is_dir():
        raise BatchRunError(
            f"source directory must be a regular, non-symlink directory: {source_dir}"
        )
    verified: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for record in receipt["artifacts"]:
        relative = _safe_relative(record["path"], "artifact path")
        path = _source_file(source_dir, relative)
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise BatchRunError(f"source artifact hash or size does not match receipt: {path}")
        verified[relative.as_posix()] = (path, record)
    return verified


def _verified_producer_manifest(
    source_dir: Path, receipt: Mapping[str, Any]
) -> tuple[Path, Mapping[str, Any]]:
    record = receipt["producer"]["manifest"]
    path = _source_file(
        source_dir, _safe_relative(record["path"], "producer.manifest.path")
    )
    if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
        raise BatchRunError(f"producer manifest hash or size does not match receipt: {path}")
    return path, record


def _destination_paths(
    verified: Mapping[str, tuple[Path, Mapping[str, Any]]],
    *,
    source_prefix: str,
    destination_task: str,
) -> dict[str, tuple[Path, Mapping[str, Any]]]:
    if not _SAFE_PREFIX_RE.fullmatch(source_prefix):
        raise BatchRunError(
            "source task prefix must use letters, digits, '_', '-', or '.' "
            "and start alphanumerically"
        )
    renamed = {
        source_prefix + suffix: destination_task + suffix for suffix in _CORE_SUFFIXES
    }
    missing = sorted(path for path in renamed if path not in verified)
    if missing:
        raise BatchRunError(
            "source alignment is missing required artifact(s): " + ", ".join(missing)
        )

    destinations: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for relative, source in verified.items():
        destination = renamed.get(relative, relative)
        _safe_relative(destination, "destination artifact path")
        if destination in destinations:
            raise BatchRunError(
                f"source artifacts collide after task-prefix rename: {destination}"
            )
        destinations[destination] = source
    return destinations


def _copy_verified(
    destinations: Mapping[str, tuple[Path, Mapping[str, Any]]], staged: Path
) -> None:
    staged.mkdir()
    for relative, (source, record) in destinations.items():
        target = staged.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target, follow_symlinks=False)
        if target.stat().st_size != record["bytes"] or sha256_file(target) != record["sha256"]:
            raise BatchRunError(f"copied artifact failed post-copy verification: {target}")


def _artifact_records_for_publish(
    staged: Path, destination: Path, batch_dir: Path
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(candidate for candidate in staged.rglob("*") if candidate.is_file()):
        final = destination / path.relative_to(staged)
        records.append(
            {
                "path": final.relative_to(batch_dir).as_posix(),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )
    return records


def import_alignment(
    batch_dir: str | os.PathLike[str],
    *,
    source_dir: str | os.PathLike[str],
    receipt_path: str | os.PathLike[str],
    task: str | None = None,
    alignment_id: str | None = None,
    source_task_prefix: str | None = None,
) -> dict[str, Any]:
    """Verify and copy one initial alignment into a never-run batch owner."""
    root = Path(batch_dir).expanduser().resolve()
    source_root = Path(source_dir).expanduser()
    receipt_file = Path(receipt_path).expanduser()

    with _batch_lock(root):
        manifest = load_manifest(root)
        # This operation executes no scientific code.  It must remain usable by
        # a newer checkout when recovering a batch pinned to an older core.
        verify_batch_identity(root, manifest, compare_current_core=False)
        row = _select_alignment(manifest, task=task, alignment_id=alignment_id)
        config_path = _verify_frozen_row(root, row)
        config = _load_config(config_path)
        owner = _assert_fresh(root, row)

        receipt, receipt_sha256, receipt_bytes = _read_receipt(receipt_file)
        _verify_receipt_identity(root, manifest, row, receipt)
        verified = _verified_sources(source_root, receipt)
        producer_manifest, producer_manifest_record = _verified_producer_manifest(
            source_root, receipt
        )
        receipt_prefix = str(receipt["source_task_filename_prefix"])
        if source_task_prefix is not None and source_task_prefix != receipt_prefix:
            raise BatchRunError(
                "--source-task-prefix does not match the transfer receipt"
            )
        source_prefix = receipt_prefix
        destinations = _destination_paths(
            verified,
            source_prefix=source_prefix,
            destination_task=config.alignmentTask.task_name,
        )

        token = uuid.uuid4().hex
        staged_artifacts = owner / f".import-artifacts-{token}"
        staged_attempt = owner / "attempts" / f".import-attempt-{token}"
        published_artifacts = owner / "artifacts"
        published_attempt = owner / "attempts" / "0001"
        artifacts_published = attempt_published = False
        try:
            _copy_verified(destinations, staged_artifacts)
            staged_attempt.mkdir()
            shutil.copyfile(config_path, staged_attempt / "config.toml", follow_symlinks=False)
            shutil.copyfile(
                receipt_file,
                staged_attempt / "import-receipt.json",
                follow_symlinks=False,
            )
            shutil.copyfile(
                producer_manifest,
                staged_attempt / "producer-manifest.json",
                follow_symlinks=False,
            )
            receipt_snapshot = staged_attempt / "import-receipt.json"
            producer_snapshot = staged_attempt / "producer-manifest.json"
            if (
                receipt_snapshot.stat().st_size != receipt_bytes
                or sha256_file(receipt_snapshot) != receipt_sha256
            ):
                raise BatchRunError("import receipt changed while it was being snapshotted")
            if (
                producer_snapshot.stat().st_size != producer_manifest_record["bytes"]
                or sha256_file(producer_snapshot) != producer_manifest_record["sha256"]
            ):
                raise BatchRunError(
                    "producer manifest changed while it was being snapshotted"
                )

            complete = {
                "schema": 1,
                "kind": "alignment",
                "id": row["id"],
                "status": "success",
                "config_sha256": row["config_sha256"],
                "core_sha256": manifest["source"]["core_sha256"],
                "finished_at": _now(),
                "attempt": "0001",
                "artifacts": _artifact_records_for_publish(
                    staged_artifacts, published_artifacts, root
                ),
                "import_provenance": {
                    "method": "verified-copy-v1",
                    "receipt": {
                        "path": (
                            published_attempt / "import-receipt.json"
                        ).relative_to(root).as_posix(),
                        "sha256": receipt_sha256,
                        "bytes": receipt_bytes,
                    },
                    "destination": receipt["destination"],
                    "producer": receipt["producer"],
                    "producer_manifest": {
                        "path": (
                            published_attempt / "producer-manifest.json"
                        ).relative_to(root).as_posix(),
                        "sha256": producer_manifest_record["sha256"],
                        "bytes": producer_manifest_record["bytes"],
                    },
                    "source_task_filename_prefix": source_prefix,
                },
            }
            atomic_write_json(staged_attempt / "status.json", {
                "status": "success",
                "finished_at": complete["finished_at"],
                "method": "import-alignment",
            })
            atomic_write_json(staged_attempt / "complete.json", complete)

            os.replace(staged_artifacts, published_artifacts)
            artifacts_published = True
            os.replace(staged_attempt, published_attempt)
            attempt_published = True
            validated = validate_completion(
                root, published_attempt / "complete.json", config
            )
            if validated is None:
                raise BatchRunError("imported alignment failed completion validation")
            return validated
        except BaseException:
            shutil.rmtree(staged_artifacts, ignore_errors=True)
            shutil.rmtree(staged_attempt, ignore_errors=True)
            if attempt_published:
                shutil.rmtree(published_attempt, ignore_errors=True)
            if artifacts_published:
                shutil.rmtree(published_artifacts, ignore_errors=True)
            raise


__all__ = ["RECEIPT_KIND", "import_alignment", "validate_import_completion"]
