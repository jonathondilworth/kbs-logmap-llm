"""Execute a frozen LogMapLLM batch using ordinary local processes.

The runner intentionally owns very little policy: the planner has already
validated and frozen every config.  This module adds bounded concurrency,
fresh attempts, safe initial-alignment reuse, and completion records that are
only written after the declared artifacts have been checked.
"""

from __future__ import annotations

import csv
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.evaluation.contract import validate_evaluation_payload
from logmap_llm.experiments.plan import (
    atomic_write_json,
    directory_digest,
    load_manifest,
    load_toml,
    select_jobs,
    sha256_file,
    source_digest,
)


class BatchRunError(RuntimeError):
    """An execution or batch-integrity check failed."""


_TERMINAL = frozenset({"success", "degraded", "failed", "timed_out", "interrupted"})
_SUCCESS = frozenset({"success", "degraded"})
_STATUS_LOCK = threading.Lock()
_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESSES: set[subprocess.Popen[Any]] = set()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _resolve(batch_dir: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else batch_dir / path


def _internal_path(batch_dir: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    if path.is_absolute():
        raise BatchRunError(f"batch-owned path must be relative: {value}")
    resolved = (batch_dir / path).resolve()
    try:
        resolved.relative_to(batch_dir.resolve())
    except ValueError as exc:
        raise BatchRunError(f"batch-owned path escapes the batch directory: {value}") from exc
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"), parse_constant=reject_constant
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BatchRunError(f"cannot read valid JSON from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BatchRunError(f"expected a JSON object in {path}")
    return value


def _artifact_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": _relative(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _artifact_records(paths: Sequence[Path], root: Path) -> list[dict[str, Any]]:
    return [_artifact_record(path, root) for path in sorted(paths)]


def _valid_import_provenance(
    batch_dir: Path,
    attempt: Path,
    complete: dict[str, Any],
) -> bool:
    """Validate the immutable receipt referenced by an imported alignment."""
    if "import_provenance" not in complete:
        return True
    # Delayed to avoid making the ordinary runner depend on transfer tooling.
    from logmap_llm.experiments.import_alignment import validate_import_completion

    return validate_import_completion(batch_dir, attempt, complete)


def validate_completion(
    batch_dir: Path,
    complete_path: Path,
    config: LogMapLLMConfig | None = None,
) -> dict[str, Any] | None:
    """Return a trustworthy completion record, or ``None`` if it is stale."""
    if not complete_path.is_file():
        return None
    try:
        complete = _read_json(complete_path)
        kind = complete.get("kind")
        attempt = complete_path.parent
        owner = attempt.parent.parent
        if complete.get("schema") != 1 or kind not in {"job", "alignment"}:
            return None
        if owner.parent.name != ("jobs" if kind == "job" else "alignments"):
            return None
        if complete.get("id") != owner.name or complete.get("attempt") != attempt.name:
            return None
        if not isinstance(complete.get("config_sha256"), str) or not complete["config_sha256"]:
            return None
        if not isinstance(complete.get("core_sha256"), str) or not complete["core_sha256"]:
            return None
        if kind == "job":
            required_identity = ("task", "model", "condition", "condition_id", "execution_hash")
            if any(
                not isinstance(complete.get(key), str) or not complete[key]
                for key in required_identity
            ):
                return None
            if not isinstance(complete.get("repeat"), int):
                return None
        if complete.get("status") not in _SUCCESS:
            return None
        if not _valid_import_provenance(batch_dir, attempt, complete):
            return None
        if config is not None:
            snapshot = attempt / "config.toml"
            if (
                not snapshot.is_file()
                or sha256_file(snapshot) != complete["config_sha256"]
            ):
                return None
        artifacts = complete.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            return None
        names: set[str] = set()
        for artifact in artifacts:
            path = _internal_path(batch_dir, artifact["path"])
            if not path.is_file() or sha256_file(path) != artifact["sha256"]:
                return None
            if artifact.get("bytes") != path.stat().st_size:
                return None
            allowed_root = attempt if kind == "job" else owner / "artifacts"
            try:
                path.relative_to(allowed_root.resolve())
            except ValueError:
                return None
            if path.name in names:
                return None
            names.add(path.name)
        if complete.get("kind") == "job" and "run_result.json" not in names:
            return None
        if complete.get("kind") == "alignment":
            if not any(name.endswith("-logmap_mappings.txt") for name in names):
                return None
            if not any(name.endswith("-logmap_mappings.tsv") for name in names):
                return None
            if not any(name.endswith("-logmap_mappings_to_ask_oracle_user_llm.txt") for name in names):
                return None
        if complete.get("status") == "success" and "rag_fallback.json" in names:
            return None
        if kind == "job":
            result_path = next(
                _internal_path(batch_dir, item["path"])
                for item in artifacts
                if Path(item["path"]).name == "run_result.json"
            )
            result_status = _read_json(result_path).get("status")
            expected = "degraded" if complete.get("status") == "degraded" else "succeeded"
            if result_status != expected:
                return None
            result_payload = _read_json(result_path)
            if complete.get("status") == "degraded":
                degradation = result_payload.get("degradation")
                if not isinstance(degradation, dict) or not degradation:
                    return None
            if config is not None:
                declared = {
                    _internal_path(batch_dir, item["path"]).resolve()
                    for item in artifacts
                }
                required = _required_job_artifacts(config, attempt / "run-root")
                if any(path.resolve() not in declared for path in required):
                    return None
                if config.evaluation.evaluate:
                    evaluation = attempt / "run-root" / "logmapllm-outputs" / "evaluation_results.json"
                    validate_evaluation_payload(
                        _read_json(evaluation),
                        config.evaluation.metrics,
                        task_name=config.alignmentTask.task_name,
                    )
        if "rag_fallback.json" in names:
            fallback_path = next(
                _internal_path(batch_dir, item["path"])
                for item in artifacts
                if Path(item["path"]).name == "rag_fallback.json"
            )
            fallback = _read_json(fallback_path)
            if fallback.get("effective_mode") not in {"zero-shot", "query-level-degradation"}:
                return None
    except (BatchRunError, KeyError, TypeError, ValueError, OSError):
        return None
    return complete


def _input_records(row: dict[str, Any]) -> Iterator[dict[str, Any]]:
    records = row.get("input_fingerprints", [])
    if isinstance(records, dict):
        records = records.values()
    for record in records:
        if isinstance(record, dict) and record.get("path") and record.get("sha256"):
            yield record


def _verify_frozen_config(batch_dir: Path, row: dict[str, Any]) -> Path:
    config_path = _internal_path(batch_dir, row["config_path"])
    if not config_path.is_file():
        raise BatchRunError(f"missing frozen config for {row['id']}: {config_path}")
    expected_config_hash = row.get("config_sha256")
    if expected_config_hash and sha256_file(config_path) != expected_config_hash:
        raise BatchRunError(f"frozen config changed after generation: {config_path}")
    return config_path


def _verify_frozen_row(batch_dir: Path, row: dict[str, Any]) -> Path:
    config_path = _verify_frozen_config(batch_dir, row)
    for record in _input_records(row):
        path = _resolve(batch_dir, record["path"])
        if record.get("kind") == "file":
            if not path.is_file():
                raise BatchRunError(f"input disappeared after generation: {path}")
            digest, size = sha256_file(path), path.stat().st_size
            if digest != record["sha256"] or size != record.get("bytes"):
                raise BatchRunError(f"input changed after generation: {path}")
        elif record.get("kind") == "directory":
            if not path.is_dir():
                raise BatchRunError(f"input directory disappeared after generation: {path}")
            digest, size, files = directory_digest(path)
            if (
                digest != record["sha256"]
                or size != record.get("bytes")
                or files != record.get("files")
            ):
                raise BatchRunError(f"input directory changed after generation: {path}")
        else:
            raise BatchRunError(f"unsupported input fingerprint kind for {path}")
    return config_path


def _load_config(path: Path) -> LogMapLLMConfig:
    return LogMapLLMConfig.model_validate(load_toml(path))


def verify_batch_identity(
    batch_dir: Path, manifest: dict[str, Any], *, compare_current_core: bool
) -> str:
    """Bind the frozen spec and, before execution, the currently loaded source."""
    if batch_dir.name != manifest["batch"]["id"]:
        raise BatchRunError(
            f"batch directory name does not match manifest ID {manifest['batch']['id']!r}"
        )
    spec = batch_dir / "batch.toml"
    if not spec.is_file() or sha256_file(spec) != manifest["source"]["spec_sha256"]:
        raise BatchRunError("frozen batch.toml changed after generation")
    current = source_digest() if compare_current_core else manifest["source"]["core_sha256"]
    if compare_current_core and current != manifest["source"]["core_sha256"]:
        raise BatchRunError(
            "LogMapLLM source changed after this batch was generated; regenerate the "
            "batch before running more jobs"
        )
    return current


def _check_secret_reference(cfg: LogMapLLMConfig) -> None:
    value = cfg.oracle.api_key
    if value == "EMPTY":
        return
    if not value.startswith("ENV:"):
        raise BatchRunError(
            "batch configs may not contain a literal oracle.api_key; use EMPTY or ENV:VARIABLE"
        )
    if cfg.pipeline.consult_oracle.value != "consult":
        return
    variable = value[4:]
    if not os.environ.get(variable):
        raise BatchRunError(
            f"oracle.api_key refers to unset or empty environment variable {variable!r}"
        )


def _check_endpoint(row: dict[str, Any], cfg: LogMapLLMConfig) -> None:
    if not row.get("verify_endpoint", False) or cfg.pipeline.consult_oracle.value != "consult":
        return
    if not cfg.oracle.base_url:
        raise BatchRunError(f"{row['id']}: endpoint verification requires oracle.base_url")
    expected = row.get("expected_served_model") or cfg.oracle.model_name
    url = cfg.oracle.base_url.rstrip("/") + "/models"
    headers = {"Accept": "application/json"}
    if cfg.oracle.api_key != "EMPTY":
        key = cfg.oracle.api_key
        if key.startswith("ENV:"):
            key = os.environ[key[4:]]
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.load(response)
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise BatchRunError(f"{row['id']}: endpoint check failed for {url}: {exc}") from exc
    models = payload.get("data", []) if isinstance(payload, dict) else []
    served = {item.get("id") for item in models if isinstance(item, dict)}
    if expected not in served:
        raise BatchRunError(
            f"{row['id']}: endpoint serves {sorted(str(v) for v in served if v)!r}, "
            f"not the requested model {expected!r}"
        )


def _next_attempt(
    parent: Path,
    resume: bool,
    config: LogMapLLMConfig | None = None,
) -> tuple[Path | None, dict[str, Any] | None]:
    attempts = parent / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in attempts.iterdir() if path.is_dir() and path.name.isdigit())
    successful = [
        (attempt, complete)
        for attempt in existing
        if (
            complete := validate_completion(
                parent.parents[1], attempt / "complete.json", config
            )
        )
        is not None
    ]
    if successful:
        if len(successful) != 1 or successful[0][0] != existing[-1]:
            raise BatchRunError(
                f"{parent.name} has contradictory attempts after a validated completion"
            )
        return None, successful[0][1]
    if existing and not resume:
        raise BatchRunError(
            f"{parent.name} has an incomplete or failed attempt; use --resume to create a fresh one"
        )
    number = (max((int(path.name) for path in existing), default=0) + 1)
    path = attempts / f"{number:04d}"
    path.mkdir()
    return path, None


@contextmanager
def _batch_lock(batch_dir: Path) -> Iterator[None]:
    lock_path = batch_dir / ".run.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise BatchRunError(f"another runner currently owns {batch_dir}") from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_status(attempt: Path, **values: Any) -> None:
    with _STATUS_LOCK:
        current: dict[str, Any] = {}
        path = attempt / "status.json"
        if path.exists():
            try:
                current = _read_json(path)
            except BatchRunError:
                current = {}
        current.update(values)
        atomic_write_json(path, current)


def _child_environment() -> dict[str, str]:
    env = os.environ.copy()
    # parents[2] is the checkout root that puts `logmap_llm` on the child's sys.path;
    # the check below ensures children import the same source verify_batch_identity verified.
    source_root = Path(__file__).resolve().parents[2]
    if not (source_root / "logmap_llm" / "__init__.py").is_file():
        raise BatchRunError(
            f"cannot locate the logmap_llm package under {source_root}; refusing to spawn "
            "child pipelines that might import an unverified copy"
        )
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(source_root) if not current else str(source_root) + os.pathsep + current
    )
    return env


def _pipeline_command(config: Path, run_root: Path, *, reuse_align: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "logmap_llm",
        "--config",
        str(config),
        "--run-root",
        str(run_root),
    ]
    if reuse_align:
        command.append("--reuse-align")
    return command


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10) -> None:
    """Terminate the entire owned group, including descendants after leader exit."""
    process_group = process.pid
    try:
        os.killpg(process_group, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_seconds
    while _process_group_exists(process_group) and time.monotonic() < deadline:
        process.poll()  # reap the leader as soon as it exits
        time.sleep(0.05)
    if _process_group_exists(process_group):
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _execute(
    command: Sequence[str],
    attempt: Path,
    timeout_seconds: int,
    cancel_event: threading.Event | None = None,
) -> tuple[int, str]:
    log_path = attempt / "run.log"
    _write_status(
        attempt,
        status="running",
        started_at=_now(),
        command=[str(arg) for arg in command],
        timeout_seconds=timeout_seconds,
    )
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        with _PROCESS_LOCK:
            if cancel_event is not None and cancel_event.is_set():
                _write_status(attempt, status="interrupted", finished_at=_now())
                raise BatchRunError("batch execution was cancelled")
            process = subprocess.Popen(
                list(command),
                cwd=attempt,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=_child_environment(),
                start_new_session=True,
            )
            _ACTIVE_PROCESSES.add(process)
        try:
            return_code = process.wait(timeout=timeout_seconds)
            state = (
                "interrupted"
                if return_code != 0
                and cancel_event is not None
                and cancel_event.is_set()
                else "finished"
            )
        except subprocess.TimeoutExpired:
            _terminate_process_group(process)
            return_code = 124
            state = "timed_out"
        except BaseException:
            _terminate_process_group(process)
            _write_status(attempt, status="interrupted", finished_at=_now())
            raise
        finally:
            with _PROCESS_LOCK:
                _ACTIVE_PROCESSES.discard(process)
    _write_status(
        attempt,
        status=state,
        return_code=return_code,
        finished_at=_now(),
        duration_seconds=round(time.monotonic() - started, 3),
    )
    return return_code, state


def _stop_active_processes() -> None:
    with _PROCESS_LOCK:
        processes = list(_ACTIVE_PROCESSES)
    for process in processes:
        _terminate_process_group(process)


def _required_job_artifacts(cfg: LogMapLLMConfig, run_root: Path) -> list[Path]:
    task = cfg.alignmentTask.task_name
    prompt = cfg.prompts.cls_usr_prompt_template_name
    output = run_root / "logmapllm-outputs"
    required = [output / "run_result.json"]
    if cfg.pipeline.align_ontologies.value != "bypass":
        required.extend(
            [
                run_root / "logmap-initial-alignment" / f"{task}-logmap_mappings.txt",
                run_root
                / "logmap-initial-alignment"
                / f"{task}-logmap_mappings_to_ask_oracle_user_llm.txt",
            ]
        )
    if cfg.pipeline.build_oracle_prompts.value != "bypass":
        required.append(output / f"{task}-{prompt}-mappings_to_ask_oracle_user_prompts.json")
        if cfg.few_shot.few_shot_k > 0:
            required.append(output / f"{task}-{prompt}-few_shot_examples.json")
    if cfg.pipeline.consult_oracle.value in {"consult", "reuse"}:
        required.append(output / f"{task}-{prompt}-mappings_to_ask_with_oracle_predictions.csv")
    # a run stopped after consultation (pipeline.stop_after_consultation) publishes neither
    stopped = cfg.pipeline.stop_after_consultation
    if cfg.pipeline.refine_alignment.value != "bypass" and not stopped:
        required.append(run_root / "logmap-refined-alignment" / f"{task}-logmap_mappings.tsv")
    if cfg.evaluation.evaluate and not stopped:
        required.append(output / "evaluation_results.json")
    # The annotated M_ask files (pipeline/annotate.py, since 22 Sep 2026) are declared and
    # checksummed when present; batches completed under an older core have none, and
    # their completion records must stay valid.
    for supplemental in (
        "rag_fallback.json", "rag_negative_fallback.json", "rag_traces.json",
        f"{task}-{prompt}-annotated.txt", f"{task}-{prompt}-annotated.tsv",
    ):
        path = output / supplemental
        if path.is_file():
            required.append(path)
    return list(dict.fromkeys(required))


def _validate_artifacts(paths: Sequence[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise BatchRunError("pipeline exited without required artifact(s): " + ", ".join(missing))
    for path in paths:
        if path.suffix == ".json":
            payload = _read_json(path)
            if path.name == "run_result.json" and payload.get("status") not in {
                "succeeded",
                "degraded",
            }:
                raise BatchRunError(f"pipeline result does not report success: {path}")
        elif path.suffix == ".csv":
            with path.open(newline="", encoding="utf-8") as stream:
                if next(csv.reader(stream), None) is None:
                    raise BatchRunError(f"CSV artifact has no header: {path}")


def _is_degraded(run_root: Path, cfg: LogMapLLMConfig) -> bool:
    fallback = run_root / "logmapllm-outputs" / "rag_fallback.json"
    if fallback.is_file():
        if cfg.few_shot.rag_failure_policy != "record_zero_shot":
            raise BatchRunError(
                "RAG fallback was published although rag_failure_policy is not record_zero_shot"
            )
        payload = _read_json(fallback)
        if payload.get("effective_mode") not in {"zero-shot", "query-level-degradation"}:
            raise BatchRunError(f"RAG fallback has no recognised effective condition: {fallback}")
    result = _read_json(run_root / "logmapllm-outputs" / "run_result.json")
    degraded = result.get("status") == "degraded"
    if fallback.is_file() and not degraded:
        raise BatchRunError("RAG fallback exists but run_result.json is not degraded")
    return degraded


def _publish_alignment(attempt: Path, cfg: LogMapLLMConfig, owner: Path) -> list[Path]:
    run_root = attempt / "run-root"
    task = cfg.alignmentTask.task_name
    initial = run_root / "logmap-initial-alignment"
    required = [
        initial / f"{task}-logmap_mappings.txt",
        initial / f"{task}-logmap_mappings.tsv",
        initial / f"{task}-logmap_mappings_to_ask_oracle_user_llm.txt",
    ]
    _validate_artifacts(required + [run_root / "logmapllm-outputs" / "run_result.json"])

    staged = owner / f".artifacts-{uuid.uuid4().hex}"
    shutil.copytree(initial, staged)
    destination = owner / "artifacts"
    if destination.exists():
        orphan = owner / f"orphaned-artifacts-{uuid.uuid4().hex}"
        os.replace(destination, orphan)
    os.replace(staged, destination)
    return sorted(path for path in destination.rglob("*") if path.is_file())


def _run_alignment(batch_dir: Path, row: dict[str, Any], resume: bool) -> dict[str, Any]:
    owner = batch_dir / "alignments" / row["id"]
    owner.mkdir(parents=True, exist_ok=True)
    config_path = _verify_frozen_row(batch_dir, row)
    cfg = _load_config(config_path)
    attempt, prior = _next_attempt(owner, resume, cfg)
    if prior is not None:
        if (
            prior.get("id") != row["id"]
            or prior.get("config_sha256") != row.get("config_sha256")
            or prior.get("core_sha256") != row.get("core_sha256")
        ):
            raise BatchRunError(f"alignment completion identity does not match {row['id']}")
        return prior
    assert attempt is not None
    _check_secret_reference(cfg)
    snapshot = attempt / "config.toml"
    shutil.copy2(config_path, snapshot)
    timeout = int(row.get("timeout_seconds") or 21600)
    return_code, state = _execute(
        _pipeline_command(snapshot, attempt / "run-root"), attempt, timeout
    )
    if return_code != 0:
        status = "timed_out" if state == "timed_out" else "failed"
        _write_status(attempt, status=status)
        raise BatchRunError(f"alignment {row['id']} failed; see {attempt / 'run.log'}")
    try:
        published = _publish_alignment(attempt, cfg, owner)
        complete = {
            "schema": 1,
            "kind": "alignment",
            "id": row["id"],
            "status": "success",
            "config_sha256": row.get("config_sha256"),
            "core_sha256": row["core_sha256"],
            "finished_at": _now(),
            "attempt": attempt.name,
            "artifacts": _artifact_records(published, batch_dir),
        }
        atomic_write_json(attempt / "complete.json", complete)
        _write_status(attempt, status="success")
        return complete
    except Exception:
        _write_status(attempt, status="failed", finished_at=_now())
        raise


def _copy_alignment(batch_dir: Path, alignment_id: str, destination: Path) -> None:
    owner = batch_dir / "alignments" / alignment_id
    complete_paths = sorted(owner.glob("attempts/*/complete.json"), reverse=True)
    complete = (
        validate_completion(batch_dir, complete_paths[0]) if complete_paths else None
    )
    if complete is None:
        raise BatchRunError(f"alignment prerequisite {alignment_id} is not complete")
    source = owner / "artifacts"
    if not source.is_dir():
        raise BatchRunError(f"alignment prerequisite {alignment_id} has no published artifacts")
    shutil.copytree(source, destination)


def _run_job(
    batch_dir: Path,
    row: dict[str, Any],
    resume: bool,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    owner = batch_dir / "jobs" / row["id"]
    owner.mkdir(parents=True, exist_ok=True)
    config_path = _verify_frozen_row(batch_dir, row)
    cfg = _load_config(config_path)
    attempt, prior = _next_attempt(owner, resume, cfg)
    if prior is not None:
        if (
            prior.get("id") != row["id"]
            or prior.get("execution_hash") != row["execution_hash"]
            or prior.get("config_sha256") != row.get("config_sha256")
            or prior.get("core_sha256") != row.get("core_sha256")
        ):
            raise BatchRunError(f"job completion identity does not match {row['id']}")
        return prior
    assert attempt is not None
    _check_secret_reference(cfg)
    _check_endpoint(row, cfg)
    snapshot = attempt / "config.toml"
    shutil.copy2(config_path, snapshot)
    run_root = attempt / "run-root"
    alignment_id = row.get("alignment_id")
    if alignment_id:
        _copy_alignment(
            batch_dir, str(alignment_id), run_root / "logmap-initial-alignment"
        )
    timeout = int(row.get("timeout_seconds") or 21600)
    return_code, state = _execute(
        _pipeline_command(snapshot, run_root, reuse_align=bool(alignment_id)),
        attempt,
        timeout,
        cancel_event,
    )
    if return_code != 0:
        status = state if state in {"timed_out", "interrupted"} else "failed"
        _write_status(attempt, status=status)
        raise BatchRunError(f"job {row['id']} failed; see {attempt / 'run.log'}")
    try:
        artifacts = _required_job_artifacts(cfg, run_root)
        _validate_artifacts(artifacts)
        if cfg.evaluation.evaluate:
            validate_evaluation_payload(
                _read_json(run_root / "logmapllm-outputs" / "evaluation_results.json"),
                cfg.evaluation.metrics,
                task_name=cfg.alignmentTask.task_name,
            )
        status = "degraded" if _is_degraded(run_root, cfg) else "success"
        complete = {
            "schema": 1,
            "kind": "job",
            "id": row["id"],
            "task": row["task"],
            "model": row["model"],
            "condition": row["condition"],
            "repeat": row["repeat"],
            "condition_id": row["condition_id"],
            "execution_hash": row["execution_hash"],
            "alignment_id": alignment_id,
            "status": status,
            "config_sha256": row.get("config_sha256"),
            "core_sha256": row["core_sha256"],
            "finished_at": _now(),
            "attempt": attempt.name,
            "artifacts": _artifact_records(artifacts, batch_dir),
        }
        atomic_write_json(attempt / "complete.json", complete)
        _write_status(attempt, status=status)
        return complete
    except Exception:
        _write_status(attempt, status="failed", finished_at=_now())
        raise


def _alignment_rows(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("alignments", [])
    if isinstance(rows, dict):
        rows = [dict(value, id=key) for key, value in rows.items()]
    return {str(row["id"]): row for row in rows}


def _model_limits(rows: Sequence[dict[str, Any]]) -> dict[str, threading.Semaphore]:
    limits: dict[str, int] = {}
    for row in rows:
        value = row.get("max_parallel_runs")
        if value is not None:
            limits[row["model"]] = min(limits.get(row["model"], int(value)), int(value))
    return {model: threading.Semaphore(value) for model, value in limits.items()}


def _fair_job_order(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave models so per-model semaphores do not strand all worker threads."""
    queues: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        queues.setdefault(row["model"], []).append(row)
    ordered: list[dict[str, Any]] = []
    while any(queues.values()):
        for model in sorted(queues):
            if queues[model]:
                ordered.append(queues[model].pop(0))
    return ordered


def _run_with_limit(
    batch_dir: Path,
    row: dict[str, Any],
    resume: bool,
    limits: dict[str, threading.Semaphore],
    cancel_event: threading.Event,
) -> dict[str, Any]:
    semaphore = limits.get(row["model"])
    if semaphore is None:
        if cancel_event.is_set():
            raise BatchRunError("batch execution was cancelled")
        return _run_job(batch_dir, row, resume, cancel_event)
    while not semaphore.acquire(timeout=0.1):
        if cancel_event.is_set():
            raise BatchRunError("batch execution was cancelled")
    try:
        if cancel_event.is_set():
            raise BatchRunError("batch execution was cancelled")
        return _run_job(batch_dir, row, resume, cancel_event)
    finally:
        semaphore.release()


def _record_latest_failure(owner: Path, error: BaseException) -> None:
    """Ensure preflight failures are visible to status and aggregation."""
    attempts = owner / "attempts"
    if not attempts.is_dir():
        return
    directories = sorted(
        (path for path in attempts.iterdir() if path.is_dir() and path.name.isdigit()),
        reverse=True,
    )
    if directories and not (directories[0] / "complete.json").exists():
        attempt = directories[0]
        current_status = ""
        try:
            current_status = str(
                _read_json(attempt / "status.json").get("status", "")
            )
        except BatchRunError:
            pass
        values = {"error": str(error), "finished_at": _now()}
        if current_status not in _TERMINAL - _SUCCESS:
            values["status"] = "failed"
        _write_status(attempt, **values)


def run_batch(
    batch_dir: str | os.PathLike[str],
    *,
    jobs_override: int | None = None,
    resume: bool = False,
    selectors: Sequence[str] = (),
    limit: int | None = None,
) -> dict[str, Any]:
    """Run selected jobs and return a concise status summary."""
    root = Path(batch_dir).resolve()
    manifest = load_manifest(root)
    core_digest = verify_batch_identity(root, manifest, compare_current_core=True)
    rows = [
        dict(row, core_sha256=core_digest)
        for row in select_jobs(manifest["jobs"], selectors, limit=limit)
    ]
    if not rows:
        raise BatchRunError("selection matched no jobs")
    configured_jobs = int(manifest.get("batch", {}).get("jobs", 1))
    workers = configured_jobs if jobs_override is None else jobs_override
    if workers < 1:
        raise BatchRunError("--jobs must be at least 1")

    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with _batch_lock(root):
        alignments = {
            key: dict(value, core_sha256=core_digest)
            for key, value in _alignment_rows(manifest).items()
        }
        for alignment_id in sorted({row.get("alignment_id") for row in rows if row.get("alignment_id")}):
            if alignment_id not in alignments:
                raise BatchRunError(f"manifest does not define alignment {alignment_id}")
            try:
                _run_alignment(root, alignments[alignment_id], resume)
            except Exception as exc:
                _record_latest_failure(root / "alignments" / str(alignment_id), exc)
                raise

        model_limits = _model_limits(rows)
        scheduled_rows = _fair_job_order(rows)
        cancel_event = threading.Event()
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="logmap-llm") as pool:
            futures: dict[Future[dict[str, Any]], dict[str, Any]] = {
                pool.submit(
                    _run_with_limit,
                    root,
                    row,
                    resume,
                    model_limits,
                    cancel_event,
                ): row
                for row in scheduled_rows
            }
            try:
                for future in as_completed(futures):
                    row = futures[future]
                    try:
                        results[row["id"]] = future.result()
                    except Exception as exc:  # every other job still gets a terminal record
                        _record_latest_failure(root / "jobs" / row["id"], exc)
                        failures[row["id"]] = str(exc)
            except KeyboardInterrupt as exc:
                cancel_event.set()
                for future in futures:
                    future.cancel()
                _stop_active_processes()
                raise BatchRunError("batch interrupted; use --resume for fresh attempts") from exc

    summary = {
        "selected": len(rows),
        "success": sum(item["status"] == "success" for item in results.values()),
        "degraded": sum(item["status"] == "degraded" for item in results.values()),
        "failed": len(failures),
        "failures": failures,
    }
    if failures:
        details = "; ".join(f"{job}: {reason}" for job, reason in sorted(failures.items()))
        raise BatchRunError(f"{len(failures)} job(s) failed: {details}")
    return summary


def _latest_state(batch_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    try:
        config_path = _verify_frozen_config(batch_dir, row)
        config = _load_config(config_path)
    except (BatchRunError, ValueError) as exc:
        return {
            "id": row["id"],
            "status": "invalid_plan",
            "attempt": "",
            "error": str(exc),
        }
    attempts = batch_dir / "jobs" / row["id"] / "attempts"
    if not attempts.is_dir():
        return {"id": row["id"], "status": "not_run", "attempt": ""}
    for attempt in sorted(
        (
            path
            for path in attempts.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        reverse=True,
    ):
        complete_path = attempt / "complete.json"
        complete = validate_completion(batch_dir, complete_path, config)
        if complete:
            expected = {
                key: row.get(key)
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
            mismatched = [
                key for key, value in expected.items() if complete.get(key) != value
            ]
            if mismatched:
                return {
                    "id": row["id"],
                    "status": "invalid_artifacts",
                    "attempt": attempt.name,
                    "error": "completion identity mismatch: " + ", ".join(mismatched),
                }
            return {"id": row["id"], "status": complete["status"], "attempt": attempt.name}
        if complete_path.exists():
            return {
                "id": row["id"],
                "status": "invalid_artifacts",
                "attempt": attempt.name,
                "error": "completion record or declared artifacts are invalid",
            }
        status_path = attempt / "status.json"
        if status_path.exists():
            try:
                status = _read_json(status_path).get("status", "incomplete")
            except BatchRunError as exc:
                return {
                    "id": row["id"],
                    "status": "incomplete",
                    "attempt": attempt.name,
                    "error": str(exc),
                }
            if status in {"finished", "success", "degraded"}:
                return {
                    "id": row["id"],
                    "status": "invalid_artifacts",
                    "attempt": attempt.name,
                    "error": "completion record or declared artifacts are invalid",
                }
            return {"id": row["id"], "status": status, "attempt": attempt.name}
        return {"id": row["id"], "status": "incomplete", "attempt": attempt.name}
    return {"id": row["id"], "status": "not_run", "attempt": ""}


def status_rows(
    batch_dir: str | os.PathLike[str], selectors: Sequence[str] = ()
) -> list[dict[str, Any]]:
    root = Path(batch_dir).resolve()
    manifest = load_manifest(root)
    verify_batch_identity(root, manifest, compare_current_core=False)
    expected_core = manifest["source"]["core_sha256"]
    rows = [
        dict(row, core_sha256=expected_core)
        for row in select_jobs(manifest["jobs"], selectors)
    ]
    return [dict(row, **_latest_state(root, row)) for row in rows]


def print_status(
    batch_dir: str | os.PathLike[str], selectors: Sequence[str] = ()
) -> list[dict[str, Any]]:
    rows = status_rows(batch_dir, selectors)
    print("JOB\tTASK\tMODEL\tCONDITION\tREPEAT\tSTATUS\tATTEMPT")
    for row in rows:
        print(
            f"{row['id']}\t{row['task']}\t{row['model']}\t{row['condition']}\t"
            f"{row['repeat']}\t{row['status']}\t{row['attempt']}"
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("\n" + "  ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
    return rows
