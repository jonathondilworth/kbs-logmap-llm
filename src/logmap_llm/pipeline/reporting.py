"""
logmap_llm.pipeline.reporting
"""
from __future__ import annotations

import hashlib
import json
import math
import numbers
import subprocess
from enum import Enum
from pathlib import Path

from logmap_llm.utils.logging import step
from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.pipeline.contracts import (
    TimingRecord,
    EvaluationResult,
    OracleResult,
    PromptBuildResult,
    ModelSelectionResult,
)
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.utils.io import atomic_json_write_strict


_REPORTABLE_ORACLE_FIELDS = frozenset({
    "model_name",
    "interaction_style",
    "temperature",
    "top_p",
    "max_completion_tokens",
    "reasoning_effort",
    "reasoning_token_budget",
    "enable_thinking",
    "max_workers",
    "answer_format",
    "response_mode",
    "request_timeout_seconds",
    "connect_timeout_seconds",
    "transient_retries",
    "seed",
    "request_logprobs",
    "openrouter_provider",
    "openrouter_allow_fallbacks",
    "openrouter_require_parameters",
    "logprobs_requested",
    "logprobs_effective",
    "capability_downgrades",
})


def format_duration(seconds: float | None) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds is None:
        return "N/A"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.1f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


def format_timing_value(val: float | str | None) -> str:
    """Format a timing value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, str):
        return val
    return format_duration(val)


def classify_endpoint(base_url: str | None) -> str:
    """Classify an LLM endpoint based on its base URL."""
    if base_url is None:
        return "OpenRouter (default)"
    url = base_url.lower()
    if "openrouter" in url:
        return "OpenRouter"
    if "localhost" in url or "127.0.0.1" in url:
        return "Local (vLLM/SGLang)"
    from urllib.parse import urlsplit
    parsed = urlsplit(base_url)
    host = parsed.hostname or "unknown-host"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return f"Custom ({host})"


def _json_safe(value):
    """Convert known scalar/container values without stringifying arbitrary objects."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Enum):
        return _json_safe(value.value)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Unsupported result value: {type(value).__name__}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(role: str, path: Path, run_paths: PipelinePaths) -> dict | None:
    if not path.is_file():
        return None
    try:
        rendered_path = str(path.resolve().relative_to(run_paths.run_root))
    except (TypeError, ValueError):
        rendered_path = str(path.resolve())
    return {
        "role": role,
        "path": rendered_path,
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _rag_degradation(run_paths: PipelinePaths) -> dict | None:
    """Return compact fallback provenance without copying query IRIs into the result."""
    path = run_paths.output_dir / "rag_fallback.json"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        fallback = json.load(stream)
    if not isinstance(fallback, dict):
        raise ValueError(f"RAG fallback artifact must contain a JSON object: {path}")
    summary = {
        key: _json_safe(fallback[key])
        for key in (
            "requested_few_shot_k",
            "requested_negative_strategy",
            "requested_encoder_kind",
            "effective_mode",
            "reason_type",
            "reason",
        )
        if key in fallback
    }
    affected = fallback.get("affected_queries")
    if isinstance(affected, dict):
        summary["affected_queries"] = len(affected)
    return summary


def validate_run_artifacts(
    cfg: LogMapLLMConfig, run_paths: PipelinePaths, stopped_after: str | None = None,
) -> None:
    """Raise when a selected stage did not publish its required artifact. A run stopped
    after consultation (`pipeline.stop_after_consultation`) owes no refined alignment and
    no evaluation, but does owe the annotated M_ask files."""
    required: list[tuple[str, Path]] = []
    if cfg.pipeline.align_ontologies.value != "bypass":
        required.extend([
            ("initial_alignment", run_paths.logmap_mappings()),
            ("m_ask", run_paths.logmap_m_ask()),
        ])
    if cfg.pipeline.build_oracle_prompts.value != "bypass":
        required.append(("prompts", run_paths.prompts_json()))
        if cfg.few_shot.few_shot_k > 0:
            required.append(("few_shot", run_paths.few_shot_json()))
    if cfg.automatic_model_selection:
        required.append(("model_selection", run_paths.model_selection_json()))
    if cfg.pipeline.consult_oracle.value in {"consult", "reuse"}:
        required.append(("predictions", run_paths.predictions_csv()))
        required.append(("annotated_txt", run_paths.annotated_txt()))
        required.append(("annotated_tsv", run_paths.annotated_tsv()))
    stopped = stopped_after == "consultation" or cfg.pipeline.stop_after_consultation
    if cfg.pipeline.refine_alignment.value == "refine" and not stopped:
        required.append(("refined_alignment", run_paths.refined_mappings_tsv()))
    if cfg.evaluation.evaluate and not stopped:
        required.append(("evaluation", run_paths.eval_json()))

    missing = [f"{role}: {path}" for role, path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Selected pipeline stages did not publish required artifacts:\n  - "
            + "\n  - ".join(missing)
        )


def get_gpu_info() -> str:
    """Attempt to get GPU information via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "N/A"


def print_timing_summary(timing: TimingRecord, n_consultations: int) -> None:
    """Print a human-readable timing summary."""
    step("[Step 6] Timing Summary")
    print()
    print(f"  Alignment          : {format_duration(timing.align_seconds)}")
    print(f"  Prompt building    : {format_duration(timing.prompt_build_seconds)}")
    if timing.model_selection_seconds is not None:
        print(f"  Model selection    : {format_duration(timing.model_selection_seconds)}")
    print(f"  Oracle consultation: {format_duration(timing.consult_seconds)}")
    if n_consultations > 0 and timing.consult_seconds:
        per_consult = timing.consult_seconds / n_consultations
        print(f"    Per consultation : {format_duration(per_consult)}")
    print(f"  Refinement         : {format_duration(timing.refine_seconds)}")
    print(f"  Evaluation         : {format_duration(timing.evaluate_seconds)}")
    print(f"  Total              : {format_duration(timing.total_seconds)}")
    print()


def print_experimental_parameters(
    cfg: LogMapLLMConfig,
    oracle_result: OracleResult,
    prompt_result: PromptBuildResult,
    timing: TimingRecord,
    model_selection: ModelSelectionResult | None = None,
) -> None:
    """Print experimental parameters."""
    step("[Step 7] Experimental Parameters")
    print()
    print(f"  Task name           : {cfg.alignmentTask.task_name}")
    selected = (
        f" (auto-selected from {len(model_selection.ranking)} candidates)"
        if model_selection is not None and model_selection.performed else ""
    )
    print(f"  Model               : {cfg.oracle.model_name}{selected}")
    print(f"  Endpoint            : {classify_endpoint(cfg.oracle.base_url)}")
    print(f"  Prompt template     : {cfg.prompts.cls_usr_prompt_template_name}")
    print(f"  Developer prompt    : {cfg.prompts.cls_dev_prompt_template_name}")
    print(f"  N prompts           : {prompt_result.n_prompts}")
    if oracle_result.oracle_params:
        params = oracle_result.oracle_params
        print(f"  Interaction style   : {params.get('interaction_style', 'N/A')}")
        print(f"  Temperature         : {params.get('temperature', 'N/A')}")
        print(f"  Top-p               : {params.get('top_p', 'N/A')}")
        print(f"  Max workers         : {params.get('max_workers', 'N/A')}")
    print(f"  GPU                 : {get_gpu_info()}")
    print()


def write_results_file(
    filepath: Path,
    cfg: LogMapLLMConfig,
    timing: TimingRecord,
    eval_result: EvaluationResult,
    oracle_result: OracleResult,
    prompt_result: PromptBuildResult,
    run_paths: PipelinePaths,
    stopped_after: str | None = None,
    model_selection: ModelSelectionResult | None = None,
) -> None:
    """Atomically publish the compact, redacted canonical run result."""
    validate_run_artifacts(cfg, run_paths, stopped_after=stopped_after)
    safe_oracle_params = {
        key: _json_safe(value)
        for key, value in oracle_result.oracle_params.items()
        if key in _REPORTABLE_ORACLE_FIELDS
    }
    predictions = oracle_result.predictions
    prediction_counts = {"candidates": 0, "true": 0, "false": 0, "error": 0, "skipped": 0}
    if predictions is not None and "Oracle_prediction" in predictions:
        counts = predictions["Oracle_prediction"].value_counts()
        prediction_counts = {
            "candidates": len(predictions),
            "true": int(counts.get(True, 0)),
            "false": int(counts.get(False, 0)),
            "error": int(counts.get("error", 0)),
            "skipped": int(counts.get("skipped", 0)),
        }

    if oracle_result.local_oracle_verdicts is not None:
        prediction_counts["local_oracle"] = {
            key: int(value) for key, value in oracle_result.local_oracle_verdicts.items()
        }

    degradation: dict[str, dict] = {}
    rag_degradation = _rag_degradation(run_paths)
    if rag_degradation is not None:
        degradation["rag"] = rag_degradation
    if prediction_counts["error"] or prediction_counts["skipped"]:
        degradation["oracle_coverage"] = {
            "errors": prediction_counts["error"],
            "skipped": prediction_counts["skipped"],
            "usable": prediction_counts["true"] + prediction_counts["false"],
            "candidates": prediction_counts["candidates"],
        }
    artifact_paths = [
        ("initial_alignment", run_paths.logmap_mappings()),
        ("initial_alignment_tsv", run_paths.logmap_mappings_tsv()),
        ("m_ask", run_paths.logmap_m_ask()),
        ("prompts", run_paths.prompts_json()),
        ("few_shot", run_paths.few_shot_json()),
        ("predictions", run_paths.predictions_csv()),
        ("annotated_txt", run_paths.annotated_txt()),
        ("annotated_tsv", run_paths.annotated_tsv()),
        ("refined_alignment", run_paths.refined_mappings_tsv()),
        ("evaluation", run_paths.eval_json()),
        ("rag_fallback", run_paths.output_dir / "rag_fallback.json"),
        # Negative-construction fallbacks are distinct from retrieval-mode degradation:
        # the block is complete and the query is not degraded. Registering the artifact
        # here checksums the per-lane fallback rate into the run result.
        ("rag_negative_fallback", run_paths.output_dir / "rag_negative_fallback.json"),
        ("rag_traces", run_paths.output_dir / "rag_traces.json"),
        ("model_ranking_prompts", run_paths.model_ranking_json()),
        ("model_selection", run_paths.model_selection_json()),
    ]
    artifacts = [
        record
        for role, path in artifact_paths
        if (record := _artifact_record(role, path, run_paths)) is not None
    ]

    results = {
        "schema_version": 1,
        "status": "degraded" if degradation else "succeeded",
        "task_name": cfg.alignmentTask.task_name,
        "model_name": cfg.oracle.model_name,
        "requested": {
            "prompt_template": cfg.prompts.cls_usr_prompt_template_name,
            "few_shot_k": cfg.few_shot.few_shot_k,
            "few_shot_strategy": cfg.few_shot.few_shot_negative_strategy,
            "rag_encoder_kind": cfg.few_shot.rag_encoder_kind,
            "rag_encoder_revision": cfg.few_shot.rag_encoder_revision,
            "rag_encoder_max_length": cfg.few_shot.rag_encoder_max_length,
        },
        "oracle": safe_oracle_params,
        "counts": {
            "prompts": prompt_result.n_prompts,
            **prediction_counts,
        },
        "timing": {
            "align_seconds": timing.align_seconds,
            "prompt_build_seconds": timing.prompt_build_seconds,
            "model_selection_seconds": timing.model_selection_seconds,
            "consult_seconds": timing.consult_seconds,
            "refine_seconds": timing.refine_seconds,
            "evaluate_seconds": timing.evaluate_seconds,
            "total_seconds": timing.total_seconds,
        },
        "evaluation": _json_safe(
            eval_result.results if eval_result.results else eval_result.metrics
        ),
        "degradation": degradation or None,
        "stopped_after": stopped_after,
        "artifacts": artifacts,
    }
    if cfg.automatic_model_selection:
        performed = model_selection is not None and model_selection.performed
        results["model_selection"] = {
            "status": "selected" if performed else "skipped",
            "questions": model_selection.questions if performed else 0,
            "ranking": model_selection.ranking if performed else [],
            "selected": model_selection.selected if performed else None,
        }

    atomic_json_write_strict(filepath, _json_safe(results), indent=2)

    step(f"[Step 8] Results written to: {filepath}")
