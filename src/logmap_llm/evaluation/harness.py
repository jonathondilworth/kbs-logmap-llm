"""
logmap_llm.evaluation.harness
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from logmap_llm.constants import PRIMARY_EVALUATION_ENGINES
from logmap_llm.evaluation.engines import (
    CustomEvaluationEngine,
    DeepOntoEvaluationEngine,
    EvaluationEngine,
    PartialReferenceEvaluationEngine,
    build_engine,
)
from logmap_llm.evaluation.engines.deeponto import _ensure_jvm_memory
from logmap_llm.evaluation.io import (
    load_mapping_pairs,
    load_oracle_predictions,
)
from logmap_llm.utils.logging import (
    metric,
    warning,
)
from logmap_llm.utils.io import atomic_json_write_strict


def select_engine(
    partial_reference: bool = False,
    force_custom: bool = False,
    jvm_memory: str = "8g",
    primary: str | None = None,
) -> EvaluationEngine:
    """
    Select the engine of the plain `global` block. Without `primary` (the historical
    path, unchanged) the [evaluation] config flags decide: partial_reference (KG-like
    tasks) => PartialReference; force_custom => Custom; else DeepOnto when available,
    falling back to Custom. With `primary` (the first entry of `evaluation.engines`)
    that engine is used: 'partial_reference', 'custom', or 'deeponto' (DeepOnto when
    importable, else Custom with a warning). The subsumption-based ranking path is not
    yet implemented.
    """
    if primary is not None:
        if primary not in PRIMARY_EVALUATION_ENGINES:
            raise ValueError(
                f"primary evaluation engine must be one of {list(PRIMARY_EVALUATION_ENGINES)}, "
                f"got {primary!r}"
            )
        if primary == "partial_reference":
            return PartialReferenceEvaluationEngine()
        if primary == "custom":
            return CustomEvaluationEngine()
        _ensure_jvm_memory(jvm_memory, overwrite=True)
        if DeepOntoEvaluationEngine.is_available():
            return DeepOntoEvaluationEngine()
        warning("evaluation.engines requested 'deeponto' but DeepOnto is not importable; "
                "the plain block falls back to the custom engine")
        return CustomEvaluationEngine()
    if partial_reference:
        return PartialReferenceEvaluationEngine()
    if force_custom:
        return CustomEvaluationEngine()
    _ensure_jvm_memory(jvm_memory, overwrite=True)
    if DeepOntoEvaluationEngine.is_available():
        return DeepOntoEvaluationEngine()
    return CustomEvaluationEngine()


###
# PRINTING HELPERS
###

_BREAKLINE_ON_ORACLE_METRIC_KEYS = frozenset({"Sensitivity"})

def _format_metric(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def _format_metric_key(metric_key: str) -> str:
    return (" ".join(metric_key.split("_"))).title()


def _print_global_metrics(metrics: dict) -> None:
    metric("Global Alignment Metrics:")
    metric(f"  running metrics evaluation backend: {metrics.get('source', 'unknown')})")
    metric("  ")
    equal_spacing = max((len(k) for k in metrics), default=0)
    for metric_key, metric_value in metrics.items():
        metric(f"   {_format_metric_key(metric_key):<{equal_spacing}}  :  {_format_metric(metric_value)}")


def _print_oracle_metrics(metrics: dict, ignore_keys: list[str] = ['false_mappings']) -> None:
    metric("Oracle Discrimination Metrics")
    metric(f"  running metrics evaluation backend: {metrics.get('source', 'unknown')})")
    metric("  ")
    reversed_metrics_dict: dict = dict(reversed(metrics.items())) # purely cosmetric
    equal_spacing = max((len(k) for k in reversed_metrics_dict), default=0)
    for metric_key, metric_value in reversed_metrics_dict.items():
        if metric_key not in ignore_keys:
            metric(f"   {_format_metric_key(metric_key):<{equal_spacing}}  :  {_format_metric(metric_value)}")
            if metric_key in _BREAKLINE_ON_ORACLE_METRIC_KEYS: print() # noqa


def _print_stratified_results(strat: dict, label: str) -> None:
    print(f"\n{label}:")
    for key, m in strat.items():
        extra = f", Ign={m['ignored']}" if "ignored" in m else ""
        metric(
            f"  {key:>15s}:  P={_format_metric(m['precision'])}  "
            f"R={_format_metric(m['recall'])}  F1={_format_metric(m['f1'])}  "
            f"(TP={m['true_positives']}, "
            f"FP={m['false_positives']}, "
            f"FN={m['false_negatives']}{extra})"
        )


###
# HELPERS FOR FILE OPERATIONS
###


_FALSE_MAPPING_FIELDS = [
    "source_entity_uri",
    "target_entity_uri",
    "oracle_prediction",
    "oracle_confidence",
    "in_reference",
    "error_type",
]


def _false_mappings_path(
    output_json_path: Path | None,
    oracle_predictions_path: Path,
) -> Path:
    if output_json_path is not None:
        return Path(output_json_path).parent / "false_mappings.csv"
    return Path(oracle_predictions_path).parent / "false_mappings.csv"


def _write_false_mappings_csv(false_mappings: list[dict], output_json_path: Path | None, oracle_predictions_path: Path) -> Path:
    """Atomically write a false-mappings CSV alongside evaluation results."""
    fm_path = _false_mappings_path(output_json_path, oracle_predictions_path)
    fm_path.parent.mkdir(parents=True, exist_ok=True)

    descriptor, temporary = tempfile.mkstemp(
        dir=fm_path.parent, prefix=f".{fm_path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as fout:
            writer = csv.DictWriter(fout, fieldnames=_FALSE_MAPPING_FIELDS)
            writer.writeheader()
            writer.writerows(false_mappings)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(temporary, fm_path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    return fm_path


###
# HELPERS FOR AUTOMATIC FILE DETECTION
# TODO: legacy, retire once the orchestration layer is fleshed out; handy for
# ad-hoc testing but coupled to our naming conventions; should use paths.py.
###

def _find_system_mappings(results_dir: Path, task_name: str) -> Path | None:
    refined_dir = Path(results_dir) / "logmap-refined-alignment"
    if not refined_dir.is_dir():
        return None
    tsv = refined_dir / f"{task_name}-logmap_mappings.tsv"
    if tsv.exists():
        return tsv
    for f in refined_dir.glob("*mappings.tsv"):
        return f
    return None

def _find_oracle_predictions(results_dir: Path, task_name: str) -> Path | None:
    output_dir = Path(results_dir) / "logmapllm-outputs"
    if not output_dir.is_dir():
        return None
    for f in output_dir.glob("*predictions*.csv"):
        return f
    return None

def _find_reference_alignment(datasets_dir: Path, task_name: str) -> Path | None:
    """Resolve a task's reference alignment by probing on-disk layouts, not track names.

    Layouts: the conference and KG prefixed forms, plus a task directory holding its
    own `refs_equiv/` either directly under `datasets_dir` or nested one level under
    any track directory — so a new track needs no change here.
    """
    datasets_dir = Path(datasets_dir)
    candidates: list[Path] = []
    if task_name.startswith("conference-"):
        pair = task_name.split("-", 1)[1]
        candidates.append(datasets_dir / "conference" / "refs_equiv" / f"{pair}.tsv")
    elif task_name.startswith("kg-"):
        pair = task_name[3:]
        candidates.append(
            datasets_dir / "knowledgegraph" / pair / "refs_equiv" / "reference_all.tsv"
        )
    else:
        # flat: <datasets>/<task>/refs_equiv/full.tsv
        candidates.append(datasets_dir / task_name / "refs_equiv" / "full.tsv")
        # nested: <datasets>/<track>/<task>/refs_equiv/full.tsv, for any track directory
        if datasets_dir.is_dir():
            for track_dir in sorted(p for p in datasets_dir.iterdir() if p.is_dir()):
                candidates.append(track_dir / task_name / "refs_equiv" / "full.tsv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


###
# EVALUATION ORCHESTRATION
# _VALID_METRICS:
#   'global': global alignment P/R/F1 with TP/FP/FN, system and reference
#       alignment sizes, and the evaluation backend as 'source'
#       (custom, deeponto, or kg_partial for partial reference).
#   'oracle': oracle metrics (P/R/F1/YI/Sen/Spec) plus TP/TN/FP/FN, error
#       count, total candidates (typically the M_ask size), oracle-excluded
#       (mappings that could not be answered), and partial-scope-excluded
#       (when using partial reference alignments).
#   'ranking': stub reserved for subsumption ranking metrics.
# _DEFAULT_METRICS: the metrics used when none are requested.
###

_VALID_METRICS = {"global", "oracle"}
_DEFAULT_METRICS = ["global", "oracle"]


def _normalise_metrics(metrics: list[str] | tuple[str, ...] | None) -> list[str]:
    requested = list(_DEFAULT_METRICS) if metrics is None else [str(m).strip() for m in metrics]
    if not requested or any(not metric for metric in requested):
        raise ValueError("At least one evaluation metric must be requested")
    unsupported = sorted(set(requested) - _VALID_METRICS)
    if unsupported:
        raise ValueError(
            f"Unsupported evaluation metric(s): {unsupported}; supported metrics are "
            f"{sorted(_VALID_METRICS)}"
        )
    return list(dict.fromkeys(requested))


def evaluate_alignment(
    system_mappings_path: Path,
    reference_path: Path,
    oracle_predictions_path: Path | None = None,
    train_reference_path: Path | None = None,
    test_cands_path: Path | None = None,
    initial_alignment_path: Path | None = None,
    task_name: str = "",
    metrics: list[str] | None = None,
    output_json_path: Path | None = None,
    force_custom: bool = False,
    partial_reference: bool = False,
    stratified_by_entity_type: bool = False,
    stratified_class_property: bool = False,
    jvm_memory: str = "8g",
    engines: list[str] | None = None,
    engine_options: dict[str, dict] | None = None,
) -> dict:
    """
    Run full evaluation and return the results dict.

    `engines` mirrors `[evaluation] engines`: `None` keeps the historical engine selection
    and result shape exactly; otherwise its first entry selects the engine of the plain
    `global`/`oracle` blocks and every further entry (a track-faithful engine such as
    `logmap_oaei` or `bioml`, constructed from `engine_options[name]`) adds a
    `global_<engine>` block under its own reference convention, plus an `oracle_<engine>`
    block when oracle metrics are requested. The plain blocks are never affected.
    """
    metrics = _normalise_metrics(metrics)
    engines = [str(name).strip() for name in engines] if engines else None
    engine_options = dict(engine_options or {})
    extra_engines: list[EvaluationEngine] = []
    if engines:
        if len(set(engines)) != len(engines):
            raise ValueError("evaluation engines must be distinct")
        for name in engines[1:]:
            if name in PRIMARY_EVALUATION_ENGINES:
                raise ValueError(f"only the first evaluation engine may be a primary engine; got {name!r}")
            extra_engines.append(build_engine(name, engine_options.get(name)))
    system_mappings_path = Path(system_mappings_path)
    reference_path = Path(reference_path)
    if not system_mappings_path.is_file():
        raise FileNotFoundError(f"System alignment not found: {system_mappings_path}")
    if not reference_path.is_file():
        raise FileNotFoundError(f"Reference alignment not found: {reference_path}")
    if "oracle" in metrics:
        oracle_path = Path(oracle_predictions_path) if oracle_predictions_path else None
        if oracle_path is None or not oracle_path.is_file():
            raise FileNotFoundError(
                "Oracle metrics were requested but the predictions artifact is missing"
            )
    # Validate the train reference here: the engines silently skip a missing
    # file, which would disable train-pair exclusion.
    if train_reference_path and not Path(train_reference_path).is_file():
        raise FileNotFoundError(
            f"Train reference alignment not found: {train_reference_path}"
        )

    # pick an engine for this task based on the config flags (or the explicit list)
    engine = select_engine(
        partial_reference=partial_reference,
        force_custom=force_custom,
        jvm_memory=jvm_memory,
        primary=engines[0] if engines else None,
    )
    results: dict[str, Any] = {
        "schema_version": 1,
        "task_name": task_name,
        "engine": engine.name(),
        "metrics": list(metrics),
    }
    if engines:
        results["engines"] = [engine.name(), *(extra.name() for extra in extra_engines)]

    if partial_reference:
        print("NOTE: Partial reference alignment (PARTIAL_SOURCE_COMPLETE_TARGET_COMPLETE semantics)")

    ###
    # GLOBAL MATCHING
    ###

    if "global" in metrics:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - -")

        global_metrics = engine.compute_global(
            system_mappings_path,
            reference_path,
            train_reference_path=(
                Path(train_reference_path) if train_reference_path else None
            ),
        )
        results["global"] = global_metrics
        _print_global_metrics(global_metrics)

        # Refuse a requested stratification the engine cannot provide rather
        # than quietly running a different evaluation.
        if stratified_by_entity_type and not engine.supports("stratified_global"):
            raise RuntimeError(
                f"stratified_by_entity_type was requested but engine '{engine.name()}' "
                "does not support stratified_global; set force_custom_eval=true (custom "
                "engine) or disable stratification."
            )

        if stratified_by_entity_type and engine.supports("stratified_global"):

            strat_refs: dict[str, Path] = {}
            ref_dir = reference_path.parent
            for etype in ("class", "property", "instance"):
                p = ref_dir / f"reference_{etype}.tsv"
                if p.exists():
                    strat_refs[etype] = p

            strat_kwargs: dict[str, Any] = {}

            if strat_refs:
                strat_kwargs["stratified_refs"] = strat_refs

            strat = engine.compute_stratified_global(
                system_mappings_path,
                reference_path,
                **strat_kwargs,
            )

            if strat:
                label_mode = "explicit refs" if strat_refs else "URI pattern"
                gs_mode = "partial" if partial_reference else "complete"
                _print_stratified_results(
                    strat, f"Stratified evaluation ({label_mode}, {gs_mode} GS)",
                )
                for etype, m in strat.items():
                    results[f"global_{etype}"] = m

        if stratified_class_property:
            from logmap_llm.utils.misc import compute_conference_m1_m2_stratified

            m1_m2 = compute_conference_m1_m2_stratified(
                system_mappings_path=system_mappings_path,
                reference_path=reference_path,
                initial_alignment_path=(
                    Path(initial_alignment_path) if initial_alignment_path else None
                ),
            )

            if m1_m2:
                _print_stratified_results(m1_m2, "Conference M1/M2 stratified evaluation")
                for key, m in m1_m2.items():
                    results[f"global_{key}"] = m

        # Track-faithful engines: one block each under its own reference convention.
        for extra in extra_engines:
            print("\n- - - - - - - - - - - - - - - - - - - - - - - -")
            block = extra.compute_global(
                system_mappings_path,
                reference_path,
                train_reference_path=(
                    Path(train_reference_path) if train_reference_path else None
                ),
            )
            results[f"global_{extra.name()}"] = block
            metric(f"[{extra.name()}] protocol: {block.get('protocol')}")
            _print_global_metrics(
                {k: v for k, v in block.items() if not isinstance(v, dict) or k == "printed_3dp"}
            )

    ###
    # ORACLE METRICS
    ###

    if "oracle" in metrics:
        oracle_path = Path(oracle_predictions_path) if oracle_predictions_path else None
        if oracle_path is not None and oracle_path.exists():
            reference_pairs = load_mapping_pairs(reference_path)
            predictions = load_oracle_predictions(oracle_path)

            oracle_metrics = engine.compute_oracle(
                predictions,
                reference_pairs,
            )

            results["oracle"] = oracle_metrics

            print("\n- - - - - - - - - - - - - - - - - - - - - - - -")
            _print_oracle_metrics(oracle_metrics)

            false_mappings = oracle_metrics.get("false_mappings", [])
            if false_mappings:
                fm_path = _write_false_mappings_csv(
                    false_mappings,
                    output_json_path,
                    oracle_path,
                )
                print(f"\nFalse mappings ({len(false_mappings)} FP+FN) saved to: {fm_path}")
            else:
                _false_mappings_path(output_json_path, oracle_path).unlink(missing_ok=True)

            # Track-faithful engines score the same verdicts against their own reference
            # pairs (orientation-insensitive / setting-specific); their FP/FN lists are
            # diagnostics of the plain block only, so they are not written out.
            for extra in extra_engines:
                extra_pairs = extra.oracle_reference_pairs(
                    reference_path,
                    train_reference_path=(
                        Path(train_reference_path) if train_reference_path else None
                    ),
                )
                extra_oracle = extra.compute_oracle(predictions, extra_pairs)
                extra_oracle.pop("false_mappings", None)
                results[f"oracle_{extra.name()}"] = extra_oracle
                print("\n- - - - - - - - - - - - - - - - - - - - - - - -")
                metric(f"[{extra.name()}] oracle metrics against {len(extra_pairs)} reference pairs")
                _print_oracle_metrics(extra_oracle)

    ###
    # WRITE JSON
    ###

    if output_json_path is not None:
        serialisable = {
            k: (
                {kk: vv for kk, vv in v.items() if kk != "false_mappings"}
                if isinstance(v, dict) and "false_mappings" in v
                else v
            )
            for k, v in results.items()
        }
        output_json_path = Path(output_json_path)
        atomic_json_write_strict(output_json_path, serialisable, indent=2)
        print(f"\nEvaluation results saved to: {output_json_path}")

    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LogMap-LLM Evaluation")
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--system", type=str, required=False)
    parser.add_argument("--reference", type=str, required=False)
    parser.add_argument("--oracle-predictions", type=str, default=None)
    parser.add_argument("--initial-alignment", type=str, default=None)
    parser.add_argument("--train-reference", type=str, default=None)
    parser.add_argument("--test-cands", type=str, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--partial-reference", action="store_true", default=False)
    parser.add_argument("--no-deeponto", action="store_true", default=False)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--no-cache", action="store_true", default=False)
    parser.add_argument(
        "--engines", type=str, default=None,
        help="comma-separated engine list (explicit-paths mode): primary engine first, "
             "then track-faithful engines, e.g. custom,logmap_oaei",
    )
    parser.add_argument(
        "--engine-option", action="append", default=[], metavar="ENGINE.KEY=VALUE",
        help="option of a track-faithful engine (explicit-paths mode), e.g. "
             "logmap_oaei.reference_path=reference.rdf or bioml.edition=2025",
    )
    return parser


def _run_from_config(args: argparse.Namespace) -> dict:
    """
    Canonical invocation mode: load and validate the TOML config via the shared
    subprocess bootstrap, derive paths through PipelinePaths (shared with the
    pipeline runner), then dispatch to evaluate_alignment.
    """
    from logmap_llm.utils.subprocess import subprocess_bootstrap

    cfg, run_paths, _tee = subprocess_bootstrap("EVALUATE", args=args)
    eval_cfg = cfg.evaluation

    force_custom = args.no_deeponto or eval_cfg.force_custom_eval
    partial_ref = args.partial_reference or eval_cfg.partial_reference

    system_path = run_paths.refined_mappings_tsv()
    oracle_path = run_paths.predictions_csv()
    # Conference M1/M2 stratification types URIs from the full initial alignment
    # (5-col pipe format with the entityType column), not the uncertain M_ask
    # subset; initial_alignment_path is only consumed by the M1/M2 path.
    initial_align_path = run_paths.logmap_mappings()

    output_path = Path(args.output) if args.output else run_paths.eval_json()

    metrics = (
        [metric.strip() for metric in args.metrics.split(",")]
        if args.metrics is not None
        else list(eval_cfg.metrics)
    )

    # A missing initial alignment silently degrades both M1/M2 buckets to the
    # full pair set (see utils/misc.py), so its absence is fatal when
    # stratified_class_property is configured.
    if eval_cfg.stratified_class_property and not initial_align_path.exists():
        raise FileNotFoundError(
            "stratified_class_property=true requires the initial alignment for URI typing, "
            f"but it does not exist: {initial_align_path}"
        )

    return evaluate_alignment(
        system_mappings_path=system_path,
        reference_path=eval_cfg.reference_alignment_path or "",
        oracle_predictions_path=oracle_path if oracle_path.exists() else None,
        train_reference_path=eval_cfg.train_alignment_path,
        test_cands_path=eval_cfg.test_cands_path,
        initial_alignment_path=initial_align_path if initial_align_path.exists() else None,
        task_name=cfg.alignmentTask.task_name,
        metrics=metrics,
        output_json_path=output_path,
        force_custom=force_custom,
        partial_reference=partial_ref,
        stratified_by_entity_type=eval_cfg.stratified_by_entity_type,
        stratified_class_property=eval_cfg.stratified_class_property,
        jvm_memory=eval_cfg.jvm_memory,
        engines=eval_cfg.engines,
        engine_options=eval_cfg.engine_options(),
    )


def _parse_engine_options(values: list[str] | None) -> dict[str, dict]:
    """`--engine-option engine.key=value` entries -> {engine: {key: value}} with
    true/false and integers coerced."""
    options: dict[str, dict] = {}
    for item in values or []:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"--engine-option expects engine.key=value, got {item!r}")
        dotted, raw = item.split("=", 1)
        engine_name, key = dotted.split(".", 1)
        value: Any = raw
        if raw.lower() in {"true", "false"}:
            value = raw.lower() == "true"
        elif raw.isdigit():
            value = int(raw)
        options.setdefault(engine_name, {})[key] = value
    return options


def _run_from_explicit_paths(args: argparse.Namespace) -> dict:
    """
    Secondary invocation mode: all paths passed explicitly on the command line,
    bypassing config loading and the bootstrap (no cfg object, no subprocess
    log). TODO: test suite.
    """
    metrics = (
        [metric.strip() for metric in args.metrics.split(",")]
        if args.metrics is not None
        else list(_DEFAULT_METRICS)
    )
    return evaluate_alignment(
        system_mappings_path=args.system,
        reference_path=args.reference,
        oracle_predictions_path=args.oracle_predictions,
        train_reference_path=args.train_reference,
        test_cands_path=args.test_cands,
        initial_alignment_path=args.initial_alignment,
        task_name=args.task_name,
        metrics=metrics,
        output_json_path=args.output,
        force_custom=args.no_deeponto,
        partial_reference=args.partial_reference,
        engines=(
            [name.strip() for name in args.engines.split(",")] if args.engines else None
        ),
        engine_options=_parse_engine_options(args.engine_option),
    )


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.config:
        results = _run_from_config(args)
    else:
        results = _run_from_explicit_paths(args)

    summary = {
        key: value
        for key, value in results.items()
        if not (isinstance(value, dict) and "false_mappings" in value)
    }

    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
