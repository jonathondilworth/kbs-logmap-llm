"""
logmap_llm.evaluation.engines.deeponto
DeepOnto-based evaluation engine; wraps ``deeponto.align.evaluation``
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from logmap_llm.constants import DEEPONTO_TSV_HEADER_PREFIXES
from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.io import load_mapping_pairs
from logmap_llm.evaluation.metrics import compute_oracle_metrics


_DEEPONTO_AVAILABLE: bool | None = None
_DEEPONTO_MODULES: dict[str, Any] = {}


def _ensure_jvm_memory(memory: str = "8g", *, overwrite: bool = False) -> None:
    """
    Set JVM memory environment variables prior to DeepOnto spinning up JVM
    """
    for var_name in ("JAVA_MEMORY", "DEEPONTO_JVM_MEMORY", "JVM_MEMORY"):
        if overwrite or var_name not in os.environ:
            os.environ[var_name] = memory


def _prestart_jvm_for_deeponto() -> None:
    """
    Pre-start the JVM before importing deeponto.align. deeponto 0.9.3 sizes its
    JVM heap via an interactive ``click.prompt`` at the first import of
    deeponto.onto / deeponto.align (it ignores JAVA_MEMORY / DEEPONTO_JVM_MEMORY
    / JVM_MEMORY), which hangs or raises click.Abort in a non-interactive
    harness. ``deeponto.init_jvm`` reads the memory string directly and skips
    the prompt. No-op when a JVM is already running via jpype — deeponto reuses it.
    """
    import deeponto
    import jpype
    if not jpype.isJVMStarted():
        deeponto.init_jvm(os.environ.get("JAVA_MEMORY", "8g"))


def _probe_availability() -> bool:
    global _DEEPONTO_AVAILABLE, _DEEPONTO_MODULES
    if _DEEPONTO_AVAILABLE is not None:
        return _DEEPONTO_AVAILABLE

    _ensure_jvm_memory()

    try:
        _prestart_jvm_for_deeponto()   # must precede the deeponto.align import
        from deeponto.align.evaluation import AlignmentEvaluator
        from deeponto.align.mapping import ReferenceMapping, EntityMapping
        from deeponto.align.oaei import ranking_eval

        _DEEPONTO_MODULES["AlignmentEvaluator"] = AlignmentEvaluator
        _DEEPONTO_MODULES["ReferenceMapping"] = ReferenceMapping
        _DEEPONTO_MODULES["EntityMapping"] = EntityMapping
        _DEEPONTO_MODULES["ranking_eval"] = ranking_eval
        _DEEPONTO_AVAILABLE = True

    except ImportError:
        _DEEPONTO_AVAILABLE = False

    return _DEEPONTO_AVAILABLE


def _is_header_line(first_field: str) -> bool:
    """
    Check whether the first field of a TSV line is a DeepOnto-style header.
    """
    return first_field in DEEPONTO_TSV_HEADER_PREFIXES


def detect_tsv_format(filepath: Path) -> str:
    """
    Inspect the first line of a TSV file and return 'deeponto' if it starts
    with a DeepOnto header, else 'oaei'. Used to decide whether the file must
    be converted before handing it to DeepOnto.
    """
    with open(filepath) as fp:
        first_line = fp.readline().rstrip("\n")
    first_field = first_line.split("\t", 1)[0].strip()
    if _is_header_line(first_field):
        return "deeponto"
    # else:
    return "oaei"


def convert_to_deeponto_tsv(input_path: Path, output_path: Path) -> None:
    """
    Rewrite an OAEI-style TSV (no header, variable column count) as a
    three-column DeepOnto TSV with a header, written to output_path.
    Rows need at least 2 tab-separated fields, (source, target); a fourth
    column is treated as a confidence score (default 1.0); other fields
    are discarded.
    """
    with open(input_path, "r") as f_in, open(output_path, "w") as f_out:
        f_out.write("SrcEntity\tTgtEntity\tScore\n")
        for raw_line in f_in:
            parts = raw_line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src = parts[0].strip()
            tgt = parts[1].strip()
            score = parts[3].strip() if len(parts) > 3 else "1.0"
            try:
                float(score)
            except ValueError:
                score = "1.0"
            f_out.write(f"{src}\t{tgt}\t{score}\n")


def _ensure_deeponto_format(filepath: Path, tmpdir: str) -> Path:
    """
    Return a path to a DeepOnto-format TSV version of the input file,
    converting into tmpdir if needed.
    """
    if detect_tsv_format(filepath) == "deeponto":
        return filepath
    converted = Path(tmpdir) / f"converted_{filepath.name}"
    convert_to_deeponto_tsv(filepath, converted)
    return converted


class DeepOntoEvaluationEngine(EvaluationEngine):
    """
    Evaluation engine that uses DeepOnto's 'AlignmentEvaluator' for global metrics
    and falls back to the pure-function path for oracle metrics.
    """

    def __init__(self) -> None:
        if not self.is_available():
            raise RuntimeError(
                "DeepOntoEvaluationEngine constructed but DeepOnto is not "
                "importable. Check 'DeepOntoEvaluationEngine.is_available()' "
                "before instantiation, or install DeepOnto."
            )


    @classmethod
    def is_available(cls) -> bool:
        """
        Return True if DeepOnto can be imported in the current process
        """
        return _probe_availability()


    def name(self) -> str:
        return "deeponto"


    def compute_global(self, system_path: Path, reference_path: Path, train_reference_path: Path | None = None, **options) -> dict:
        """
        Compute global P/R/F1 via DeepOnto's AlignmentEvaluator.f1.
        """
        if options.get("partial_reference", False):
            raise ValueError(
                "DeepOntoEvaluationEngine does not support partial_reference. "
                "Use CustomEvaluationEngine or a track-specific engine instead."
            )

        EntityMapping = _DEEPONTO_MODULES["EntityMapping"]
        ReferenceMapping = _DEEPONTO_MODULES["ReferenceMapping"]
        AlignmentEvaluator = _DEEPONTO_MODULES["AlignmentEvaluator"]

        with tempfile.TemporaryDirectory() as tmpdir:
            sys_converted = _ensure_deeponto_format(system_path, tmpdir)
            ref_converted = _ensure_deeponto_format(reference_path, tmpdir)
            preds = EntityMapping.read_table_mappings(str(sys_converted))
            refs = ReferenceMapping.read_table_mappings(str(ref_converted))

            kwargs = {}

            if train_reference_path is not None and Path(train_reference_path).exists():
                train_converted = _ensure_deeponto_format(train_reference_path, tmpdir)
                null_refs = ReferenceMapping.read_table_mappings(str(train_converted))
                kwargs["null_reference_mappings"] = null_refs

            results = AlignmentEvaluator.f1(preds, refs, **kwargs)

        precision = results.get("P", results.get("Precision", 0.0))
        recall = results.get("R", results.get("Recall", 0.0))
        f1 = results.get("F1", results.get("F-score", 0.0))

        system = load_mapping_pairs(system_path)
        reference = load_mapping_pairs(reference_path)

        if train_reference_path is not None and Path(train_reference_path).exists():
            train_pairs = load_mapping_pairs(train_reference_path)
            system = system - train_pairs
            reference = reference - train_pairs

        tp = len(system & reference)
        fp = len(system - reference)
        fn = len(reference - system)

        metric_notes: dict[str, str] = {}
        if tp + fp == 0:
            precision = None
            metric_notes["precision"] = "undefined: no evaluated system mappings (tp+fp=0)"
        if tp + fn == 0:
            recall = None
            metric_notes["recall"] = "undefined: no reference mappings (tp+fn=0)"
        if precision is None or recall is None:
            f1 = None
            metric_notes["f1"] = "undefined: precision or recall undefined"
        elif precision + recall == 0:
            f1 = None
            metric_notes["f1"] = "undefined: precision+recall=0"

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "system_size": len(system),
            "reference_size": len(reference),
            "metric_notes": metric_notes,
            "source": "deeponto",
        }


    def compute_oracle(self, predictions: list[dict], reference_pairs: set[tuple[str, str]], **options) -> dict:
        """
        Compute oracle discrimination metrics via the pure-function path.
        """
        partial_reference = options.get("partial_reference", False)
        return compute_oracle_metrics(
            predictions,
            reference_pairs,
            partial_reference=partial_reference,
        )
