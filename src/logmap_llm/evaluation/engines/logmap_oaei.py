"""
logmap_llm.evaluation.engines.logmap_oaei

The evaluation LogMap itself performs in its OAEI harness (``HashAlignment`` +
``StandardMeasures``, the code behind EACL 2025 Table 3 and the OAEI 2021 LargeBio
results): relation-aware, orientation-insensitive, reference cells flagged ``?`` ignored.
Pure Python, no JVM; see ``evaluation/conventions.py`` for the semantics.

Options (``[evaluation.logmap_oaei]``):

* ``reference_path`` — the reference with every relation kept (``reference.rdf`` or a
  ``reference_full.tsv``); defaults to ``evaluation.reference_alignment_path``, which for
  an ``=``-only ``reference.tsv`` makes the engine identical to the plain one up to the
  relation rule and orientation.
* ``rounded`` — when true, the block's ``precision``/``recall``/``f1`` are the values LogMap
  prints (3 dp, F from the rounded P and R); ``printed_3dp`` always carries them.
* ``orientation_insensitive`` — oracle metrics label a candidate pair as positive when
  the reference holds it in either orientation (EACL ``src/evaluate.py`` labels pairs as
  frozensets); every reference cell counts, ``?`` cells included (EACL Table 2 uses the
  LargeBio ``refs_equiv/full.tsv`` with the flagged cells as positives).

A train reference is rejected: LogMap's evaluator has no null-reference notion.
"""
from __future__ import annotations

from pathlib import Path

from logmap_llm.evaluation.conventions import (
    cells_to_pairs,
    load_alignment_cells,
    load_reference_cells,
    prf_logmap_oaei,
)
from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.metrics import compute_oracle_metrics


class LogMapOAEIEvaluationEngine(EvaluationEngine):
    """LogMap's own OAEI evaluator (relation-aware, orientation-insensitive, ``?`` ignored)."""

    def __init__(
        self,
        reference_path: str | Path | None = None,
        rounded: bool = False,
        orientation_insensitive: bool = True,
        **ignored,
    ) -> None:
        self.reference_path = Path(reference_path) if reference_path else None
        self.rounded = bool(rounded)
        self.orientation_insensitive = bool(orientation_insensitive)

    def name(self) -> str:
        return "logmap_oaei"

    def resolve_reference_path(self, reference_path: Path | str | None) -> Path:
        """The engine's own reference when configured, else the harness's."""
        if self.reference_path is not None:
            return self.reference_path
        if reference_path is None:
            raise ValueError("logmap_oaei: no reference path configured")
        return Path(reference_path)

    def compute_global(
        self,
        system_path: Path,
        reference_path: Path,
        train_reference_path: Path | None = None,
        **options,
    ) -> dict:
        if train_reference_path is not None:
            raise ValueError(
                "LogMapOAEIEvaluationEngine does not accept train_reference_path: LogMap's "
                "OAEI evaluator has no null-reference (train split) notion."
            )
        reference = self.resolve_reference_path(reference_path)
        if not reference.is_file():
            raise FileNotFoundError(f"logmap_oaei reference not found: {reference}")
        rounded = bool(options.get("rounded", self.rounded))
        block = prf_logmap_oaei(
            load_alignment_cells(system_path),
            load_reference_cells(reference),
            rounded=rounded,
            source=self.name(),
        )
        block["reference_path"] = str(reference)
        return block

    def oracle_reference_pairs(self, reference_path: Path | str | None, **options) -> set[tuple[str, str]]:
        """Every cell of the engine's reference as pairs, symmetrised when
        ``orientation_insensitive`` (so oriented predictions match either way round)."""
        reference = self.resolve_reference_path(reference_path)
        pairs = cells_to_pairs(load_reference_cells(reference))
        if bool(options.get("orientation_insensitive", self.orientation_insensitive)):
            pairs = pairs | {(t, s) for s, t in pairs}
        return pairs

    def compute_oracle(self, predictions: list[dict], reference_pairs: set[tuple[str, str]], **options) -> dict:
        """``compute_oracle_metrics`` over the (symmetrised) reference pairs handed in."""
        metrics = compute_oracle_metrics(predictions, reference_pairs, partial_reference=False)
        metrics["source"] = self.name()
        metrics["orientation_insensitive"] = self.orientation_insensitive
        return metrics

    def supports(self, metric_name: str) -> bool:
        if metric_name == "stratified_global":
            return False
        return super().supports(metric_name)
