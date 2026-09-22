"""
logmap_llm.evaluation.engines.custom
Pure-Python evaluation engine. No DeepOnto, no JVM, no external libraries
beyond pandas (already a project dependency via 'utils.data').
"""
from __future__ import annotations

from pathlib import Path

from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.io import load_mapping_pairs
from logmap_llm.evaluation.metrics import (
    compute_prf,
    compute_kg_partial_prf,
    compute_oracle_metrics,
)


class CustomEvaluationEngine(EvaluationEngine):
    """
    Pure-Python evaluation engine. Implements the two required methods
    of 'EvaluationEngine' by composing 'io' and 'metrics'
    """

    def name(self) -> str:
        return "custom"


    def compute_global(self, system_path: Path, reference_path: Path, train_reference_path: Path | None = None, **options) -> dict:
        """
        Compute global P/R/F1 from on-disk alignment files.
        """
        system = load_mapping_pairs(system_path)
        reference = load_mapping_pairs(reference_path)

        # train-reference exclusion: drop training-reference pairs from both
        # system and reference before evaluation (though I'm not sure whether
        # this should actually be performed). Only used by tracks that ship a
        # separate training reference.
        if train_reference_path is not None and Path(train_reference_path).exists():
            train_pairs = load_mapping_pairs(train_reference_path)
            system = system - train_pairs
            reference = reference - train_pairs

        partial_reference = options.get("partial_reference", False)

        if partial_reference:
            return compute_kg_partial_prf(system, reference)

        return compute_prf(system, reference)


    def compute_oracle(self, predictions: list[dict], reference_pairs: set[tuple[str, str]], **options) -> dict:
        """
        Compute oracle discrimination metrics from in-memory data.
        """
        partial_reference = options.get("partial_reference", False)
        return compute_oracle_metrics(
            predictions,
            reference_pairs,
            partial_reference=partial_reference,
        )


    def compute_stratified_global(self, system_path: Path, reference_path: Path, stratified_refs=None, **options) -> dict | None:
        """
        Per-entity-type global P/R/F1 under complete gold-standard semantics.
        """
        return self._stratified_global(system_path, reference_path, compute_prf, stratified_refs=stratified_refs)


    def supports(self, metric_name: str) -> bool:
        if metric_name == "stratified_global":
            return True
        return super().supports(metric_name)