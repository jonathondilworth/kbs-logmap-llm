from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class EvaluationEngine(ABC):

    @abstractmethod
    def name(self) -> str:
        """
        Return a short identifier for this engine (used in the 'source'
        field of returned metric dicts).
        """
        ...

    @abstractmethod
    def compute_global(
        self,
        system_path: Path,
        reference_path: Path,
        train_reference_path: Path | None = None,
        **options,
    ) -> dict:
        """
        Compute global precision, recall, and F1 for an alignment.
        """
        ...

    @abstractmethod
    def compute_oracle(
        self,
        predictions: list[dict],
        reference_pairs: set[tuple[str, str]],
        **options,
    ) -> dict:
        """
        Compute oracle discrimination metrics for a set of predictions.
        """
        ...

    ###
    # Optional methods
    ###

    def compute_stratified_global(self, system_path: Path, reference_path: Path, **options) -> dict | None:
        """
        Optional: compute per-entity-type global metrics.
        """
        return None

    # Engines that can stratify override compute_stratified_global to call this
    # helper with their own P/R/F function (compute_prf for complete GS,
    # compute_kg_partial_prf for KG partial GS) and flip supports('stratified_global').
    def _stratified_global(
        self,
        system_path: Path,
        reference_path: Path,
        prf_fn,
        stratified_refs: dict[str, Path] | None = None,
    ) -> dict:
        """
        Partition system and reference alignments into class/property/instance
        strata and score each with ``prf_fn(system_stratum, reference_stratum)``.

        An explicit per-type reference file (stratified_refs[etype]) is used when
        supplied; otherwise the full reference is partitioned by URI convention
        (classify_mapping_pair). The system is always partitioned by URI
        convention, since a system mapping carries no gold typing. Strata with
        neither system nor reference mappings are omitted (undefined, not zero).
        """
        from logmap_llm.evaluation.io import load_mapping_pairs
        from logmap_llm.evaluation.metrics import classify_mapping_pair

        system = load_mapping_pairs(system_path)
        reference = load_mapping_pairs(reference_path)

        out: dict[str, dict] = {}
        for etype in ("class", "property", "instance"):
            if stratified_refs and etype in stratified_refs:
                ref_et = load_mapping_pairs(stratified_refs[etype])
            else:
                ref_et = {pair for pair in reference if classify_mapping_pair(*pair) == etype}
            sys_et = {pair for pair in system if classify_mapping_pair(*pair) == etype}
            if not ref_et and not sys_et:
                continue
            metrics = prf_fn(sys_et, ref_et)
            metrics["source"] = f"{self.name()}_stratified_{etype}"
            out[etype] = metrics
        return out

    def oracle_reference_pairs(self, reference_path: Path | str | None, **options) -> set[tuple[str, str]]:
        """
        The reference pairs this engine's oracle metrics are scored against. The default
        is the plain convention (the oriented pairs of ``reference_path``, i.e.
        ``io.load_mapping_pairs``); engines with their own reference semantics (LogMap's
        orientation-insensitive evaluator, the Bio-ML settings) override it.
        """
        from logmap_llm.evaluation.io import load_mapping_pairs

        if reference_path is None:
            raise ValueError(f"{self.name()}: no reference path configured for oracle metrics")
        return load_mapping_pairs(Path(reference_path))

    def compute_ranking(self, test_cands_path: Path, oracle_predictions_path: Path, **options) -> dict | None:
        """
        Optional: compute local ranking metrics (MRR, Hits@K).
        Stub for embedding-based ranking (ie. \\w onto embs).
        """
        return None

    ###
    # Capability query
    ###

    def supports(self, metric_name: str) -> bool:
        """
        Return True if this engine can compute the named metric.
        """
        return metric_name in {"global", "oracle"}
