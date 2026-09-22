"""
logmap_llm.evaluation.engines.bioml

The OAEI Bio-ML track protocols, edition by edition, as an evaluation engine. Pure
Python; the semantics are the functions of ``evaluation/conventions.py``.

Options (``[evaluation.bioml]``):

* ``edition`` — 2022, 2023, 2024, 2025 or 2026;
* ``setting`` — ``unsupervised`` (default), ``semi_supervised`` (editions ≤ 2025) or
  ``codabench`` (2026 only); this chooses the **headline** block, every other view whose
  inputs are configured is reported under ``views``;
* ``reference_path`` — the complete equivalence reference (``refs_equiv/full.tsv`` for
  2022–2025, the standard ``reference.rdf`` for 2026); defaults to
  ``evaluation.reference_alignment_path``;
* ``ignored_classes_path`` — IRIs annotated ``use_in_alignment = false`` (editions ≥ 2023):
  every prediction touching one is dropped before scoring (the OAEI 2025 official rule);
* ``train_alignment_path`` / ``test_reference_path`` — the semi-supervised split
  (``refs_equiv/train.tsv`` and ``test.tsv``; without a test file the test slice is the
  full reference minus the training pairs); ``train_alignment_path`` falls back to
  ``evaluation.train_alignment_path``;
* ``reference_repaired_path`` — the 2026 coherence-repaired reference with ``?`` flags;
* ``deprecated_classes_path`` — 2026 ``owl:deprecated`` IRIs, dropped from predictions and
  references alike (the organisers' rule);
* ``split_path`` — the 2026 ``split.tsv`` (``SrcEntity``, ``TgtEntity``, ``split``) for
  the CodaBench protocol (reference restricted to the ``test`` rows, ``train``+``valid``
  rows masked from the predictions).

| edition | unsupervised | semi_supervised | codabench |
|---|---|---|---|
| 2022 | plain vs full | null-reference vs test with train null | – |
| 2023–2025 | ignored classes dropped, then plain vs full | ignored classes dropped, then null-reference | – |
| 2026 | deprecated dropped; ``standard`` (all cells) and ``repaired`` (coherence-aware, headline) | – | ``masked_split`` on the test slice, both references |

Views are named ``unsupervised``, ``semi_supervised``, ``standard``, ``repaired``,
``codabench_standard`` and ``codabench_repaired``. Oracle metrics use plain semantics on
the headline reference's positive pairs (a candidate touching an ignored class is simply
not in the reference, i.e. it counts as a negative, exactly as the plain engine does).
"""
from __future__ import annotations

import copy
from pathlib import Path

from logmap_llm.evaluation.conventions import (
    cells_to_pairs,
    filter_ignored,
    flagged_pairs,
    load_alignment_cells,
    load_iri_list,
    load_reference_cells,
    load_split,
    prf_coherence_aware,
    prf_masked_split,
    prf_null_reference,
    prf_plain,
    prf_standard_all_cells,
)
from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.metrics import compute_oracle_metrics

EDITIONS: tuple[int, ...] = (2022, 2023, 2024, 2025, 2026)
SETTINGS: tuple[str, ...] = ("unsupervised", "semi_supervised", "codabench")


def _path(value: str | Path | None) -> Path | None:
    return Path(value) if value else None


def _require_file(path: Path | None, label: str) -> Path:
    if path is None:
        raise ValueError(f"bioml: {label} is required for this edition/setting")
    if not path.is_file():
        raise FileNotFoundError(f"bioml: {label} not found: {path}")
    return path


class BioMLEvaluationEngine(EvaluationEngine):
    """The OAEI Bio-ML protocols (2022–2026) with the configured setting as headline."""

    def __init__(
        self,
        edition: int = 2025,
        setting: str = "unsupervised",
        reference_path: str | Path | None = None,
        reference_repaired_path: str | Path | None = None,
        test_reference_path: str | Path | None = None,
        train_alignment_path: str | Path | None = None,
        ignored_classes_path: str | Path | None = None,
        deprecated_classes_path: str | Path | None = None,
        split_path: str | Path | None = None,
        **ignored,
    ) -> None:
        if int(edition) not in EDITIONS:
            raise ValueError(f"bioml: edition must be one of {EDITIONS}, got {edition!r}")
        if setting not in SETTINGS:
            raise ValueError(f"bioml: setting must be one of {SETTINGS}, got {setting!r}")
        self.edition = int(edition)
        self.setting = setting
        self.reference_path = _path(reference_path)
        self.reference_repaired_path = _path(reference_repaired_path)
        self.test_reference_path = _path(test_reference_path)
        self.train_alignment_path = _path(train_alignment_path)
        self.ignored_classes_path = _path(ignored_classes_path)
        self.deprecated_classes_path = _path(deprecated_classes_path)
        self.split_path = _path(split_path)
        if self.setting == "codabench" and self.edition != 2026:
            raise ValueError("bioml: the codabench setting exists for edition 2026 only")
        if self.setting == "semi_supervised" and self.edition == 2026:
            raise ValueError("bioml: edition 2026 has no semi_supervised setting (use codabench)")

    def name(self) -> str:
        return "bioml"

    # ------------------------------------------------------------------ helpers

    def resolve_reference_path(self, reference_path: Path | str | None) -> Path:
        if self.reference_path is not None:
            return self.reference_path
        if reference_path is None:
            raise ValueError("bioml: no reference path configured")
        return Path(reference_path)

    def _resolve_train(self, train_reference_path: Path | str | None) -> Path | None:
        return self.train_alignment_path or _path(train_reference_path)

    def _headline_name(self, views: dict[str, dict]) -> str:
        if self.edition <= 2025:
            wanted = self.setting
        elif self.setting == "codabench":
            wanted = "codabench_repaired" if "codabench_repaired" in views else "codabench_standard"
        else:
            wanted = "repaired" if "repaired" in views else "standard"
        if wanted not in views:
            raise ValueError(
                f"bioml: the {self.setting!r} setting of edition {self.edition} needs inputs that "
                f"are not configured (available views: {sorted(views)})"
            )
        return wanted

    # ------------------------------------------------------------------ views

    def _views_pre_2026(
        self, system_cells: list, reference: Path, train_reference_path: Path | str | None,
    ) -> tuple[dict[str, dict], dict]:
        ignored: set[str] = set()
        if self.edition >= 2023 and self.ignored_classes_path is not None:
            ignored = load_iri_list(_require_file(self.ignored_classes_path, "ignored_classes_path"))
        kept = filter_ignored(system_cells, ignored)
        system_pairs = cells_to_pairs(kept)            # every relation counts, oriented
        reference_pairs = cells_to_pairs(load_reference_cells(reference))
        views = {"unsupervised": prf_plain(system_pairs, reference_pairs, source=self.name())}
        meta = {
            "ignored_classes": len(ignored),
            "ignored_classes_path": str(self.ignored_classes_path) if self.ignored_classes_path else None,
            "system_cells": len(system_cells),
            "system_cells_ignored": len(system_cells) - len(kept),
        }
        train = self._resolve_train(train_reference_path)
        if train is not None:
            train_pairs = cells_to_pairs(load_alignment_cells(_require_file(train, "train_alignment_path")))
            if self.test_reference_path is not None:
                test_pairs = cells_to_pairs(
                    load_alignment_cells(_require_file(self.test_reference_path, "test_reference_path"))
                )
            else:
                test_pairs = reference_pairs - train_pairs
            views["semi_supervised"] = prf_null_reference(
                system_pairs, test_pairs, train_pairs, source=self.name(),
            )
            meta["train_alignment_path"] = str(train)
            meta["test_reference_path"] = (
                str(self.test_reference_path) if self.test_reference_path else None
            )
        elif self.setting == "semi_supervised":
            raise ValueError("bioml: the semi_supervised setting needs train_alignment_path")
        return views, meta

    def _views_2026(self, system_cells: list, reference: Path) -> tuple[dict[str, dict], dict]:
        deprecated: set[str] = set()
        if self.deprecated_classes_path is not None:
            deprecated = load_iri_list(_require_file(self.deprecated_classes_path, "deprecated_classes_path"))
        system = filter_ignored(system_cells, deprecated)
        standard = filter_ignored(load_reference_cells(reference), deprecated)
        repaired = None
        if self.reference_repaired_path is not None:
            repaired = filter_ignored(
                load_reference_cells(_require_file(self.reference_repaired_path, "reference_repaired_path")),
                deprecated,
            )
        views = {"standard": prf_standard_all_cells(system, standard, source=self.name())}
        if repaired is not None:
            views["repaired"] = prf_coherence_aware(system, repaired, source=self.name())
        meta = {
            "deprecated_classes": len(deprecated),
            "deprecated_classes_path": (
                str(self.deprecated_classes_path) if self.deprecated_classes_path else None
            ),
            "reference_repaired_path": (
                str(self.reference_repaired_path) if self.reference_repaired_path else None
            ),
            "system_cells": len(system_cells),
            "system_cells_ignored": len(system_cells) - len(system),
        }
        if self.split_path is not None:
            split = load_split(_require_file(self.split_path, "split_path"))
            test = split.get("test", set())
            masked = split.get("train", set()) | split.get("valid", set())
            system_pairs = cells_to_pairs(system, drop_flagged=True)
            views["codabench_standard"] = prf_masked_split(
                system_pairs, cells_to_pairs(standard), test, masked, source=self.name(),
            )
            if repaired is not None:
                views["codabench_repaired"] = prf_masked_split(
                    system_pairs, cells_to_pairs(repaired), test, masked,
                    flagged=flagged_pairs(repaired), source=self.name(),
                )
            meta["split_path"] = str(self.split_path)
        elif self.setting == "codabench":
            raise ValueError("bioml: the codabench setting needs split_path")
        return views, meta

    def compute_views(
        self, system_path: Path, reference_path: Path | str | None, train_reference_path: Path | str | None = None,
    ) -> tuple[dict[str, dict], dict, str]:
        """Every view whose inputs are configured, the shared metadata and the headline name."""
        reference = _require_file(self.resolve_reference_path(reference_path), "reference_path")
        system_cells = load_alignment_cells(system_path)
        if self.edition <= 2025:
            views, meta = self._views_pre_2026(system_cells, reference, train_reference_path)
        else:
            views, meta = self._views_2026(system_cells, reference)
        meta["reference_path"] = str(reference)
        return views, meta, self._headline_name(views)

    # ------------------------------------------------------------------ engine API

    def compute_global(
        self,
        system_path: Path,
        reference_path: Path,
        train_reference_path: Path | None = None,
        **options,
    ) -> dict:
        views, meta, headline = self.compute_views(system_path, reference_path, train_reference_path)
        block = copy.deepcopy(views[headline])
        block.update(meta)
        block.update({
            "edition": self.edition,
            "setting": self.setting,
            "headline_view": headline,
            "views": views,
        })
        return block

    def oracle_reference_pairs(self, reference_path: Path | str | None, **options) -> set[tuple[str, str]]:
        """The headline view's positive reference pairs (plain, oriented semantics)."""
        reference = _require_file(self.resolve_reference_path(reference_path), "reference_path")
        if self.edition <= 2025:
            pairs = cells_to_pairs(load_reference_cells(reference))
            if self.setting == "semi_supervised":
                train = self._resolve_train(options.get("train_reference_path"))
                if train is None:
                    raise ValueError("bioml: the semi_supervised setting needs train_alignment_path")
                train_pairs = cells_to_pairs(load_alignment_cells(train))
                if self.test_reference_path is not None:
                    pairs = cells_to_pairs(load_alignment_cells(self.test_reference_path))
                pairs = pairs - train_pairs
            return pairs
        deprecated: set[str] = set()
        if self.deprecated_classes_path is not None:
            deprecated = load_iri_list(self.deprecated_classes_path)
        if self.reference_repaired_path is not None:
            cells = filter_ignored(load_reference_cells(self.reference_repaired_path), deprecated)
            pairs = cells_to_pairs(cells) - flagged_pairs(cells)
        else:
            pairs = cells_to_pairs(filter_ignored(load_reference_cells(reference), deprecated))
        if self.setting == "codabench" and self.split_path is not None:
            pairs = pairs & load_split(self.split_path).get("test", set())
        return pairs

    def compute_oracle(self, predictions: list[dict], reference_pairs: set[tuple[str, str]], **options) -> dict:
        metrics = compute_oracle_metrics(predictions, reference_pairs, partial_reference=False)
        metrics["source"] = self.name()
        metrics["edition"] = self.edition
        metrics["setting"] = self.setting
        return metrics

    def supports(self, metric_name: str) -> bool:
        if metric_name == "stratified_global":
            return False
        return super().supports(metric_name)
