"""
logmap_llm.evaluation.conventions

Cell-level alignment loaders and the reference conventions of the OAEI tracks and of
the EACL 2025 paper ("Large Language Models as Oracles for Ontology Alignment"), as
pure functions over ``(source_iri, target_iri, relation)`` cells.

The plain evaluator (``engines/custom.py`` + ``metrics.compute_prf``) scores oriented
``(source, target)`` pairs against the ``=`` cells of a reference and ignores every
relation symbol. The published tracks do not all work that way:

* LogMap's own OAEI evaluator (``HashAlignment`` + ``StandardMeasures``, Daniel Faria's
  SEALS client code; the evaluator behind EACL 2025 Table 3 and the OAEI 2021 LargeBio
  results) is relation-aware, orientation-insensitive, ignores reference cells flagged
  ``?`` and prints P/R rounded to three decimals before F is derived from them
  (:func:`prf_logmap_oaei`).
* The Bio-ML organisers' package scores relation-agnostic pair existence; its
  coherence-aware variant removes the ``?`` cells from both sides
  (:func:`prf_coherence_aware`), its standard variant counts them as positives
  (:func:`prf_standard_all_cells`).
* The OAEI 2025 Bio-ML protocol drops every prediction touching a class annotated
  ``use_in_alignment = false`` (:func:`filter_ignored`), DeepOnto's semi-supervised
  evaluation subtracts the training split from both sides (:func:`prf_null_reference`)
  and the 2026 CodaBench scorer restricts the reference to the hidden test split while
  masking the train+valid pairs from the predictions (:func:`prf_masked_split`).

Every scoring function returns the ``metrics.compute_prf`` dict shape (``precision``,
``recall``, ``f1``, ``true_positives``, ``false_positives``, ``false_negatives``,
``system_size``, ``reference_size``, ``metric_notes``, ``source``) plus a ``protocol``
key and protocol-specific counts, so ``contract.validate_evaluation_payload`` can
reconcile every block the same way: ``system_size`` is always the *evaluated* system
size (``tp + fp``) and ``reference_size`` the evaluated reference size (``tp + fn``);
the raw cell counts travel in the extra keys. Undefined ratios are ``None`` with a
``metric_notes`` reason, never 0.0.

Validated against: EACL 2025 Tables 2 and 3 (9/9 tasks), the OAEI 2025 Bio-ML official
figures (6/6), the OAEI 2021 LargeBio published figures (15/15) and the campaign's
organiser-package scores (``scripts/regression_eacl.py`` in ``logmap_llm_evals``).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

from logmap_llm.constants import DEEPONTO_TSV_HEADER_PREFIXES
from logmap_llm.evaluation.metrics import _nullable_prf_from_counts, compute_prf

Cell = tuple[str, str, str]
Pair = tuple[str, str]

EQUIVALENCE = "="
SUBSUMED_BY = "<"
SUBSUMES = ">"
FLAGGED = "?"
RELATIONS: frozenset[str] = frozenset({EQUIVALENCE, SUBSUMED_BY, SUBSUMES, FLAGGED})
REVERSED_RELATION: dict[str, str] = {
    EQUIVALENCE: EQUIVALENCE,
    SUBSUMED_BY: SUBSUMES,
    SUBSUMES: SUBSUMED_BY,
    FLAGGED: FLAGGED,
}

# Relation spellings seen in OAEI Alignment RDF files. Unknown text is read as an
# equivalence, exactly as LogMap's HashAlignment constructor does for any mapping that is
# not a subsumption or an unknown; an absent <relation> element means equivalence too.
_RELATION_ALIASES: dict[str, str] = {
    "": EQUIVALENCE, "=": EQUIVALENCE, "==": EQUIVALENCE, "equivalence": EQUIVALENCE,
    "equivalent": EQUIVALENCE, "equiv": EQUIVALENCE,
    "<": SUBSUMED_BY, "<=": SUBSUMED_BY, "&lt;": SUBSUMED_BY,
    ">": SUBSUMES, ">=": SUBSUMES, "&gt;": SUBSUMES,
    "?": FLAGGED,
}

_RDF_SUFFIXES = frozenset({".rdf", ".xml", ".owl"})
_RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
# First field of a header line in a TSV/pipe alignment file (case-insensitive).
_HEADER_FIRST_FIELDS = frozenset(
    {prefix.lower() for prefix in DEEPONTO_TSV_HEADER_PREFIXES} | {"source", "src", "entity1"}
)


###
# Loaders
###


def normalise_relation(text: str | None) -> str:
    """Map a relation spelling to one of ``= < > ?``; unknown spellings are equivalences."""
    if text is None:
        return EQUIVALENCE
    return _RELATION_ALIASES.get(text.strip().lower(), EQUIVALENCE)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _rdf_cells_from_root(root: ET.Element) -> list[Cell]:
    """Every element carrying an ``entity1`` child is a cell (typed ``Cell`` or bare
    ``rdf:Description``), as in the Bio-ML organisers' loader."""
    cells: list[Cell] = []
    for element in root.iter():
        entity1 = entity2 = None
        relation: str | None = None
        for child in element:
            tag = _local_name(child.tag)
            if tag == "entity1":
                entity1 = child.get(_RDF_RESOURCE) or child.get("resource")
            elif tag == "entity2":
                entity2 = child.get(_RDF_RESOURCE) or child.get("resource")
            elif tag == "relation":
                relation = child.text
        if entity1 is not None and entity2 is not None:
            cells.append((entity1, entity2, normalise_relation(relation)))
    return cells


def _rdf_cells(path: Path) -> list[Cell]:
    try:
        return _rdf_cells_from_root(ET.parse(path).getroot())
    except ET.ParseError:
        # LogMap writes its subsumption relations as bare '<' / '>' inside <relation>,
        # which is not well-formed XML. Repair those two spellings and parse again; any
        # other parse error propagates.
        text = path.read_text(encoding="utf-8", errors="replace")
        repaired = (
            text.replace("<relation><</relation>", "<relation>&lt;</relation>")
            .replace("<relation>></relation>", "<relation>&gt;</relation>")
        )
        if repaired == text:
            raise
        return _rdf_cells_from_root(ET.fromstring(repaired))


def load_alignment_cells(path: str | Path, *, sep: str | None = None) -> list[Cell]:
    """
    Load an alignment file as an ordered list of ``(source_iri, target_iri, relation)``
    cells with the relation kept (``= < > ?``).

    Formats, chosen by suffix:

    * ``.rdf`` / ``.xml`` / ``.owl`` — OAEI Alignment RDF/XML (``<Cell>`` with
      ``entity1``/``entity2``/``relation``; ElementTree unescapes ``&lt;``);
    * ``.txt`` — LogMap pipe format ``src|tgt|rel|conf|type``;
    * anything else — TSV with two or more columns (a DeepOnto ``SrcEntity`` header or a
      ``source``/``src``/``entity1`` header is skipped); the relation comes from the third
      column only when it is exactly one of ``= < > ?``, otherwise it is ``=`` (a third
      column holding a score is not a relation).

    ``sep`` overrides the separator for text files. Cells are not de-duplicated and their
    orientation is kept; the scoring functions decide both. A non-empty file that yields
    no cell raises ``ValueError`` (wrong separator or malformed file), like
    ``io.load_mapping_pairs``.
    """
    path = Path(path)
    if path.suffix.lower() in _RDF_SUFFIXES:
        return _rdf_cells(path)
    if sep is None:
        sep = "|" if path.suffix.lower() == ".txt" else "\t"
    cells: list[Cell] = []
    data_lines = 0
    first = True
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            fields = line.split(sep)
            if first:
                first = False
                if fields[0].strip().lower() in _HEADER_FIRST_FIELDS:
                    continue
            data_lines += 1
            if len(fields) < 2:
                continue
            relation = EQUIVALENCE
            if len(fields) > 2 and fields[2].strip() in RELATIONS:
                relation = fields[2].strip()
            cells.append((fields[0].strip(), fields[1].strip(), relation))
    if data_lines > 0 and not cells:
        raise ValueError(
            f"{path}: {data_lines} data line(s) but no cells parsed with sep={sep!r} — wrong "
            "separator or malformed file; refusing to score an empty alignment."
        )
    return cells


def load_reference_cells(path: str | Path) -> list[Cell]:
    """A reference alignment with every cell and relation kept (RDF references carry the
    ``?`` cells; ``reference.tsv`` files carry ``=`` only; ``reference_full.tsv`` files
    carry every relation in their third column)."""
    return load_alignment_cells(path)


def load_iri_list(path: str | Path) -> set[str]:
    """One IRI per line; blank lines and ``#`` comments are skipped."""
    iris: set[str] = set()
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if line and not line.startswith("#"):
                iris.add(line)
    return iris


def load_split(path: str | Path) -> dict[str, set[Pair]]:
    """A Bio-ML 2026 ``split.tsv`` (``SrcEntity``, ``TgtEntity``, ``split``) as
    ``{split_name: pairs}``."""
    out: dict[str, set[Pair]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        try:
            src_col, tgt_col, split_col = (
                header.index("SrcEntity"), header.index("TgtEntity"), header.index("split"),
            )
        except ValueError as exc:
            raise ValueError(f"{path}: expected SrcEntity/TgtEntity/split columns") from exc
        for raw in handle:
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) <= max(src_col, tgt_col, split_col):
                continue
            out.setdefault(fields[split_col].strip(), set()).add(
                (fields[src_col].strip(), fields[tgt_col].strip())
            )
    return out


def cells_to_pairs(cells: Iterable[Cell], *, drop_flagged: bool = False) -> set[Pair]:
    """Oriented ``(source, target)`` pairs of the cells (optionally without ``?`` cells)."""
    return {(s, t) for s, t, r in cells if not (drop_flagged and r == FLAGGED)}


def flagged_pairs(cells: Iterable[Cell]) -> set[Pair]:
    """The ``(source, target)`` pairs of the ``?`` cells."""
    return {(s, t) for s, t, r in cells if r == FLAGGED}


def filter_ignored(cells: Iterable[Cell], ignored_iris: set[str]) -> list[Cell]:
    """Drop every cell whose source **or** target is in ``ignored_iris`` (DeepOnto's
    ``remove_ignored_mappings``; the Bio-ML ``use_in_alignment = false`` classes and the
    2026 ``owl:deprecated`` classes)."""
    if not ignored_iris:
        return list(cells)
    return [cell for cell in cells if cell[0] not in ignored_iris and cell[1] not in ignored_iris]


def filter_ignored_pairs(pairs: Iterable[Pair], ignored_iris: set[str]) -> set[Pair]:
    """The pair form of :func:`filter_ignored`."""
    if not ignored_iris:
        return set(pairs)
    return {pair for pair in pairs if pair[0] not in ignored_iris and pair[1] not in ignored_iris}


###
# LogMap's OAEI evaluator (HashAlignment + StandardMeasures)
###


class HashAlignment:
    """
    Port of ``uk.ac.ox.krr.logmap2.test.oaei.HashAlignment`` (Daniel Faria's SEALS client
    class as vendored in LogMap): an alignment stored as ``source -> target -> [relations]``
    where a pair is looked up in either orientation, with the relation reversed when the
    pair is stored the other way round. ``size`` counts distinct (pair, relation) entries.
    """

    def __init__(self, cells: Iterable[Cell] = ()):
        self._alignment: dict[str, dict[str, list[str]]] = {}
        self.size = 0
        for source, target, relation in cells:
            self.add(source, target, relation)

    def contains_pair(self, uri1: str, uri2: str) -> bool:
        return uri1 in self._alignment and uri2 in self._alignment[uri1]

    def contains(self, uri1: str, uri2: str, relation: str) -> bool:
        forward = self._alignment.get(uri1, {}).get(uri2)
        if forward is not None and relation in forward:
            return True
        backward = self._alignment.get(uri2, {}).get(uri1)
        return backward is not None and REVERSED_RELATION[relation] in backward

    def add(self, uri1: str, uri2: str, relation: str) -> None:
        if self.contains(uri1, uri2, relation):
            return
        self.size += 1
        if self.contains_pair(uri1, uri2):
            self._alignment[uri1][uri2].append(relation)
        elif self.contains_pair(uri2, uri1):
            self._alignment[uri2][uri1].append(REVERSED_RELATION[relation])
        else:
            self._alignment.setdefault(uri1, {})[uri2] = [relation]

    def items(self):
        for source, targets in self._alignment.items():
            for target, relations in targets.items():
                yield source, target, relations

    def evaluation(self, system: "HashAlignment") -> dict[str, int]:
        """
        ``reference.evaluation(system)``: true positives, false positives and false
        negatives of ``system`` against this reference, plus the number of system pairs
        discounted from the false positives because the reference flags them ``?``.
        """
        tp = 0
        fn = 0
        fp = system.size
        discounted = 0
        for source, target, relations in self.items():
            if FLAGGED in relations:
                if system.contains_pair(source, target) or system.contains_pair(target, source):
                    fp -= 1
                    discounted += 1
                continue
            for relation in relations:
                if system.contains(source, target, relation):
                    tp += 1
                else:
                    fn += 1
        fp -= tp
        return {"tp": tp, "fp": fp, "fn": fn, "discounted": discounted}


def java_round_3dp(value: float) -> float:
    """``Math.round(value * 1000.0) / 1000.0`` for a non-negative ``value``."""
    return math.floor(value * 1000.0 + 0.5) / 1000.0


def printed_measures(tp: int, fp: int, fn: int) -> dict[str, float]:
    """
    ``StandardMeasures.evaluationParameters``: the values LogMap prints (and the EACL
    paper and the OAEI LargeBio results tables reproduce). Precision and recall are
    rounded to three decimals with Java ``Math.round`` semantics and F is computed from
    the rounded values; every measure is 0.0 when either denominator is zero.
    """
    if tp + fp == 0 or tp + fn == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    precision = min(math.floor(tp * 1000.0 / (tp + fp) + 0.5) / 1000.0, 1.0)
    recall = min(math.floor(tp * 1000.0 / (tp + fn) + 0.5) / 1000.0, 1.0)
    if precision + recall == 0.0:
        # Java: 0.0 / 0.0 is NaN and Math.round(NaN) is 0, so F prints as 0.000
        return {"precision": precision, "recall": recall, "f1": 0.0}
    f1 = math.floor((2000.0 * precision * recall) / (precision + recall) + 0.5) / 1000.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _block(tp: int, fp: int, fn: int, *, protocol: str, source: str, **extra) -> dict:
    precision, recall, f1, notes = _nullable_prf_from_counts(tp, fp, fn)
    block = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "system_size": tp + fp,
        "reference_size": tp + fn,
        "metric_notes": notes,
        "source": source,
        "protocol": protocol,
    }
    block.update(extra)
    return block


def prf_logmap_oaei(
    system_cells: Iterable[Cell],
    reference_cells: Iterable[Cell],
    *,
    rounded: bool = False,
    source: str = "logmap_oaei",
) -> dict:
    """
    P/R/F1 the way LogMap's OAEI evaluator computes them (EACL 2025 Table 3, OAEI 2021
    LargeBio): pairs are orientation-insensitive; a system cell is a true positive only
    if the reference holds the pair with the same relation (``<`` and ``>`` swap when the
    pair is stored the other way round); reference cells flagged ``?`` are ignored and a
    system pair on such a cell is discounted from the false positives.

    ``printed_3dp`` always carries the three-decimal values LogMap prints (P and R rounded
    with Java ``Math.round`` before F is derived). With ``rounded=True`` those printed
    values also become the block's ``precision``/``recall``/``f1`` and ``rounded_3dp`` is
    ``true`` (the contract then reconciles against the rounded ratios); undefined ratios
    stay ``None``.
    """
    system_cells = list(system_cells)
    reference_cells = list(reference_cells)
    system = HashAlignment(system_cells)
    reference = HashAlignment(reference_cells)
    counts = reference.evaluation(system)
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    wrong_relation = 0
    for s, t, relations in system.items():
        for relation in relations:
            if relation == FLAGGED:
                continue
            if (reference.contains_pair(s, t) or reference.contains_pair(t, s)) and not (
                reference.contains(s, t, relation)
                or reference.contains(s, t, FLAGGED)
            ):
                wrong_relation += 1
    block = _block(
        tp, fp, fn,
        protocol="logmap_oaei",
        source=source,
        printed_3dp=printed_measures(tp, fp, fn),
        rounded_3dp=False,
        system_cells=len(system_cells),
        system_hash_size=system.size,
        system_flagged_discounted=counts["discounted"],
        system_cells_with_wrong_relation=wrong_relation,
        reference_cells=len(reference_cells),
        reference_hash_size=reference.size,
        reference_flagged=sum(1 for _, _, r in reference_cells if r == FLAGGED),
    )
    if rounded:
        printed = block["printed_3dp"]
        for key in ("precision", "recall", "f1"):
            if block[key] is not None:
                block[key] = printed[key]
        block["rounded_3dp"] = True
    return block


###
# Bio-ML organiser conventions
###


def _coherence_aware_pairs(
    system_pairs: set[Pair],
    reference_pairs: set[Pair],
    flagged: set[Pair],
    *,
    protocol: str,
    source: str,
    **extra,
) -> dict:
    positive = reference_pairs - flagged          # R+ = R - U
    scored = system_pairs - flagged               # A - U: a predicted '?' pair is neither TP nor FP
    tp = len(scored & positive)
    return _block(
        tp, len(scored) - tp, len(positive) - tp,
        protocol=protocol,
        source=source,
        system_pairs=len(system_pairs),
        predicted_flagged=len(system_pairs & flagged),
        reference_pairs=len(reference_pairs),
        reference_flagged=len(flagged),
        **extra,
    )


def prf_coherence_aware(
    system_cells: Iterable[Cell], reference_cells: Iterable[Cell], *, source: str = "bioml",
) -> dict:
    """
    The Bio-ML organisers' ``global_prf1_coherence_aware`` (the OAEI 2026 Bio-ML headline
    and the LargeBio "official" view of the campaign's appendix): relations are ignored,
    the reference is every cell whatever its relation, the ``?`` cells (U) are removed from
    the reference (R+ = R − U) and from the system side (predicted pairs in U are neither
    true nor false positives). System cells with relation ``?`` are not predictions.
    """
    system_cells = list(system_cells)
    reference_cells = list(reference_cells)
    return _coherence_aware_pairs(
        cells_to_pairs(system_cells, drop_flagged=True),
        cells_to_pairs(reference_cells),
        flagged_pairs(reference_cells),
        protocol="coherence_aware",
        source=source,
        system_cells=len(system_cells),
        reference_cells=len(reference_cells),
    )


def prf_standard_all_cells(
    system_cells: Iterable[Cell], reference_cells: Iterable[Cell], *, source: str = "bioml",
) -> dict:
    """
    The organisers' ``global_prf1``: every reference cell, including the ``?`` cells, is a
    positive; relations are ignored on both sides; system cells with relation ``?`` are not
    predictions.
    """
    system_cells = list(system_cells)
    reference_cells = list(reference_cells)
    system_pairs = cells_to_pairs(system_cells, drop_flagged=True)
    reference_pairs = cells_to_pairs(reference_cells)
    tp = len(system_pairs & reference_pairs)
    return _block(
        tp, len(system_pairs) - tp, len(reference_pairs) - tp,
        protocol="standard_all_cells",
        source=source,
        system_cells=len(system_cells),
        reference_cells=len(reference_cells),
        reference_flagged=len(flagged_pairs(reference_cells)),
    )


###
# Plain, null-reference and masked-split conventions over oriented pairs
###


def prf_plain(system_pairs: set[Pair], reference_pairs: set[Pair], *, source: str = "custom") -> dict:
    """``metrics.compute_prf`` with the ``protocol`` key (no behaviour change)."""
    block = compute_prf(set(system_pairs), set(reference_pairs))
    block["source"] = source
    block["protocol"] = "plain"
    return block


def prf_null_reference(
    system_pairs: set[Pair],
    reference_pairs: set[Pair],
    null_pairs: set[Pair],
    *,
    source: str = "bioml",
) -> dict:
    """
    DeepOnto ``AlignmentEvaluator.f1(..., null_reference_mappings)`` (the Bio-ML
    semi-supervised setting with the training split as null references): the null pairs
    are removed from both the predictions and the reference before plain scoring. This is
    what the plain engine does with ``evaluation.train_alignment_path``.
    """
    system_pairs, reference_pairs, null_pairs = set(system_pairs), set(reference_pairs), set(null_pairs)
    scored = system_pairs - null_pairs
    positive = reference_pairs - null_pairs
    tp = len(scored & positive)
    return _block(
        tp, len(scored) - tp, len(positive) - tp,
        protocol="null_reference",
        source=source,
        system_pairs=len(system_pairs),
        system_null_removed=len(system_pairs) - len(scored),
        reference_pairs=len(reference_pairs),
        reference_null_removed=len(reference_pairs) - len(positive),
        null_size=len(null_pairs),
    )


def prf_masked_split(
    system_pairs: set[Pair],
    reference_pairs: set[Pair],
    test_pairs: set[Pair],
    masked_pairs: set[Pair],
    *,
    flagged: set[Pair] | None = None,
    source: str = "bioml",
) -> dict:
    """
    The OAEI 2026 Bio-ML CodaBench protocol: the reference (and its ``?`` subset, when
    ``flagged`` is given) is restricted to the hidden test slice, the train+valid pairs are
    masked from the predictions, and every remaining prediction outside the test reference
    is a false positive. With ``flagged`` the coherence-aware rule applies on the slice,
    otherwise the standard rule.
    """
    system_pairs, reference_pairs = set(system_pairs), set(reference_pairs)
    test_pairs, masked_pairs = set(test_pairs), set(masked_pairs)
    scored = system_pairs - masked_pairs
    sliced = reference_pairs & test_pairs
    extra = {
        "system_pairs_before_masking": len(system_pairs),
        "system_masked_removed": len(system_pairs) - len(scored),
        "reference_pairs_before_slicing": len(reference_pairs),
        "test_slice_size": len(test_pairs),
        "masked_size": len(masked_pairs),
    }
    if flagged is not None:
        # _coherence_aware_pairs records system_pairs (= |A - masked|), reference_pairs
        # (= |R ∩ test|), predicted_flagged and reference_flagged itself.
        return _coherence_aware_pairs(
            scored, sliced, set(flagged) & test_pairs,
            protocol="masked_split", source=source, **extra,
        )
    tp = len(scored & sliced)
    return _block(
        tp, len(scored) - tp, len(sliced) - tp,
        protocol="masked_split", source=source,
        system_pairs=len(scored), reference_pairs=len(sliced), **extra,
    )


__all__ = [
    "Cell", "Pair", "EQUIVALENCE", "SUBSUMED_BY", "SUBSUMES", "FLAGGED", "RELATIONS",
    "REVERSED_RELATION", "HashAlignment", "normalise_relation",
    "load_alignment_cells", "load_reference_cells", "load_iri_list", "load_split",
    "cells_to_pairs", "flagged_pairs", "filter_ignored", "filter_ignored_pairs",
    "java_round_3dp", "printed_measures",
    "prf_plain", "prf_logmap_oaei", "prf_coherence_aware", "prf_standard_all_cells",
    "prf_null_reference", "prf_masked_split",
]
