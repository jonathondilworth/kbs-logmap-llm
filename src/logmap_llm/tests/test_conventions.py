"""
Unit tests for `evaluation/conventions.py`: the cell-level loaders and every reference
convention on a small hand-worked fixture (`tests/data/conventions/`, 12 system cells and
10 reference cells).

The fixture (a# = source namespace, b# = target namespace):

reference (10 cells)                      system (12 cells)
  R1  a1 = b1                               S1  a1  = b1     true positive everywhere
  R2  a2 = b2                               S2  a2  < b2     right pair, wrong relation
  R3  a3 < b3                               S3  a3  = b3     right pair, wrong relation
  R4  a4 > b4                               S4  b4  < a4     R4 reversed (orientation and relation)
  R5  a5 ? b5   (flagged)                   S5  a5  = b5     hits the flagged R5
  R6  a6 ? b6   (flagged)                   S6  a7  = b7     R7 reversed
  R7  b7 = a7   (reversed orientation)      S7  a8  = b8     the train / null pair
  R8  a8 = b8   (train / null pair)         S8  a10 = b10    touches the ignored class b10
  R9  a9 = b9   (test-only pair)            S9  a11 = b11    false positive
  R10 a10 = b10 (touches the ignored b10)   S10 a12 > b12    false positive with a relation
                                            S11 a1  = b1     duplicate of S1
                                            S12 a13 ? b13    a '?' system cell

Hand-worked expectations (independently cross-checked with logmap_llm_evals/scripts/
hashalignment_score.py and score_conventions.py):

* LogMap HashAlignment: reference entries 10 (8 non-flagged), system entries 11 (S11 is a
  duplicate). TP = R1, R4 (via S4), R7 (via S6), R8, R10 = 5; FN = R2, R3, R9 = 3; the
  flagged R5 discounts S5, so FP = 11 - 1 - 5 = 5. Printed 0.500 / 0.625 / 0.556.
* plain (oriented pairs vs the '=' cells): 11 distinct system pairs, 6 reference pairs;
  TP = a1b1, a2b2, a8b8, a10b10 = 4, FP 7, FN 2 (b7a7, a9b9).
* coherence-aware: R = 10 pairs, U = {a5b5, a6b6}, R+ = 8; A = 10 pairs (S12 dropped),
  A - U = 9; TP = a1b1, a2b2, a3b3, a8b8, a10b10 = 5, FP 4, FN 3.
* standard (all cells positive): TP = the five above + a5b5 = 6, FP 4, FN 4.
* null reference (train a8b8): 10 scored system pairs, 5 positives, TP 3, FP 7, FN 2.
* ignored class b10: S8 dropped -> plain TP 3, FP 7, FN 3.
* masked split (test = a1b1, a9b9, a10b10, b7a7, a5b5; masked = a2b2, a8b8), standard rule
  on the 6 '=' pairs: sliced reference 4, scored system 9, TP 2 (a1b1, a10b10), FP 7, FN 2;
  coherence-aware rule on all 10 pairs with U ∩ test = {a5b5}: positives 4, scored 8,
  TP 2, FP 6, FN 2.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from logmap_llm.evaluation import conventions as C

DATA = Path(__file__).parent / "data" / "conventions"
A = "http://example.org/a#"
B = "http://example.org/b#"


def _p(*names: str) -> set[tuple[str, str]]:
    """pairs from 'a1b1'-style names"""
    out = set()
    for name in names:
        left, right = name[0], name[1:]
        # names are like 'a1b1' or 'b7a7'
        first_ns, rest = left, right
        digits = ""
        i = 0
        while i < len(rest) and rest[i].isdigit():
            digits += rest[i]
            i += 1
        second_ns, second_digits = rest[i], rest[i + 1:]
        ns = {"a": A, "b": B}
        out.add((ns[first_ns] + digits, ns[second_ns] + second_digits))
    return out


@pytest.fixture(scope="module")
def system_cells():
    return C.load_alignment_cells(DATA / "system.txt")


@pytest.fixture(scope="module")
def reference_cells():
    return C.load_reference_cells(DATA / "reference.rdf")


# ---------------------------------------------------------------- loaders

def test_pipe_loader_keeps_relation_order_and_duplicates(system_cells):
    assert len(system_cells) == 12
    assert system_cells[0] == (A + "1", B + "1", "=")
    assert system_cells[1] == (A + "2", B + "2", "<")
    assert system_cells[3] == (B + "4", A + "4", "<")
    assert system_cells[9] == (A + "12", B + "12", ">")
    assert system_cells[11] == (A + "13", B + "13", "?")
    assert system_cells[10] == system_cells[0]  # duplicates are the callers' business


def test_tsv_loader_matches_pipe_loader(system_cells):
    assert C.load_alignment_cells(DATA / "system.tsv") == system_cells


def test_rdf_loader_reads_relations_and_unescapes(reference_cells):
    assert len(reference_cells) == 10
    relations = [r for _, _, r in reference_cells]
    assert relations == ["=", "=", "<", ">", "?", "?", "=", "=", "=", "="]
    assert reference_cells[6] == (B + "7", A + "7", "=")


def test_rdf_loader_repairs_logmaps_bare_relations():
    cells = C.load_alignment_cells(DATA / "bare_relations.rdf")
    assert [r for _, _, r in cells] == ["<", ">"]


def test_full_tsv_third_column_is_a_relation_only_when_it_is_a_symbol(reference_cells):
    assert C.load_alignment_cells(DATA / "reference_full.tsv") == reference_cells
    # a Score third column is not a relation: every cell is '='
    eq = C.load_alignment_cells(DATA / "reference.tsv")
    assert len(eq) == 6 and all(r == "=" for _, _, r in eq)


def test_unknown_rdf_relation_spelling_is_equivalence():
    assert C.normalise_relation(" equivalence ") == "="
    assert C.normalise_relation("<=") == "<"
    assert C.normalise_relation(">=") == ">"
    assert C.normalise_relation("?") == "?"
    assert C.normalise_relation("something-else") == "="
    assert C.normalise_relation(None) == "="


def test_empty_data_lines_raise(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("only-one-column\n")
    with pytest.raises(ValueError):
        C.load_alignment_cells(bad)
    empty = tmp_path / "empty.tsv"
    empty.write_text("")
    assert C.load_alignment_cells(empty) == []


def test_iri_list_and_split_loaders():
    assert C.load_iri_list(DATA / "ignored.txt") == {B + "10"}
    split = C.load_split(DATA / "split.tsv")
    assert split["test"] == _p("a1b1", "a9b9", "a10b10", "b7a7", "a5b5")
    assert split["train"] == _p("a2b2") and split["valid"] == _p("a8b8")


# ---------------------------------------------------------------- HashAlignment port

def test_hash_alignment_dedupes_and_reverses(system_cells, reference_cells):
    system = C.HashAlignment(system_cells)
    reference = C.HashAlignment(reference_cells)
    assert system.size == 11 and reference.size == 10
    assert reference.contains(A + "4", B + "4", ">")
    assert reference.contains(B + "4", A + "4", "<")        # reversed lookup
    assert not reference.contains(A + "4", B + "4", "=")
    assert reference.contains_pair(B + "7", A + "7") and not reference.contains_pair(A + "7", B + "7")
    assert reference.contains(A + "7", B + "7", "=")        # but the relation lookup is symmetric


def test_prf_logmap_oaei_hand_values(system_cells, reference_cells):
    block = C.prf_logmap_oaei(system_cells, reference_cells)
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (5, 5, 3)
    assert block["system_size"] == 10 and block["reference_size"] == 8
    assert block["system_cells"] == 12 and block["system_hash_size"] == 11
    assert block["reference_cells"] == 10 and block["reference_hash_size"] == 10
    assert block["reference_flagged"] == 2 and block["system_flagged_discounted"] == 1
    assert block["system_cells_with_wrong_relation"] == 2
    assert block["precision"] == 0.5 and block["recall"] == 0.625
    assert math.isclose(block["f1"], 2 * 0.5 * 0.625 / 1.125)
    assert block["printed_3dp"] == {"precision": 0.5, "recall": 0.625, "f1": 0.556}
    assert block["rounded_3dp"] is False and block["protocol"] == "logmap_oaei"
    assert block["metric_notes"] == {}


def test_prf_logmap_oaei_rounded_reports_printed_values(system_cells, reference_cells):
    block = C.prf_logmap_oaei(system_cells, reference_cells, rounded=True)
    assert (block["precision"], block["recall"], block["f1"]) == (0.5, 0.625, 0.556)
    assert block["rounded_3dp"] is True
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (5, 5, 3)


def test_printed_measures_java_rounding():
    # 2/3 -> 0.667, F from the rounded P and R; Java rounds half up
    assert C.printed_measures(2, 1, 1) == {"precision": 0.667, "recall": 0.667, "f1": 0.667}
    assert C.printed_measures(1, 1, 3) == {"precision": 0.5, "recall": 0.25, "f1": 0.333}
    assert C.printed_measures(0, 5, 3) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    assert C.printed_measures(3, 0, 0) == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert C.java_round_3dp(0.0005) == 0.001 and C.java_round_3dp(0.00049) == 0.0


def test_logmap_oaei_undefined_ratios_are_none():
    block = C.prf_logmap_oaei([], [(A + "1", B + "1", "=")])
    assert block["precision"] is None and block["recall"] == 0.0 and block["f1"] is None
    assert "precision" in block["metric_notes"] and "f1" in block["metric_notes"]
    assert block["printed_3dp"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


# ---------------------------------------------------------------- pair conventions

def test_prf_plain_hand_values(system_cells, reference_cells):
    block = C.prf_plain(C.cells_to_pairs(system_cells), C.cells_to_pairs(c for c in reference_cells if c[2] == "="))
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (4, 7, 2)
    assert block["system_size"] == 11 and block["reference_size"] == 6
    assert block["protocol"] == "plain" and block["source"] == "custom"
    assert math.isclose(block["precision"], 4 / 11) and math.isclose(block["recall"], 4 / 6)


def test_prf_coherence_aware_hand_values(system_cells, reference_cells):
    block = C.prf_coherence_aware(system_cells, reference_cells)
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (5, 4, 3)
    assert block["system_size"] == 9 and block["reference_size"] == 8
    assert block["system_pairs"] == 10 and block["predicted_flagged"] == 1
    assert block["reference_pairs"] == 10 and block["reference_flagged"] == 2
    assert block["system_cells"] == 12 and block["reference_cells"] == 10
    assert math.isclose(block["precision"], 5 / 9) and block["recall"] == 0.625
    assert block["protocol"] == "coherence_aware"


def test_prf_standard_all_cells_hand_values(system_cells, reference_cells):
    block = C.prf_standard_all_cells(system_cells, reference_cells)
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (6, 4, 4)
    assert block["system_size"] == 10 and block["reference_size"] == 10
    assert block["reference_flagged"] == 2 and block["precision"] == 0.6 and block["recall"] == 0.6


def test_prf_null_reference_hand_values(system_cells, reference_cells):
    train = C.cells_to_pairs(C.load_alignment_cells(DATA / "train.tsv"))
    assert train == _p("a8b8")
    block = C.prf_null_reference(
        C.cells_to_pairs(system_cells), C.cells_to_pairs(c for c in reference_cells if c[2] == "="), train,
    )
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (3, 7, 2)
    assert block["system_size"] == 10 and block["reference_size"] == 5
    assert block["system_null_removed"] == 1 and block["reference_null_removed"] == 1 and block["null_size"] == 1
    assert block["precision"] == 0.3 and block["recall"] == 0.6


def test_filter_ignored_then_plain(system_cells, reference_cells):
    ignored = C.load_iri_list(DATA / "ignored.txt")
    kept = C.filter_ignored(system_cells, ignored)
    assert len(kept) == 11 and all(B + "10" not in cell[:2] for cell in kept)
    block = C.prf_plain(C.cells_to_pairs(kept), C.cells_to_pairs(c for c in reference_cells if c[2] == "="))
    assert (block["true_positives"], block["false_positives"], block["false_negatives"]) == (3, 7, 3)
    assert C.filter_ignored(system_cells, set()) == list(system_cells)
    assert C.filter_ignored_pairs(C.cells_to_pairs(system_cells), ignored) == C.cells_to_pairs(kept)


def test_prf_masked_split_standard_and_coherence_rules(system_cells, reference_cells):
    split = C.load_split(DATA / "split.tsv")
    test, masked = split["test"], split["train"] | split["valid"]
    system_pairs = C.cells_to_pairs(system_cells)
    standard = C.prf_masked_split(
        system_pairs, C.cells_to_pairs(c for c in reference_cells if c[2] == "="), test, masked,
    )
    assert (standard["true_positives"], standard["false_positives"], standard["false_negatives"]) == (2, 7, 2)
    assert standard["system_size"] == 9 and standard["reference_size"] == 4
    assert standard["system_masked_removed"] == 2 and standard["test_slice_size"] == 5 and standard["masked_size"] == 2
    assert standard["protocol"] == "masked_split"
    coherent = C.prf_masked_split(
        system_pairs, C.cells_to_pairs(reference_cells), test, masked, flagged=C.flagged_pairs(reference_cells),
    )
    assert (coherent["true_positives"], coherent["false_positives"], coherent["false_negatives"]) == (2, 6, 2)
    assert coherent["system_size"] == 8 and coherent["reference_size"] == 4
    assert coherent["reference_flagged"] == 1 and coherent["predicted_flagged"] == 1
    assert coherent["precision"] == 0.25 and coherent["recall"] == 0.5


def test_every_block_reconciles_with_the_contract(system_cells, reference_cells):
    from logmap_llm.evaluation.contract import _validate_global

    blocks = [
        C.prf_logmap_oaei(system_cells, reference_cells),
        C.prf_logmap_oaei(system_cells, reference_cells, rounded=True),
        C.prf_coherence_aware(system_cells, reference_cells),
        C.prf_standard_all_cells(system_cells, reference_cells),
        C.prf_plain(C.cells_to_pairs(system_cells), C.cells_to_pairs(reference_cells)),
        C.prf_null_reference(C.cells_to_pairs(system_cells), C.cells_to_pairs(reference_cells), _p("a8b8")),
        C.prf_masked_split(C.cells_to_pairs(system_cells), C.cells_to_pairs(reference_cells), _p("a1b1"), _p("a2b2")),
        C.prf_logmap_oaei([], reference_cells),
    ]
    for block in blocks:
        _validate_global(block, label=f"test.{block['protocol']}")
