"""
Regression tests for bidirectional prompt building.

`build_oracle_user_prompts_bidirectional` must build a forward and reverse prompt
for every M_ask row regardless of its `relation` column: the relation is LogMap's
hypothesis, not ground truth, and mutual subsumption is an equivalence test
(A ⊑ B ∧ B ⊑ A ⟺ A ≡ B), so non-'=' rows get the same candidates as any other
equivalence template. Prompt text must also be deterministic: name selection over
a set is sorted, never iteration-order.
"""
from __future__ import annotations

import pandas as pd
import pytest

from logmap_llm.oracle.prompts.context import PromptContext

from logmap_llm.oracle.prompts import templates as T
from logmap_llm.oracle.prompts.formatting import get_single_name


class _FakeEntity:
    """Minimal stand-in for OntologyEntryAttr: the builder only needs it to construct."""

    def __init__(self, uri, onto):
        if "PROP" in str(uri):                      # a non-class row, as owlready2 would fail on
            raise ValueError(f"not a class: {uri}")
        self.uri = uri


def _m_ask(rows):
    return pd.DataFrame(rows, columns=["src", "tgt", "relation"])


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(T, "OntologyEntryAttr", _FakeEntity)
    # the resolver binds a prompt context, so the stub takes it too
    monkeypatch.setattr(T, "get_oracle_user_prompt_template_function",
                        lambda name, ctx=None: (lambda s, t: f"IS {s.uri} A {t.uri}?"))


def test_non_equivalence_rows_are_ASKED_not_silently_dropped():
    """Dropping non-'=' rows would force-reject genuine reference mappings and
    hide them from the confusion matrix; every candidate must be asked."""
    df = _m_ask([("A", "B", "="), ("C", "D", ">"), ("E", "F", "<")])
    prompts, n_built, _ = T.build_oracle_user_prompts_bidirectional(
        "sub_parents_synonyms", None, None, df, OA_source=object(), OA_target=object(),
        ctx=PromptContext())

    assert n_built == 3, f"all 3 candidates must be asked; only {n_built} were"
    for src, tgt in (("A", "B"), ("C", "D"), ("E", "F")):
        base = src + T.PAIRS_SEPARATOR + tgt
        assert base in prompts, f"forward prompt missing for the {src}->{tgt} candidate"
        assert base + T.PAIRS_SEPARATOR + "REVERSE" in prompts, f"reverse prompt missing for {src}->{tgt}"
    assert len(prompts) == 6, "3 candidates x (forward + reverse) = 6 prompts"


def test_forward_is_src_to_tgt_and_reverse_is_the_swap():
    """The headline decomposition (fwd vs rev) is meaningless if the two are transposed."""
    df = _m_ask([("A", "B", "=")])
    prompts, _, _ = T.build_oracle_user_prompts_bidirectional(
        "sub_parents_synonyms", None, None, df, OA_source=object(), OA_target=object(),
        ctx=PromptContext())
    base = "A" + T.PAIRS_SEPARATOR + "B"
    assert prompts[base] == "IS A A B?"
    assert prompts[base + T.PAIRS_SEPARATOR + "REVERSE"] == "IS B A A?"


def test_non_class_rows_are_recorded_not_counted_as_class_candidates():
    """OPROP/DPROP rows cannot be built as classes; they must not inflate the class count."""
    df = _m_ask([("A", "B", "="), ("PROP1", "PROP2", "=")])
    prompts, n_built, _ = T.build_oracle_user_prompts_bidirectional(
        "sub_parents_synonyms", None, None, df, OA_source=object(), OA_target=object(),
        ctx=PromptContext())
    assert n_built == 1, "only the class row is a class candidate"
    assert len(prompts) == 2


def test_PT1_single_name_is_deterministic_over_a_set():
    """A set has no order; `next(iter(...))` would vary the prompt text between runs."""
    s = {"zeta", "alpha", "mu"}
    assert get_single_name(s) == "alpha"
    assert all(get_single_name(set(s)) == "alpha" for _ in range(50))
