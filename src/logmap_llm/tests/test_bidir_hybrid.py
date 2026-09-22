"""
Hybrid bidirectional lanes.

In bidirectional (mutual-subsumption) mode only CLASS candidates have a subsumption reading.
Rows that LogMap types OPROP/DPROP/INST must (a) be built with the forward property/instance
templates when those are configured, (b) be exempt from the reverse-coverage guard, and (c) be
folded into the per-row verdict unchanged instead of being marked "skipped" (a reject).
Also pins the property template's quoting: the label is quoted once.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import pandas as pd
import pytest

from logmap_llm.oracle import consultation as C
from logmap_llm.oracle.prompts import templates as T
from logmap_llm.oracle.prompts.context import PromptContext


class _FakeClass:
    def __init__(self, uri, onto):
        if "/property/" in str(uri) or "/resource/" in str(uri):
            raise ValueError(f"not a class: {uri}")
        self.uri = uri


def _m_ask(rows):
    return pd.DataFrame(rows, columns=["src", "tgt", "relation", "confidence", "entityType"])


def _ctx():
    return PromptContext.__new__(PromptContext)


@pytest.fixture(autouse=True)
def _patch(monkeypatch):
    monkeypatch.setattr(T, "OntologyEntryAttr", _FakeClass)
    monkeypatch.setattr(T, "get_oracle_user_prompt_template_function",
                        lambda name, ctx=None: (lambda s, t: f"IS {s.uri} A {t.uri}?"))
    # the forward builder is exercised through a stub that echoes which rows it received
    monkeypatch.setattr(T, "build_oracle_user_prompts",
                        lambda oupt, sp, tp, df, **kw: {f"{r.iloc[0]}|{r.iloc[1]}": f"FWD {r.iloc[4]}" for _, r in df.iterrows()})


def test_typed_non_class_rows_take_the_forward_lane():
    df = _m_ask([
        ["http://a#Paper", "http://b#Article", "=", 0.5, "CLS"],
        ["http://a/property/year", "http://b/property/year", "=", 0.9, "DPROP"],
        ["http://a/resource/x", "http://b/resource/x", "=", 0.7, "INST"],
    ])
    prompts, n_equiv, _ = T.build_oracle_user_prompts_bidirectional(
        "sub_parents_synonyms", "s.owl", "t.owl", df, OA_source=object(), OA_target=object(),
        property_prompt_name="prop_domain_range", instance_prompt_name="inst_full_context_entropy",
        ctx=_ctx())
    assert n_equiv == 1
    assert prompts["http://a#Paper|http://b#Article"].startswith("IS http://a#Paper")
    assert prompts["http://a#Paper|http://b#Article|REVERSE"].startswith("IS http://b#Article")
    assert prompts["http://a/property/year|http://b/property/year"] == "FWD DPROP"
    assert prompts["http://a/resource/x|http://b/resource/x"] == "FWD INST"
    assert "http://a/property/year|http://b/property/year|REVERSE" not in prompts


def test_without_lane_templates_non_class_rows_are_still_skipped():
    df = _m_ask([
        ["http://a#Paper", "http://b#Article", "=", 0.5, "CLS"],
        ["http://a/property/year", "http://b/property/year", "=", 0.9, "DPROP"],
    ])
    prompts, n_equiv, _ = T.build_oracle_user_prompts_bidirectional(
        "sub_parents_synonyms", "s.owl", "t.owl", df, OA_source=object(), OA_target=object(), ctx=_ctx())
    assert n_equiv == 1 and len(prompts) == 2


def test_reverse_coverage_exempts_forward_only_keys():
    prompts = {"c|d": "x", "c|d|REVERSE": "y", "p|q": "z"}
    with pytest.raises(ValueError):
        C._require_reverse_prompt_coverage(prompts)
    C._require_reverse_prompt_coverage(prompts, forward_only_keys={"p|q"})
    with pytest.raises(ValueError):   # a reverse key on an exempt candidate is an error
        C._require_reverse_prompt_coverage({**prompts, "p|q|REVERSE": "w"}, forward_only_keys={"p|q"})


def _usage(i, o):
    return SimpleNamespace(input_tokens=i, output_tokens=o)


def test_aggregation_ands_classes_and_passes_lanes_through():
    df = _m_ask([
        ["c1", "d1", "=", 0.5, "CLS"],
        ["c2", "d2", "=", 0.5, "CLS"],
        ["p", "q", "=", 0.9, "OPROP"],
        ["i", "j", "=", 0.7, "INST"],
        ["c3", "d3", "=", 0.5, "CLS"],
    ])
    results = {
        "c1|d1": (True, 0.9, _usage(10, 1)), "c1|d1|REVERSE": (True, 0.8, _usage(11, 1)),
        "c2|d2": (True, 0.9, _usage(10, 1)), "c2|d2|REVERSE": (False, 0.95, _usage(11, 1)),
        "p|q": (False, 0.99, _usage(20, 2)),
        "i|j": (True, 0.6, _usage(30, 3)),
        # c3|d3 has no results at all -> skipped
    }
    agg = C._aggregate_bidirectional(results, df, C._forward_only_keys(df))
    assert agg["pred"] == [True, False, False, True, "skipped"]
    assert agg["conf"][0] == pytest.approx(0.8) and agg["in"][0] == 21
    assert agg["fwd"][2] is False and agg["rev"][2] == "n/a" and math.isnan(agg["rev_conf"][2])
    assert agg["in"][3] == 30 and agg["out"][3] == 3
    assert agg["skipped"] == 1


def test_property_template_quotes_label_once():
    class _Prop:
        is_data_property = True
        def get_preferred_names(self):
            return {"email"}
        def get_domain_names(self):
            return {"Person"}
        def get_range_names(self):
            return {"str"}
    ctx = SimpleNamespace(domain_preamble="We have two entities.", response_instruction="Respond.")
    text = T.oupt_prop_domain_range(_Prop(), _Prop(), ctx=ctx)
    assert 'The first property is "email" on "Person" which has data values of type "str".' in text
    assert '""' not in text
