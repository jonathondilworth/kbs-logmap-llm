"""
Pooled collective anchors. A prebuilt few-shot bundle is leave-one-task-out by default (every
demonstration comes from another task); a plan with anchor_pool = "pooled" lets the receiver's
own anchors compete as well. These tests cover the plan, the selector, the generated document
and the loader's acceptance rules. CPU-only: ontologies are replaced by a fake view.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from logmap_llm.config.schema import FewShotConfig
from logmap_llm.oracle.rag import bundle
from logmap_llm.oracle.rag.bundle import BundleError, PairRow, _Anchor, _Query, _select, load_plan
from logmap_llm.oracle.rag.types import EntityKind
from logmap_llm.pipeline.rag_fewshot import (
    DEFAULT_PREBUILT_ANCHOR_POOL,
    POOLED_PREBUILT_SELECTION_POLICY,
    STRICT_PREBUILT_SELECTION_POLICY,
    load_prebuilt_few_shot_bundle,
)

REVISION = "0" * 40
CLS = EntityKind.CLS


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------

def _plan_file(tmp_path, **extra):
    plan = {
        "schema": 1, "kind": bundle.PLAN_KIND,
        "encoder": {"kind": "cls_transformer", "model": "m", "revision": REVISION, "device": "cpu"},
        "tasks": [{"task_id": t, "batch_dir": f"batch-{t}", "alignment_id": "a1"}
                  for t in ("recv", "other")],
        "receiver_task_ids": ["recv"],
        **extra,
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")
    return path


def test_plan_defaults_to_leave_one_task_out(tmp_path):
    assert load_plan(_plan_file(tmp_path)).anchor_pool == DEFAULT_PREBUILT_ANCHOR_POOL


def test_plan_accepts_pooled(tmp_path):
    assert load_plan(_plan_file(tmp_path, anchor_pool="pooled")).anchor_pool == "pooled"


@pytest.mark.parametrize("extra", [{"anchor_pool": "everything"}, {"pool": "pooled"}])
def test_plan_rejects_unknown_pool_or_key(tmp_path, extra):
    with pytest.raises(BundleError):
        load_plan(_plan_file(tmp_path, **extra))


# --------------------------------------------------------------------------
# selector
# --------------------------------------------------------------------------

def _anchor(task, index, src, tgt):
    row = PairRow(src, tgt, "=", CLS)
    return _Anchor(f"{task}:{index}", task, "a1", row, f"{src} {tgt}", "msha", "csha")


def _selector_fixture():
    anchors = [
        _anchor("recv", 0, "r0", "s0"), _anchor("recv", 1, "r1", "s1"),
        _anchor("other", 2, "o2", "p2"), _anchor("other", 3, "o3", "p3"),
    ]
    # the receiver's own anchors are the most similar to the query
    matrix = np.array([[1.0, 0.0], [0.9, 0.0], [0.5, 0.0], [0.4, 0.0]])
    vector = np.array([1.0, 0.0])
    query = _Query(PairRow("q", "z", "=", CLS), "q z")
    positives = {anchor.row.pair for anchor in anchors}
    return query, anchors, matrix, vector, positives


def test_leave_one_task_out_never_selects_receiver_anchors():
    query, anchors, matrix, vector, positives = _selector_fixture()
    demos = _select(query, "recv", anchors, matrix, vector, set(), positives)
    assert [demo.label for demo in demos] == [True, False, True, False]
    assert {demo.donor for demo in demos} == {"other"}


def test_pooled_selects_receiver_anchors_when_they_rank_higher():
    query, anchors, matrix, vector, positives = _selector_fixture()
    demos = _select(query, "recv", anchors, matrix, vector, set(), positives, pooled=True)
    assert [demo.label for demo in demos] == [True, False, True, False]
    assert [demo.donor for demo in demos] == ["recv"] * 4
    assert [demo.row.pair for demo in demos[::2]] == [("r0", "s0"), ("r1", "s1")]
    # negatives are still crossed within one donor task and are never a known positive
    assert all(demo.row.pair not in positives for demo in demos[1::2])


# --------------------------------------------------------------------------
# generated document -> loader round trip
# --------------------------------------------------------------------------

class _FakeView:
    def embed_text(self, row):
        return f"{row.source} {row.target}"

    def render(self, row, prompt, *, reverse=False):
        return f"{'rev' if reverse else 'fwd'}:{row.source}->{row.target}"


class _FakeEncoder:
    repo = "m"
    revision = REVISION
    preprocessing_version = "pre-v1"

    def encode(self, texts):
        # receiver anchors and the query share a direction; other tasks are orthogonal
        return np.array([[1.0, 0.0] if t[0] in "rq" else [0.0, 1.0] for t in texts])


def _task(task_id, mappings_rows, m_ask_rows, m_ask_sha256="asha"):
    @contextlib.contextmanager
    def open_view():
        yield _FakeView()

    return SimpleNamespace(
        task_id=task_id, alignment_id="a1",
        mappings=pd.DataFrame(mappings_rows),
        m_ask=pd.DataFrame(m_ask_rows, columns=range(5)),
        mappings_sha256=f"msha-{task_id}", m_ask_sha256=m_ask_sha256,
        complete_sha256=f"csha-{task_id}", config_sha256="cfg", core_sha256="core",
        prompt=bundle.PromptIdentity("true_false", "structured", "synonyms_only",
                                     None, None, None, False),
        open_view=open_view,
    )


def _campaign(tmp_path, anchor_pool):
    m_ask_path = tmp_path / "recv-m_ask.txt"
    m_ask_path.write_text("q1|z1|=|0.5|CLS\n", encoding="utf-8")
    recv = _task(
        "recv",
        [["r0", "s0", "=", 1.0, "CLS"], ["r1", "s1", "=", 1.0, "CLS"], ["q1", "z1", "=", 0.5, "CLS"]],
        [["q1", "z1", "=", 0.5, "CLS"]],
        m_ask_sha256=hashlib.sha256(m_ask_path.read_bytes()).hexdigest(),
    )
    other = _task("other", [["o2", "p2", "=", 1.0, "CLS"], ["o3", "p3", "=", 1.0, "CLS"]], [])
    plan = bundle.PlanSpec(
        Path("plan.json"), "psha", bundle.EncoderSpec("m", REVISION, "cpu"),
        tuple({"task_id": t.task_id, "batch_dir": "b", "alignment_id": "a1"} for t in (recv, other)),
        ("recv",), anchor_pool,
    )
    document = bundle.generate_bundle_documents(plan, [recv, other], _FakeEncoder())["recv"]
    bundle_path = tmp_path / f"{anchor_pool}.json"
    bundle_path.write_text(json.dumps(document), encoding="utf-8")
    loader_kwargs = dict(
        mappings=recv.mappings, m_ask_df=recv.m_ask, m_ask_path=str(m_ask_path),
        expected_query_keys=["q1|z1"], receiver_task="recv", train_tsv_path=None, k=4,
        strategy="query-rag", encoder_kind="cls_transformer", encoder_model="m",
        encoder_revision=REVISION, answer_format="true_false", response_mode="structured",
        prompt_family="synonyms_only", property_prompt_family=None,
        data_property_prompt_family=None, instance_prompt_family=None, bidirectional=False,
    )
    return document, str(bundle_path), loader_kwargs


def test_pooled_document_records_policy_and_receiver_donors(tmp_path):
    document, _, _ = _campaign(tmp_path, "pooled")
    binding = document["binding"]
    assert binding["anchor_pool"] == "pooled"
    assert binding["selection_policy"] == POOLED_PREBUILT_SELECTION_POLICY
    assert binding["donor_task_ids"] == ["other", "recv"]
    selected = document["traces"]["q1|z1"]["selected"]
    assert [record["label"] for record in selected] == [True, False, True, False]
    assert {record["donor_task"] for record in selected} == {"recv"}
    assert document["traces"]["q1|z1"]["selection_policy"] == POOLED_PREBUILT_SELECTION_POLICY


def test_leave_one_task_out_document_is_unchanged(tmp_path):
    document, _, _ = _campaign(tmp_path, DEFAULT_PREBUILT_ANCHOR_POOL)
    binding = document["binding"]
    assert binding["anchor_pool"] == DEFAULT_PREBUILT_ANCHOR_POOL
    assert binding["selection_policy"] == STRICT_PREBUILT_SELECTION_POLICY
    assert binding["donor_task_ids"] == ["other"]
    assert {r["donor_task"] for r in document["traces"]["q1|z1"]["selected"]} == {"other"}


def test_loader_round_trip_pooled(tmp_path):
    _, path, kwargs = _campaign(tmp_path, "pooled")
    examples, traces = load_prebuilt_few_shot_bundle(path, anchor_pool="pooled", **kwargs)
    assert set(examples) == set(traces) == {"q1|z1"}
    assert [pair[1] for pair in examples["q1|z1"]] == [
        '{"answer": true}', '{"answer": false}', '{"answer": true}', '{"answer": false}']


def test_loader_round_trip_leave_one_task_out_by_default(tmp_path):
    _, path, kwargs = _campaign(tmp_path, DEFAULT_PREBUILT_ANCHOR_POOL)
    examples, _ = load_prebuilt_few_shot_bundle(path, **kwargs)
    assert len(examples["q1|z1"]) == 4


def test_loader_refuses_a_pooled_bundle_under_leave_one_task_out(tmp_path):
    _, path, kwargs = _campaign(tmp_path, "pooled")
    with pytest.raises(ValueError, match="selection_policy"):
        load_prebuilt_few_shot_bundle(path, **kwargs)


def test_loader_refuses_a_leave_one_task_out_bundle_under_pooled(tmp_path):
    _, path, kwargs = _campaign(tmp_path, DEFAULT_PREBUILT_ANCHOR_POOL)
    with pytest.raises(ValueError, match="selection_policy"):
        load_prebuilt_few_shot_bundle(path, anchor_pool="pooled", **kwargs)


def test_loader_still_refuses_receiver_donor_under_leave_one_task_out(tmp_path):
    # a leave-one-task-out policy label with a receiver donor is an inconsistent bundle
    document, path, kwargs = _campaign(tmp_path, DEFAULT_PREBUILT_ANCHOR_POOL)
    for record in document["traces"]["q1|z1"]["selected"]:
        record["donor_task"] = "recv"
    Path(path).write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="different from receiver_task"):
        load_prebuilt_few_shot_bundle(path, **kwargs)


def test_loader_rejects_unknown_pool(tmp_path):
    _, path, kwargs = _campaign(tmp_path, DEFAULT_PREBUILT_ANCHOR_POOL)
    with pytest.raises(ValueError, match="Unknown prebuilt anchor pool"):
        load_prebuilt_few_shot_bundle(path, anchor_pool="everything", **kwargs)


# --------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------

def _bundle_cfg(**extra):
    return FewShotConfig(
        few_shot_k=4, few_shot_negative_strategy="query-rag", rag_encoder_kind="hashing",
        prebuilt_few_shot_bundle_path="bundle.json", **extra,
    )


def test_schema_pool_requires_bundle_path():
    with pytest.raises(ValueError, match="prebuilt_anchor_pool"):
        FewShotConfig(prebuilt_anchor_pool="pooled")


def test_schema_pool_accepted_with_bundle_and_absent_from_default_dumps():
    assert _bundle_cfg(prebuilt_anchor_pool="pooled").prebuilt_anchor_pool == "pooled"
    # unset stays out of exclude_none dumps, so frozen job configs keep their identity
    assert "prebuilt_anchor_pool" not in FewShotConfig().model_dump(exclude_none=True)
    assert "prebuilt_anchor_pool" not in _bundle_cfg().model_dump(exclude_none=True)


def test_legacy_sapbert_encoder_kind_is_migrated():
    # the July campaign's frozen configs, whose sealed alignments feed the bundle builder
    cfg = FewShotConfig(few_shot_k=4, few_shot_negative_strategy="query-rag",
                        rag_encoder_kind="sapbert", rag_encoder_model="m", rag_encoder_revision="r")
    assert cfg.rag_encoder_kind == "cls_transformer"
