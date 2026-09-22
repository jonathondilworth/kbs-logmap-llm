"""Build strict leave-one-task-out, typed, query-RAG bundles.

Inputs are sealed harness alignments.  Evaluation references are never read:
positives are accepted initial LogMap equivalence mappings outside the union
of every campaign M_ask, and negatives are crossed only within one donor task.

CLI::

    python -m logmap_llm.oracle.rag.bundle PLAN.json --output-dir BUNDLES
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from logmap_llm.config.loader import load_config
from logmap_llm.constants import DEFAULT_OWLREADY2_CACHE_DIR, PAIRS_SEPARATOR
from logmap_llm.experiments.plan import (
    atomic_write_json, canonical_hash, load_manifest, sha256_file,
)
from logmap_llm.experiments.run import (
    BatchRunError,
    _verify_frozen_row,
    validate_completion,
)
from logmap_llm.oracle.rag.encoder import Encoder, ClsPooledEncoder
from logmap_llm.oracle.rag.types import EntityKind
from logmap_llm.pipeline.rag_fewshot import (
    _ANSWER_PAIRS,
    STRICT_PREBUILT_SELECTION_POLICY,
    rag_dataset_fingerprint,
)
from logmap_llm.utils.data import dedupe_m_ask_by_uri_pair

PLAN_KIND = "logmap-llm-rag-bundle-plan"
BUNDLE_KIND = "logmap-llm-prebuilt-few-shot"
SELECTION_POLICY = STRICT_PREBUILT_SELECTION_POLICY
PREPROCESSING_VERSION = "v1"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_REVISION = re.compile(r"^[0-9a-fA-F]{40,64}$")


class BundleError(RuntimeError):
    """Exact P,N,P,N treatment cannot be prepared."""


def _value(value: Any) -> str:
    return str(getattr(value, "value", value))


def _pair(source: str, target: str) -> tuple[str, str]:
    left, right = sorted((str(source), str(target)))
    return left, right


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BundleError(f"expected a JSON object: {path}")
    return value


def _internal(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise BundleError(f"batch path escapes its root: {relative!r}")
    path = (root / candidate).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise BundleError(f"batch path escapes its root: {relative!r}") from exc
    return path


def _alignment(path: Path, empty_ok: bool = False) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path, sep="|", header=None)
    except pd.errors.EmptyDataError:
        if empty_ok:
            return pd.DataFrame(columns=range(5))
        raise BundleError(f"empty initial alignment: {path}")
    if frame.shape[1] < 5:
        raise BundleError(f"expected five columns: {path}")
    frame = frame.iloc[:, :5].copy()
    for column in (0, 1, 2, 4):
        frame[column] = frame[column].map(str)
    if any(PAIRS_SEPARATOR in value for column in (0, 1) for value in frame[column]):
        raise BundleError(f"IRI contains reserved separator {PAIRS_SEPARATOR!r}: {path}")
    return frame


@dataclass(frozen=True)
class EncoderSpec:
    model: str
    revision: str
    device: str


@dataclass(frozen=True)
class PlanSpec:
    source_path: Path
    source_sha256: str
    encoder: EncoderSpec
    task_descriptors: tuple[dict[str, str], ...]
    receiver_task_ids: tuple[str, ...]


@dataclass(frozen=True)
class PromptIdentity:
    answer_format: str
    response_mode: str
    class_family: str
    property_family: str | None
    data_property_family: str | None
    instance_family: str | None
    bidirectional: bool

    def family_for(self, kind: EntityKind) -> str:
        family = {
            EntityKind.CLS: self.class_family,
            EntityKind.OPROP: self.property_family,
            EntityKind.DPROP: self.data_property_family or self.property_family,
            EntityKind.INST: self.instance_family,
        }[kind]
        if not family:
            raise BundleError(f"no prompt family for {kind.value}")
        return family


@dataclass(frozen=True)
class PairRow:
    source: str
    target: str
    relation: str
    kind: EntityKind

    @property
    def pair(self) -> tuple[str, str]:
        return _pair(self.source, self.target)

    @property
    def key(self) -> str:
        return f"{self.source}{PAIRS_SEPARATOR}{self.target}"


def load_plan(path: str | os.PathLike[str]) -> PlanSpec:
    source = Path(path).resolve()
    raw = _json(source)
    if set(raw) != {"schema", "kind", "encoder", "tasks", "receiver_task_ids"}:
        raise BundleError("unsupported bundle-plan fields")
    if raw["schema"] != 1 or raw["kind"] != PLAN_KIND:
        raise BundleError(f"plan must be schema 1, kind {PLAN_KIND!r}")
    enc = raw["encoder"]
    if not isinstance(enc, dict) or set(enc) != {"kind", "model", "revision", "device"}:
        raise BundleError("invalid encoder descriptor")
    if enc["kind"] != "cls_transformer" or not _REVISION.fullmatch(str(enc["revision"])):
        raise BundleError(
            "encoder must be the CLS-pooled transformer at an immutable hexadecimal revision"
        )
    if not all(isinstance(enc[key], str) and enc[key] for key in ("model", "device")):
        raise BundleError("encoder model/device must be non-empty strings")
    if not isinstance(raw["tasks"], list) or len(raw["tasks"]) < 2:
        raise BundleError("at least two task descriptors are required")

    tasks, ids = [], set()
    for index, task in enumerate(raw["tasks"]):
        if not isinstance(task, dict) or set(task) != {"task_id", "batch_dir", "alignment_id"}:
            raise BundleError(f"invalid tasks[{index}]")
        if not all(isinstance(task[key], str) and task[key] for key in task):
            raise BundleError(f"empty field in tasks[{index}]")
        task_id = task["task_id"]
        if not _ID.fullmatch(task_id) or task_id in ids or not _ID.fullmatch(task["alignment_id"]):
            raise BundleError(f"invalid/duplicate task or alignment ID: {task_id!r}")
        ids.add(task_id)
        batch = Path(task["batch_dir"]).expanduser()
        batch = batch if batch.is_absolute() else source.parent / batch
        tasks.append({**task, "batch_dir": str(batch.resolve())})
    receivers = raw["receiver_task_ids"]
    if not isinstance(receivers, list) or not receivers or len(receivers) != len(set(receivers)):
        raise BundleError("receiver_task_ids must be a non-empty unique list")
    if any(not isinstance(item, str) or item not in ids for item in receivers):
        raise BundleError("receiver_task_ids contains an unknown task")
    return PlanSpec(
        source, sha256_file(source),
        EncoderSpec(enc["model"], enc["revision"].lower(), enc["device"]),
        tuple(tasks), tuple(sorted(receivers)),
    )


def _rows(frame: pd.DataFrame, label: str) -> tuple[PairRow, ...]:
    result = []
    for index, row in frame.iterrows():
        try:
            kind = EntityKind.coerce(row.iloc[4])
        except ValueError as exc:
            raise BundleError(f"{label} row {index} has invalid type") from exc
        result.append(PairRow(str(row.iloc[0]), str(row.iloc[1]), str(row.iloc[2]), kind))
    return tuple(result)


def _prompt(cfg: Any) -> PromptIdentity:
    from logmap_llm.oracle.prompts import templates

    family = cfg.prompts.cls_usr_prompt_template_name
    if templates.registry.requires_siblings(family):
        raise BundleError(f"sibling-dependent template is unsupported: {family}")
    return PromptIdentity(
        _value(cfg.oracle.answer_format), _value(cfg.oracle.response_mode), family,
        cfg.prompts.prop_usr_prompt_template_name,
        cfg.prompts.dprop_usr_prompt_template_name,
        cfg.prompts.inst_usr_prompt_template_name,
        templates.registry.is_bidirectional(family),
    )


class _ProductionOntologyView:
    """Resolve and render against one donor's actual ontology pair."""

    def __init__(self, task: Any, source: Any, target: Any):
        self.task, self.source, self.target = task, source, target

    def _resolve(self, iri: str, ontology: Any, kind: EntityKind) -> tuple[Any, str]:
        from logmap_llm.ontology.object import resolve_entity_as

        # kind is an EntityKind, so these two branches are exhaustive.
        if kind == EntityKind.INST:
            entity, lane = resolve_entity_as(iri, ontology, "instance")
        else:
            entity, lane = resolve_entity_as(iri, ontology, kind.value)
        expected = {
            EntityKind.CLS: "class",
            EntityKind.OPROP: "property",
            EntityKind.DPROP: "property",
            EntityKind.INST: "instance",
        }[kind]
        if lane != expected:
            raise BundleError(f"{self.task.task_id}: {iri} resolved as {lane}, expected {kind.value}")
        return entity, lane

    def embed_text(self, row: PairRow) -> str:
        def names(iri: str, ontology: Any) -> str:
            entity, _ = self._resolve(iri, ontology, row.kind)
            values = getattr(entity, "get_preferred_names", lambda: ())()
            return " ".join(sorted(map(str, values))) or iri.split("#")[-1].split("/")[-1]
        return f"{names(row.source, self.source)} {names(row.target, self.target)}"

    def render(self, row: PairRow, prompt: PromptIdentity, *, reverse: bool = False) -> str:
        from logmap_llm.oracle.prompts import templates
        from logmap_llm.oracle.prompts.context import PromptContext

        # An immutable context cannot leak into a concurrent caller.
        context = PromptContext(
            answer_format=prompt.answer_format,
            response_mode=prompt.response_mode,
            ontology_domain=self.task.ontology_domain,
        )
        left = (row.target, self.target) if reverse else (row.source, self.source)
        right = (row.source, self.source) if reverse else (row.target, self.target)
        source, _ = self._resolve(*left, row.kind)
        target, _ = self._resolve(*right, row.kind)
        try:
            value = templates.registry.get(prompt.family_for(row.kind)).fn(
                source, target, ctx=context,
            )
        except Exception as exc:
            raise BundleError(f"{self.task.task_id}: cannot render {row.key}: {exc}") from exc
        if not isinstance(value, str) or not value.strip():
            raise BundleError(f"{self.task.task_id}: empty rendered prompt for {row.key}")
        return value


@dataclass
class SealedTask:
    task_id: str
    alignment_id: str
    cfg: Any
    mappings: pd.DataFrame
    m_ask: pd.DataFrame
    mappings_sha256: str
    m_ask_sha256: str
    complete_sha256: str
    config_sha256: str
    core_sha256: str
    prompt: PromptIdentity
    ontology_domain: str | None
    ontology_cache_dir: Path | None

    @contextlib.contextmanager
    def open_view(self):
        from logmap_llm.ontology.access import load_ontologies

        source, target = load_ontologies(
            self.cfg.alignmentTask.onto_source_filepath,
            self.cfg.alignmentTask.onto_target_filepath,
            cache_dir=str(self.ontology_cache_dir) if self.ontology_cache_dir else None,
            stub_import_iris=self.cfg.alignmentTask.stub_import_iris,
            vocabulary=self.cfg.alignmentTask.resolved_vocabulary,
        )
        try:
            yield _ProductionOntologyView(self, source, target)
        finally:
            for ontology in (source, target):
                close = getattr(getattr(ontology, "world", None), "close", None)
                if callable(close):
                    close()


def load_sealed_task(descriptor: dict[str, str], *, ontology_cache_dir=None) -> SealedTask:
    task_id, alignment_id = descriptor["task_id"], descriptor["alignment_id"]
    batch = Path(descriptor["batch_dir"]).resolve()
    manifest = load_manifest(batch)
    matches = [row for row in manifest["alignments"] if row["id"] == alignment_id]
    if len(matches) != 1 or matches[0]["task"] != task_id:
        raise BundleError(f"{task_id}: alignment is absent or belongs to another task")
    row = matches[0]
    try:
        config_path = _verify_frozen_row(batch, row)
    except (BatchRunError, OSError, ValueError) as exc:
        raise BundleError(f"{task_id}: frozen alignment input changed: {exc}") from exc
    cfg = load_config(config_path)
    if cfg.alignmentTask.task_name != task_id or cfg.evaluation.train_alignment_path:
        raise BundleError(f"{task_id}: task mismatch or forbidden training reference")
    fingerprints = row["input_fingerprints"]
    for field in ("onto_source_filepath", "onto_target_filepath"):
        name = f"alignmentTask.{field}"
        record = fingerprints.get(name)
        if not record:
            raise BundleError(f"{task_id}: alignment lacks {name} fingerprint")
        frozen = Path(record["path"])
        frozen = frozen if frozen.is_absolute() else batch / frozen
        if frozen.resolve() != Path(getattr(cfg.alignmentTask, field)).resolve():
            raise BundleError(f"{task_id}: {name} fingerprint/config mismatch")

    complete_paths = sorted((batch / "alignments" / alignment_id).glob("attempts/*/complete.json"))
    valid = [(path, value) for path in complete_paths
             if (value := validate_completion(batch, path)) is not None]
    if len(valid) != 1 or valid[0][0] != complete_paths[-1]:
        raise BundleError(f"{task_id}: no unambiguous latest completion")
    complete_path, complete = valid[0]
    if (complete["config_sha256"] != row["config_sha256"]
            or complete["core_sha256"] != manifest["source"]["core_sha256"]):
        raise BundleError(f"{task_id}: completion identity mismatch")
    records = {Path(item["path"]).name: item for item in complete["artifacts"]}
    names = (f"{task_id}-logmap_mappings.txt",
             f"{task_id}-logmap_mappings_to_ask_oracle_user_llm.txt")
    if any(name not in records for name in names):
        raise BundleError(f"{task_id}: completion lacks mappings/M_ask")
    mappings_record, mask_record = map(records.__getitem__, names)
    mappings = _alignment(_internal(batch, mappings_record["path"]))
    mask = dedupe_m_ask_by_uri_pair(_alignment(_internal(batch, mask_record["path"]), True))
    return SealedTask(
        task_id, alignment_id, cfg, mappings, mask,
        mappings_record["sha256"], mask_record["sha256"], sha256_file(complete_path),
        row["config_sha256"], manifest["source"]["core_sha256"], _prompt(cfg),
        cfg.alignmentTask.ontology_domain,
        Path(ontology_cache_dir).resolve() if ontology_cache_dir else None,
    )


@dataclass(frozen=True)
class _Anchor:
    id: str
    task: str
    alignment: str
    row: PairRow
    text: str
    mappings_sha: str
    completion_sha: str


@dataclass(frozen=True)
class _Query:
    row: PairRow
    text: str


@dataclass(frozen=True)
class _Demo:
    row: PairRow
    label: bool
    donor: str
    alignment: str
    id: str
    similarity: float
    mappings_sha: str
    completion_sha: str
    anchor_ids: tuple[str, ...]


def _prepare(tasks, receivers, excluded):
    anchors, queries = [], {receiver: [] for receiver in receivers}
    for task in sorted(tasks, key=lambda item: item.task_id):
        seen, rows = set(), []
        for row in _rows(task.mappings, f"{task.task_id} mappings"):
            if row.relation != "=":
                continue
            identity = row.kind, row.pair
            if row.pair not in excluded and identity not in seen:
                seen.add(identity)
                rows.append(row)
        qrows = _rows(task.m_ask, f"{task.task_id} M_ask") if task.task_id in receivers else ()
        with task.open_view() as view:
            for row in rows:
                identifier = f"{task.task_id}:{row.kind.value}:{_sha_text(row.source + chr(0) + row.target)[:20]}"
                anchors.append(_Anchor(identifier, task.task_id, task.alignment_id, row,
                                       view.embed_text(row), task.mappings_sha256, task.complete_sha256))
            queries[task.task_id] = [_Query(row, view.embed_text(row)) for row in qrows]
    return sorted(anchors, key=lambda anchor: anchor.id), queries


def _encode(encoder: Encoder, texts: list[str], label: str) -> np.ndarray:
    matrix = np.asarray(encoder.encode(texts), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(texts) or not np.isfinite(matrix).all():
        raise BundleError(f"invalid encoder matrix for {label}: {matrix.shape}")
    return matrix


def _select(query, receiver, anchors, matrix, vector, excluded, positives):
    eligible = [i for i, anchor in enumerate(anchors)
                if anchor.task != receiver and anchor.row.kind == query.row.kind]
    if len(eligible) < 2:
        raise BundleError(f"{receiver}/{query.row.key}: fewer than two held-out {query.row.kind.value} anchors")
    scores = matrix @ vector.reshape(-1)
    ranked = sorted(eligible, key=lambda i: (-float(scores[i]), anchors[i].id))

    def demo(anchor, index):
        return _Demo(anchor.row, True, anchor.task, anchor.alignment, anchor.id,
                     float(scores[index]), anchor.mappings_sha, anchor.completion_sha, (anchor.id,))
    selected_pos = [demo(anchors[index], index) for index in ranked[:2]]
    negatives, used = [], set()
    for first_index in ranked:
        for second_index in ranked:
            first, second = anchors[first_index], anchors[second_index]
            pair = _pair(first.row.source, second.row.target)
            if (first_index == second_index or first.task != second.task or first.row.source == second.row.target
                    or pair in excluded or pair in positives or pair in used):
                continue
            used.add(pair)
            row = PairRow(first.row.source, second.row.target, "=", query.row.kind)
            identifier = "neg:hard:" + _sha_text(first.id + chr(0) + second.id)[:24]
            negatives.append(_Demo(row, False, first.task, first.alignment, identifier,
                                   float(scores[first_index]), first.mappings_sha,
                                   first.completion_sha, (first.id, second.id)))
            if len(negatives) == 2:
                break
        if len(negatives) == 2:
            break
    if len(negatives) != 2:
        raise BundleError(f"{receiver}/{query.row.key}: cannot construct two legal same-donor negatives")
    return selected_pos[0], negatives[0], selected_pos[1], negatives[1]


def _render(task_map, prompt, selected, reverse):
    requested = {}
    for demos in selected.values():
        for demo in demos:
            requested.setdefault(demo.donor, set()).add(demo.row)
    output = {}
    for donor in sorted(requested):
        with task_map[donor].open_view() as view:
            for row in sorted(requested[donor], key=lambda item: (item.kind.value, item.source, item.target)):
                output[(donor, row.kind.value, row.source, row.target)] = view.render(row, prompt, reverse=reverse)
    return output


def _selected(demo, prompt, direction, rank):
    return {
        "example_id": demo.id, "label": demo.label,
        "source": "anchor" if demo.label else "constructed",
        "direction": direction, "rank": rank, "similarity": demo.similarity,
        "entity_type": demo.row.kind.value, "src_iri": demo.row.source,
        "tgt_iri": demo.row.target, "donor_task": demo.donor,
        "donor_alignment_id": demo.alignment, "donor_anchor_ids": list(demo.anchor_ids),
        "donor_mappings_sha256": demo.mappings_sha,
        "donor_completion_sha256": demo.completion_sha,
        "prompt_sha256": _sha_text(prompt),
    }


def generate_bundle_documents(plan: PlanSpec, tasks: Sequence[Any], encoder: Encoder):
    """Return one strict bundle document per receiver."""
    if encoder.repo != plan.encoder.model or encoder.revision.lower() != plan.encoder.revision:
        raise BundleError("loaded encoder does not match the pinned plan")
    encoder_pre = getattr(encoder, "preprocessing_version", "")
    if not encoder_pre:
        raise BundleError("encoder has no preprocessing identity")
    task_map = {task.task_id: task for task in tasks}
    planned = {task["task_id"] for task in plan.task_descriptors}
    if len(task_map) != len(tasks) or set(task_map) != planned:
        raise BundleError("loaded tasks do not exactly match the plan")

    excluded = {_pair(row.iloc[0], row.iloc[1]) for task in tasks for _, row in task.m_ask.iterrows()}
    excluded_text = "".join(f"{a}\t{b}\n" for a, b in sorted(excluded))
    excluded_sha = _sha_text(excluded_text)
    anchors, queries = _prepare(tasks, set(plan.receiver_task_ids), excluded)
    if not anchors:
        raise BundleError("campaign has no eligible anchors")
    matrix = _encode(encoder, [anchor.text for anchor in anchors], "anchors")
    known_positive = {anchor.row.pair for anchor in anchors}
    documents = {}

    for receiver_id in plan.receiver_task_ids:
        receiver, receiver_queries = task_map[receiver_id], queries[receiver_id]
        query_matrix = _encode(encoder, [query.text for query in receiver_queries], receiver_id)
        chosen = {query.row.key: _select(query, receiver_id, anchors, matrix, query_matrix[index],
                                         excluded, known_positive)
                  for index, query in enumerate(receiver_queries)}
        if len(chosen) != len(receiver_queries):
            raise BundleError(f"{receiver_id}: duplicate query keys")
        forward = _render(task_map, receiver.prompt, chosen, False) if chosen else {}
        reverse = (_render(task_map, receiver.prompt, chosen, True)
                   if chosen and receiver.prompt.bidirectional else {})
        try:
            positive_answer, negative_answer = _ANSWER_PAIRS[
                (receiver.prompt.answer_format, receiver.prompt.response_mode)]
        except KeyError as exc:
            raise BundleError(f"{receiver_id}: unsupported answer format") from exc
        dataset_sha = rag_dataset_fingerprint(receiver.mappings, receiver.m_ask, None)
        by_key = {query.row.key: query for query in receiver_queries}
        corpus_hashes = {
            kind: canonical_hash([(a.id, a.task, _sha_text(a.text)) for a in anchors
                                  if a.task != receiver_id and a.row.kind == kind])
            for kind in EntityKind
        }
        examples, traces = {}, {}
        for base_key in sorted(chosen):
            query = by_key[base_key]
            for direction, rendered in (["forward", forward], ["reverse", reverse]):
                if direction == "reverse" and not receiver.prompt.bidirectional:
                    continue
                key = base_key if direction == "forward" else f"{base_key}{PAIRS_SEPARATOR}REVERSE"
                pairs, records = [], []
                for rank, demo in enumerate(chosen[base_key]):
                    prompt = rendered[(demo.donor, demo.row.kind.value, demo.row.source, demo.row.target)]
                    pairs.append([prompt, positive_answer if demo.label else negative_answer])
                    records.append(_selected(demo, prompt, direction, rank))
                if [record["label"] for record in records] != [True, False, True, False]:
                    raise BundleError(f"{receiver_id}/{key}: P,N,P,N invariant failed")
                corpus_hash = corpus_hashes[query.row.kind]
                examples[key] = pairs
                traces[key] = {
                    "requested_mode": "query_rag", "effective_mode": "query_rag",
                    "requested_k": 4, "effective_k": 4,
                    "entity_type": query.row.kind.value, "relation": query.row.relation,
                    "prompt_family": receiver.prompt.family_for(query.row.kind),
                    "answer_format": receiver.prompt.answer_format, "selected": records,
                    "exclusions": [{"count": len(excluded), "reason": "campaign M_ask union"}],
                    "fallback_reason": None, "corpus_hash": corpus_hash,
                    "index_hash": canonical_hash((corpus_hash, plan.encoder.model,
                                                   plan.encoder.revision, encoder_pre, SELECTION_POLICY)),
                    "encoder_repo": plan.encoder.model, "encoder_revision": plan.encoder.revision,
                    "encoder_kind": "cls_transformer", "encoder_device": plan.encoder.device,
                    "preprocessing_version": PREPROCESSING_VERSION,
                    "encoder_preprocessing_version": encoder_pre,
                    "selection_policy": SELECTION_POLICY, "dataset_sha": dataset_sha,
                    "campaign_m_ask_sha256": excluded_sha, "query_direction": direction,
                }
        expected = {query.row.key for query in receiver_queries}
        if receiver.prompt.bidirectional:
            expected |= {f"{key}{PAIRS_SEPARATOR}REVERSE" for key in tuple(expected)}
        if set(examples) != expected or set(traces) != expected:
            raise BundleError(f"{receiver_id}: incomplete prompt-key coverage")
        documents[receiver_id] = {
            "schema": 1, "kind": BUNDLE_KIND,
            "binding": {
                "receiver_task": receiver_id, "dataset_sha": dataset_sha,
                "m_ask_sha256": receiver.m_ask_sha256, "few_shot_k": 4,
                "few_shot_strategy": "query-rag", "encoder_kind": "cls_transformer",
                "encoder_model": plan.encoder.model, "encoder_revision": plan.encoder.revision,
                "preprocessing_version": PREPROCESSING_VERSION,
                "answer_format": receiver.prompt.answer_format,
                "response_mode": receiver.prompt.response_mode,
                "prompt_family": receiver.prompt.class_family,
                "property_prompt_family": receiver.prompt.property_family,
                "data_property_prompt_family": receiver.prompt.data_property_family,
                "instance_prompt_family": receiver.prompt.instance_family,
                "bidirectional": receiver.prompt.bidirectional,
                "selection_policy": SELECTION_POLICY,
                "encoder_preprocessing_version": encoder_pre,
                "receiver_alignment_id": receiver.alignment_id,
                "receiver_config_sha256": receiver.config_sha256,
                "receiver_core_sha256": receiver.core_sha256,
                "campaign_m_ask_sha256": excluded_sha,
                "campaign_m_ask_pairs": len(excluded),
                "donor_task_ids": sorted(set(task_map) - {receiver_id}),
                "plan_sha256": plan.source_sha256,
            },
            "examples": examples, "traces": traces,
        }
    return documents


def write_bundle_documents(documents, output_dir) -> list[Path]:
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for task_id, document in sorted(documents.items()):
        path = destination / f"{task_id}.json"
        if path.exists() and _json(path) != document:
            raise BundleError(f"refusing to overwrite a different bundle: {path}")
        if not path.exists():
            atomic_write_json(path, document)
        paths.append(path)
    return paths


def build_from_plan(plan_path, output_dir, *, ontology_cache_dir=DEFAULT_OWLREADY2_CACHE_DIR,
                    encoder: Encoder | None = None) -> list[Path]:
    plan = load_plan(plan_path)
    cache = Path(ontology_cache_dir).expanduser().resolve() if ontology_cache_dir else None
    tasks = [load_sealed_task(task, ontology_cache_dir=cache) for task in plan.task_descriptors]
    encoder = encoder or ClsPooledEncoder(plan.encoder.model, plan.encoder.revision, plan.encoder.device)
    return write_bundle_documents(generate_bundle_documents(plan, tasks, encoder), output_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ontology-cache-dir", default=str(DEFAULT_OWLREADY2_CACHE_DIR))
    args = parser.parse_args(argv)
    try:
        paths = build_from_plan(args.plan, args.output_dir,
                                ontology_cache_dir=args.ontology_cache_dir or None)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(*paths, sep="\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
