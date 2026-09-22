"""
logmap_llm.pipeline.rag_fewshot

Owlready2-dependent glue that wires the query-specific RAG retriever (oracle/rag) into stage_two.
It builds the two OntologyAccess-backed callables the retriever needs — embed_text_fn (similarity
text) and render_fn (renders an example with the same per-kind template as the live query, so
CLS/OPROP/DPROP/INST queries get correctly-typed examples) — runs per-query retrieval over the
M_ask, and returns:

    per_query : {m_ask_key -> [(user_prompt, assistant_answer), ...]}   -> persisted to few_shot_json()
    traces    : {m_ask_key -> RetrievalTrace.to_dict()}                 -> persisted for provenance

STATIC_* modes yield the same query-agnostic examples, preserving the ablation baselines. This
module needs owlready2 (resolve_entity + OntologyAccess), so it lives on the pipeline path, not
in the CPU-only oracle/rag package.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from enum import Enum
from typing import Callable, Optional

import pandas as pd

from logmap_llm.constants import PAIRS_SEPARATOR
from logmap_llm.oracle.prompts.routing import resolve_pair_lane
from logmap_llm.oracle.rag import (
    build_retriever_from_pipeline,
    query_mappings_from_m_ask,
    m_ask_exclusion_keys,
    EntityKind,
    FallbackPolicy,
    HashingEncoder,
    ClsPooledEncoder,
    SbertEncoder,
)


# Kept dependency-free so the RAG adapter can be imported in CPU-only tooling.
# These are the same four wire-format pairs used by oracle.prompts.few_shot.
_ANSWER_PAIRS = {
    ("true_false", "structured"): ('{"answer": true}', '{"answer": false}'),
    ("true_false", "plain"): ("True", "False"),
    ("yes_no", "structured"): ('{"answer": "Yes"}', '{"answer": "No"}'),
    ("yes_no", "plain"): ("Yes", "No"),
}

_PREBUILT_BUNDLE_KIND = "logmap-llm-prebuilt-few-shot"
_PREBUILT_BUNDLE_SCHEMA = 1
_RAG_PREPROCESSING_VERSION = "v1"
STRICT_PREBUILT_SELECTION_POLICY = "strict-loo-typed-equivalence-pnpn-v2"
POOLED_PREBUILT_SELECTION_POLICY = "strict-pooled-typed-equivalence-pnpn-v2"
# few_shot.prebuilt_anchor_pool -> the selection policy a bundle must have been built under.
# leave-one-task-out: every demonstration comes from another task (the ISWC campaigns);
# pooled: the receiver's own anchors are eligible as well.
PREBUILT_SELECTION_POLICIES = {
    "leave-one-task-out": STRICT_PREBUILT_SELECTION_POLICY,
    "pooled": POOLED_PREBUILT_SELECTION_POLICY,
}
DEFAULT_PREBUILT_ANCHOR_POOL = "leave-one-task-out"


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wire_scalar(value):
    """Return an enum's JSON/TOML scalar while leaving ordinary values unchanged."""
    return value.value if isinstance(value, Enum) else value


def validate_prebuilt_few_shot_examples(
    examples,
    *,
    expected_query_keys,
    k: int,
    answer_format: str,
    response_mode: str,
) -> None:
    """Validate the strict per-query JSON handoff used by prebuilt bundles."""
    answer_format = _wire_scalar(answer_format)
    response_mode = _wire_scalar(response_mode)
    if not isinstance(examples, dict):
        raise ValueError(
            "Prebuilt few-shot examples must be a JSON object keyed by prompt query"
        )
    expected_keys = set(expected_query_keys)
    example_keys = set(examples)
    if example_keys != expected_keys:
        missing = sorted(expected_keys - example_keys)
        extra = sorted(example_keys - expected_keys)
        raise ValueError(
            "Prebuilt few-shot bundle examples do not match prompt keys "
            f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
        )
    if k != 4:
        raise ValueError("Strict prebuilt few-shot examples require few_shot_k = 4")
    try:
        positive_answer, negative_answer = _ANSWER_PAIRS[(answer_format, response_mode)]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported (answer_format, response_mode) = "
            f"({answer_format!r}, {response_mode!r})"
        ) from exc
    expected_answers = [
        positive_answer,
        negative_answer,
        positive_answer,
        negative_answer,
    ]
    for query_key in sorted(expected_keys):
        query_examples = examples[query_key]
        if not isinstance(query_examples, list) or len(query_examples) != k:
            raise ValueError(
                f"Prebuilt few-shot query {query_key!r} must contain exactly {k} examples"
            )
        for index, (pair, expected_answer) in enumerate(
            zip(query_examples, expected_answers, strict=True)
        ):
            if (
                not isinstance(pair, list)
                or len(pair) != 2
                or not all(isinstance(value, str) for value in pair)
                or not pair[0]
            ):
                raise ValueError(
                    f"Prebuilt few-shot query {query_key!r} example {index} "
                    "must be [non-empty user prompt, assistant answer]"
                )
            if pair[1] != expected_answer:
                raise ValueError(
                    f"Prebuilt few-shot query {query_key!r} example {index} "
                    "does not follow P,N,P,N with the configured exact answers"
                )


def load_prebuilt_few_shot_bundle(
    path: str,
    *,
    mappings: pd.DataFrame,
    m_ask_df: pd.DataFrame,
    m_ask_path: str,
    expected_query_keys,
    receiver_task: str,
    train_tsv_path: Optional[str],
    k: int,
    strategy: str,
    encoder_kind: str,
    encoder_model: str,
    encoder_revision: Optional[str],
    answer_format: str,
    response_mode: str,
    prompt_family: str,
    property_prompt_family: Optional[str],
    data_property_prompt_family: Optional[str],
    instance_prompt_family: Optional[str],
    bidirectional: bool,
    anchor_pool: Optional[str] = None,
) -> tuple[dict, dict]:
    """Load one strictly bound, already-rendered per-query RAG bundle.

    A prebuilt bundle is a scientific input, not a cache.  It is accepted only
    when it names the exact alignment/M_ask dataset and exact requested RAG and
    prompt configuration.  The published artifacts retain the ordinary
    ``few_shot_examples.json``/``rag_traces.json`` wire format, so consultation
    does not need a second code path.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prebuilt few-shot bundle does not exist: {path}")
    if not os.path.isfile(m_ask_path):
        raise FileNotFoundError(f"M_ask file does not exist: {m_ask_path}")
    if k != 4:
        raise ValueError("Strict prebuilt few-shot bundles require few_shot_k = 4")
    anchor_pool = anchor_pool or DEFAULT_PREBUILT_ANCHOR_POOL
    try:
        selection_policy = PREBUILT_SELECTION_POLICIES[anchor_pool]
    except KeyError as exc:
        raise ValueError(f"Unknown prebuilt anchor pool {anchor_pool!r}") from exc
    answer_format = _wire_scalar(answer_format)
    response_mode = _wire_scalar(response_mode)

    try:
        with open(path, "r", encoding="utf-8") as stream:
            bundle = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read prebuilt few-shot bundle {path}: {exc}") from exc
    if not isinstance(bundle, dict):
        raise ValueError("Prebuilt few-shot bundle root must be a JSON object")
    allowed_root_keys = {"schema", "kind", "binding", "examples", "traces", "metadata"}
    unknown_root_keys = sorted(set(bundle) - allowed_root_keys)
    if unknown_root_keys:
        raise ValueError(
            f"Prebuilt few-shot bundle has unsupported root keys: {unknown_root_keys!r}"
        )
    if "metadata" in bundle and not isinstance(bundle["metadata"], dict):
        raise ValueError("Prebuilt few-shot bundle metadata must be a JSON object")
    if (
        type(bundle.get("schema")) is not int
        or bundle["schema"] != _PREBUILT_BUNDLE_SCHEMA
    ):
        raise ValueError(
            f"Prebuilt few-shot bundle schema must be {_PREBUILT_BUNDLE_SCHEMA}"
        )
    if bundle.get("kind") != _PREBUILT_BUNDLE_KIND:
        raise ValueError(
            f"Prebuilt few-shot bundle kind must be {_PREBUILT_BUNDLE_KIND!r}"
        )

    binding = bundle.get("binding")
    examples = bundle.get("examples")
    traces = bundle.get("traces")
    if not isinstance(binding, dict):
        raise ValueError("Prebuilt few-shot bundle binding must be a JSON object")
    if not isinstance(examples, dict) or not isinstance(traces, dict):
        raise ValueError("Prebuilt few-shot bundle examples and traces must be JSON objects")

    dataset_sha = rag_dataset_fingerprint(mappings, m_ask_df, train_tsv_path)
    expected_binding = {
        "receiver_task": receiver_task,
        "dataset_sha": dataset_sha,
        "m_ask_sha256": _sha256_file(m_ask_path),
        "few_shot_k": k,
        "few_shot_strategy": strategy,
        "encoder_kind": encoder_kind,
        "encoder_model": encoder_model,
        "encoder_revision": encoder_revision,
        "preprocessing_version": _RAG_PREPROCESSING_VERSION,
        "answer_format": answer_format,
        "response_mode": response_mode,
        "prompt_family": prompt_family,
        "property_prompt_family": property_prompt_family,
        "data_property_prompt_family": data_property_prompt_family,
        "instance_prompt_family": instance_prompt_family,
        "bidirectional": bidirectional,
        "selection_policy": selection_policy,
    }
    for field, expected in expected_binding.items():
        if field not in binding:
            raise ValueError(f"Prebuilt few-shot bundle binding is missing {field!r}")
        if type(binding[field]) is not type(expected) or binding[field] != expected:
            raise ValueError(
                f"Prebuilt few-shot bundle binding mismatch for {field}: "
                f"expected {expected!r}, got {binding[field]!r}"
            )

    expected_entity_types = {}
    for _, row in m_ask_df.iterrows():
        query_key = f"{row.iloc[0]}{PAIRS_SEPARATOR}{row.iloc[1]}"
        try:
            expected_entity_types[query_key] = EntityKind.coerce(row.iloc[4]).value
        except (IndexError, ValueError) as exc:
            raise ValueError(
                f"M_ask query {query_key!r} has no valid entity type"
            ) from exc
    expected_keys = set(expected_entity_types)
    if bidirectional:
        for key, entity_type in tuple(expected_entity_types.items()):
            reverse_key = f"{key}{PAIRS_SEPARATOR}REVERSE"
            expected_keys.add(reverse_key)
            expected_entity_types[reverse_key] = entity_type
    prompt_keys = set(expected_query_keys)
    if prompt_keys != expected_keys:
        missing = sorted(expected_keys - prompt_keys)
        extra = sorted(prompt_keys - expected_keys)
        raise ValueError(
            "Live prompt keys do not exactly cover M_ask "
            f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
        )
    trace_keys = set(traces)
    validate_prebuilt_few_shot_examples(
        examples,
        expected_query_keys=expected_keys,
        k=k,
        answer_format=answer_format,
        response_mode=response_mode,
    )
    if trace_keys != expected_keys:
        missing = sorted(expected_keys - trace_keys)
        extra = sorted(trace_keys - expected_keys)
        raise ValueError(
            "Prebuilt few-shot bundle traces do not match prompt keys "
            f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
        )

    expected_labels = [True, False, True, False]

    for query_key in sorted(expected_keys):
        query_examples = examples[query_key]
        trace = traces[query_key]
        if not isinstance(trace, dict):
            raise ValueError(f"Prebuilt few-shot trace {query_key!r} must be a JSON object")
        required_trace = {
            "requested_mode": "query_rag",
            "effective_mode": "query_rag",
            "requested_k": 4,
            "effective_k": 4,
            "dataset_sha": dataset_sha,
            "encoder_kind": encoder_kind,
            "encoder_repo": encoder_model,
            "encoder_revision": encoder_revision,
            "preprocessing_version": _RAG_PREPROCESSING_VERSION,
            "selection_policy": selection_policy,
            "fallback_reason": None,
        }
        for field, expected in required_trace.items():
            if field not in trace or type(trace[field]) is not type(expected) or trace[field] != expected:
                raise ValueError(
                    f"Prebuilt few-shot trace {query_key!r} mismatch for {field}: "
                    f"expected {expected!r}, got {trace.get(field)!r}"
                )
        selected = trace.get("selected")
        if not isinstance(selected, list) or len(selected) != 4:
            raise ValueError(
                f"Prebuilt few-shot trace {query_key!r} must select exactly four examples"
            )
        labels = []
        for index, (record, pair) in enumerate(
            zip(selected, query_examples, strict=True)
        ):
            if not isinstance(record, dict) or type(record.get("label")) is not bool:
                raise ValueError(
                    f"Prebuilt few-shot trace {query_key!r} selected[{index}] "
                    "must have a boolean label"
                )
            donor_task = record.get("donor_task")
            # Under 'pooled' the receiver's own anchors are legal donors.
            if (
                not isinstance(donor_task, str)
                or not donor_task.strip()
                or (anchor_pool != "pooled" and donor_task == receiver_task)
            ):
                raise ValueError(
                    f"Prebuilt few-shot trace {query_key!r} selected[{index}] must "
                    "name a non-empty donor_task"
                    + ("" if anchor_pool == "pooled" else " different from receiver_task")
                )
            if record.get("entity_type") != expected_entity_types[query_key]:
                raise ValueError(
                    f"Prebuilt few-shot trace {query_key!r} selected[{index}] "
                    f"entity_type must be {expected_entity_types[query_key]!r}"
                )
            expected_prompt_sha = hashlib.sha256(pair[0].encode("utf-8")).hexdigest()
            if record.get("prompt_sha256") != expected_prompt_sha:
                raise ValueError(
                    f"Prebuilt few-shot trace {query_key!r} selected[{index}] "
                    "prompt_sha256 does not match the rendered demonstration prompt"
                )
            labels.append(record["label"])
        if labels != expected_labels:
            raise ValueError(
                f"Prebuilt few-shot trace {query_key!r} labels must be P,N,P,N"
            )

    return examples, traces


def _local_name(iri: str) -> str:
    return iri.split("#")[-1].split("/")[-1]


def _resolve_from_either(iri: str, primary, secondary):
    """Resolve an IRI, trying the expected ontology first, then the other.

    Bidirectional demonstrations reverse the ontology positions; trying both
    keeps reverse examples ontology-grounded instead of silently degrading to
    local-name-only prompts.
    """
    # Keep owlready2 off the module import path so encoder/fingerprint helpers
    # remain usable in light CPU environments.
    from logmap_llm.ontology.object import resolve_entity

    first_error = None
    for ontology in (primary, secondary):
        try:
            return resolve_entity(iri, ontology)
        except Exception as exc:
            if first_error is None:
                first_error = exc
    assert first_error is not None
    raise first_error


def make_embed_text_fn(OA_source, OA_target) -> Callable:
    """
    Similarity text for a (src, tgt) pair = concatenated preferred labels (fallback: local names).
    Never raises — an unresolvable IRI degrades to its local name.
    """
    def _names(iri: str, primary, secondary) -> str:
        try:
            entity, _ = _resolve_from_either(iri, primary, secondary)
            fn = getattr(entity, "get_preferred_names", None)
            if fn:
                names = fn()
                if names:
                    return " ".join(sorted(str(n) for n in names))
        except Exception:
            pass
        return _local_name(iri)

    def embed_text_fn(src_iri, tgt_iri, kind):
        return (
            f"{_names(src_iri, OA_source, OA_target)} "
            f"{_names(tgt_iri, OA_target, OA_source)}"
        )

    return embed_text_fn


def _normalised_names(entity) -> set[str]:
    """Casefolded, whitespace-collapsed preferred names and synonyms of an entity."""
    names: set[str] = set()
    for accessor in ("get_preferred_names", "get_synonyms"):
        function = getattr(entity, accessor, None)
        if function is None:
            continue
        try:
            values = function() or ()
        except Exception:
            continue
        for value in values:
            collapsed = " ".join(str(value).split()).casefold()
            if collapsed:
                names.add(collapsed)
    return names


def _entity_iri(entity) -> str | None:
    iri = getattr(entity, "iri", None)
    if iri:
        return str(iri)
    annotation = getattr(entity, "annotation", None)
    if isinstance(annotation, dict) and annotation.get("uri"):
        return str(annotation["uri"])
    uri = getattr(entity, "uri", None)
    return str(uri) if uri else None


def make_sibling_fn(OA_source, OA_target, selector, *, max_candidates=None) -> Callable:
    """Build the ``sibling_fn`` the RagRetriever injects for ``paired-sibling-v2``.

    Contract (see ``oracle/rag/retriever.SiblingFn``):

        sibling_fn(target_iri, kind, max_count)
            -> (ranked [candidate dict], reason_when_empty)

    where each candidate is ``{"iri", "score", "type_specificity", "rule"}``. It lives
    here, not in ``oracle/rag``, because it needs owlready2 — the retriever is
    deliberately owlready2-free.

    Three properties it must have:

    * Never raises: an unresolvable IRI returns ``[]`` with a machine-readable reason,
      which the retriever records on the donor-rule negative it builds instead —
      failure is recorded, never silent.
    * Always ranks: ``force_rank=True`` stops ``select_siblings`` short-circuiting to
      alphabetical order when a class has few siblings, which would run the
      alphanumeric condition under the configured semantic condition's name.
    * Drops candidates sharing a normalised preferred name or synonym with the true
      target — those are very likely equivalent, which would make the "negative" true.
      The retriever's eligibility rules cannot see labels, so the filter lives here.
    """
    def sibling_fn(target_iri: str, kind, max_count: int):
        try:
            entity, resolved = _resolve_from_either(target_iri, OA_target, OA_source)
        except Exception as exc:
            return [], f"unresolvable-target:{type(exc).__name__}"

        # Which sibling notion applies is decided by how the target resolves, not by the
        # query's LogMap lane, and the two disagree in practice: OPROP/DPROP targets can
        # resolve as InstanceEntity (dbkwik predicate URIs are themselves typed subjects
        # in the graph), so the property lanes run the shared-type instance rule, which
        # yields better near-misses than the declared-domain property rule on such
        # corpora. The resolved rule is recorded rather than forced back to the lane.
        rule = {"class": "class", "instance": "instance", "property": "property"}.get(
            str(resolved), "class")

        try:
            ranked = selector.select_siblings(
                entity, max_count=max_count, max_candidates=max_candidates, force_rank=True,
            )
        except NotImplementedError:
            return [], f"sibling-selection-unsupported:{kind.value}"
        except Exception as exc:
            return [], f"sibling-lookup-failed:{type(exc).__name__}"

        if not ranked:
            return [], "no-siblings"

        target_names = _normalised_names(entity)
        candidates: list = []
        dropped_by_label = 0
        dropped_uninformative = 0
        for sibling, score in ranked:
            iri = _entity_iri(sibling)
            if not iri or iri == target_iri:
                continue
            if target_names and (_normalised_names(sibling) & target_names):
                dropped_by_label += 1
                continue
            # IndexedInstance carries how discriminating the shared type was; classes and
            # properties have no analogue, so it stays None for those rules.
            specificity = getattr(sibling, "type_specificity", None)
            if specificity == "uninformative":
                # A candidate whose only shared type is owl:Thing (or a peer) is an
                # arbitrary resource, not a near-miss; the ranked-donor rule gives a
                # better negative instead, and the swap is recorded per negative and
                # per lane rather than being silent.
                dropped_uninformative += 1
                continue
            candidates.append({
                "iri": iri,
                "score": float(score),
                "type_specificity": specificity,
                "rule": rule,
            })

        if not candidates:
            if dropped_uninformative:
                return [], "type-bucket-uninformative"
            if dropped_by_label:
                return [], "sibling-label-identical"
            return [], "no-siblings"
        return candidates, None

    return sibling_fn


def make_render_fn(OA_source, OA_target, cls_fn, property_fn, data_property_fn, instance_fn) -> Callable:
    """
    Render an example prompt with the same per-kind template the live query uses: resolve the
    pair to entities, pick the lane from the authoritative kind (carried in payload) via
    resolve_pair_lane, and dispatch to the matching bound template fn. A DPROP example is
    rendered with the data-property template (or the datatype-aware object-property template),
    so the datatype range surfaces in the example exactly as in the live query.
    """
    def render_fn(src_iri, tgt_iri, payload):
        try:
            src_e, src_t = _resolve_from_either(src_iri, OA_source, OA_target)
            tgt_e, tgt_t = _resolve_from_either(tgt_iri, OA_target, OA_source)
        except Exception:
            # a corpus example that no longer resolves -> minimal label prompt (never abort a query)
            return f'Are "{_local_name(src_iri)}" and "{_local_name(tgt_iri)}" equivalent?'
        kind = payload if isinstance(payload, str) else None   # constructed negatives pass a (src,tgt) tuple
        lane = resolve_pair_lane(kind, src_t, tgt_t)
        if lane == "instance" and instance_fn is not None:
            return instance_fn(src_e, tgt_e)
        if lane == "property":
            is_dprop = (kind == "DPROP") or getattr(src_e, "is_data_property", False)
            fn = data_property_fn if (is_dprop and data_property_fn is not None) else property_fn
            if fn is not None:
                return fn(src_e, tgt_e)
        return cls_fn(src_e, tgt_e)

    return render_fn


def build_rag_encoder(
    *,
    kind: str,
    model: Optional[str] = None,
    revision: Optional[str] = None,
    device: Optional[str] = None,
    max_length: Optional[int] = None,
):
    """Construct the explicitly configured RAG encoder.

    Hashing is retained as an intentional, named offline baseline. Semantic
    retrieval requires a pinned model revision so a later model update cannot
    silently change an experiment or its cache identity.
    """
    normalised = str(kind).strip().lower()
    if normalised == "hashing":
        return HashingEncoder()
    if normalised not in {"cls_transformer", "sbert"}:
        raise ValueError("rag_encoder_kind must be 'cls_transformer', 'sbert', or 'hashing'")
    if not model or not str(model).strip():
        raise ValueError("rag_encoder_model is required for a semantic encoder")
    if not revision or not str(revision).strip():
        raise ValueError(
            "rag_encoder_revision must pin a model commit/tag for a semantic encoder"
        )
    if normalised == "sbert":
        return SbertEncoder(
            model_name_or_path=str(model), revision=str(revision), device=device,
            max_length=128 if max_length is None else int(max_length),
        )
    return ClsPooledEncoder(
        model_name_or_path=str(model),
        revision=str(revision),
        device=device,
        max_length=64 if max_length is None else int(max_length),
    )


def rag_dataset_fingerprint(
    mappings: pd.DataFrame,
    m_ask_df: pd.DataFrame,
    train_tsv_path: Optional[str] = None,
) -> str:
    """Return a content fingerprint for the inputs that form the RAG corpus.

    The task label is metadata, not identity. We hash canonicalised alignment
    rows plus the exact authorised training-alignment bytes, when present. Row
    sorting makes the identifier stable when LogMap emits an equivalent set in
    a different order.
    """
    h = hashlib.sha256()

    def _value(value) -> str:
        try:
            if bool(pd.isna(value)):
                return "<NA>"
        except (TypeError, ValueError):
            pass
        return str(value)

    for label, frame in (("initial_alignment", mappings), ("m_ask", m_ask_df)):
        h.update(label.encode("utf-8"))
        h.update(b"\x00")
        rows = [tuple(_value(value) for value in row) for row in frame.itertuples(index=False, name=None)]
        for row in sorted(rows):
            h.update(json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            h.update(b"\n")

    if train_tsv_path:
        train_path = os.fspath(train_tsv_path)
        if not os.path.isfile(train_path):
            raise FileNotFoundError(f"RAG training alignment does not exist: {train_path}")
        h.update(b"train_alignment\x00")
        with open(train_path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)

    return h.hexdigest()


def _directional_trace(trace: dict, direction: str) -> dict:
    """Create one trace per actual prompt key without sharing mutable state."""
    copied = dict(trace)
    copied["query_direction"] = direction
    selected = []
    for item in trace.get("selected", ()):
        record = {**item, "direction": direction}
        similarity = record.get("similarity")
        if isinstance(similarity, float) and not math.isfinite(similarity):
            record["similarity"] = None
        selected.append(record)
    copied["selected"] = tuple(selected)
    return copied


def build_query_specific_few_shot(
    *,
    mappings: pd.DataFrame,
    m_ask_df: pd.DataFrame,
    OA_source,
    OA_target,
    cls_fn: Callable,
    property_fn: Optional[Callable],
    data_property_fn: Optional[Callable],
    instance_fn: Optional[Callable],
    strategy: str,
    k: int,
    seed: int,
    bidirectional: bool,
    answer_format: str,
    response_mode: str,
    prompt_family: str,
    negative_layout: str,
    sibling_fn: Optional[Callable] = None,
    sibling_strategy: str = "",
    sibling_encoder_revision: str = "",
    train_tsv_path: Optional[str] = None,
    dataset_sha: str = "",
    cache_dir: Optional[str] = None,
    encoder=None,
    encoder_kind: Optional[str] = None,
    encoder_model: Optional[str] = None,
    encoder_revision: Optional[str] = None,
    encoder_device: Optional[str] = None,
    encoder_max_length: Optional[int] = None,
    failure_policy: str = "record_zero_shot",
    pairs_separator: str = "|",
) -> tuple[dict, dict]:
    """Build query-specific few-shot examples + retrieval traces for every M_ask candidate."""
    try:
        answer_pos, answer_neg = _ANSWER_PAIRS[(answer_format, response_mode)]
    except KeyError:
        raise ValueError(f"Unsupported (answer_format, response_mode) = ({answer_format!r}, {response_mode!r})")

    embed_text_fn = make_embed_text_fn(OA_source, OA_target)
    render_fn = make_render_fn(OA_source, OA_target, cls_fn, property_fn, data_property_fn, instance_fn)

    if failure_policy not in {"error", "record_zero_shot"}:
        raise ValueError("failure_policy must be 'error' or 'record_zero_shot'")
    if encoder is None and not encoder_kind:
        # Deliberate baseline default for CPU tooling and tests; the pipeline caller
        # (stage_two) always passes the configured kind explicitly. Resolving it here
        # makes the trace record 'hashing' instead of a class name, so the recorded
        # encoder identity matches the config vocabulary.
        encoder_kind = "hashing"
    effective_encoder = encoder or build_rag_encoder(
        kind=encoder_kind,
        model=encoder_model,
        revision=encoder_revision,
        device=encoder_device,
        max_length=encoder_max_length,
    )
    fallback = FallbackPolicy.strict() if failure_policy == "error" else FallbackPolicy()

    retriever = build_retriever_from_pipeline(
        initial_alignment_df=mappings, m_ask_df=m_ask_df,
        encoder=effective_encoder,
        render_fn=render_fn, embed_text_fn=embed_text_fn,
        # Direction is handled below at the actual prompt-key boundary. This
        # keeps `k` intuitive: each forward/reverse oracle request receives k
        # examples, rather than sharing k across two requests.
        strategy=strategy, k=k, seed=seed, bidirectional=False,
        negative_layout=negative_layout,
        answer_pos=answer_pos, answer_neg=answer_neg,
        dataset_sha=dataset_sha, train_tsv_path=train_tsv_path, cache_dir=cache_dir,
        payload_fn=lambda s, t, kind: kind.value,   # carry the authoritative kind to render_fn
        fallback=fallback,
        sibling_fn=sibling_fn,
        sibling_strategy=sibling_strategy,
        sibling_encoder_revision=sibling_encoder_revision,
    )
    retriever.warmup()
    excl = m_ask_exclusion_keys(m_ask_df)

    per_query: dict = {}
    traces: dict = {}
    for key, qm in query_mappings_from_m_ask(m_ask_df, embed_text_fn, pairs_separator=pairs_separator):
        result = retriever.retrieve(
            qm, qm.kind, "=", prompt_family, k, corpus_id=dataset_sha, exclude_keys=excl,
        )
        per_query[key] = result.message_pairs()
        trace = result.trace.to_dict()
        trace.update({
            "dataset_sha": dataset_sha,
            "encoder_kind": encoder_kind or type(effective_encoder).__name__,
            "encoder_device": getattr(effective_encoder, "device", "cpu"),
            "encoder_preprocessing_version": getattr(
                effective_encoder, "preprocessing_version", ""
            ),
        })
        index = retriever._indexes.get(qm.kind)
        # 'absent' = no index exists for this kind at all (empty-pool zero-shot fallback).
        if index is None:
            trace["index_cache_status"] = "absent"
        elif index.rebuilt_reason is None:
            trace["index_cache_status"] = "hit"
        else:
            trace["index_cache_status"] = "rebuilt"
        trace["index_rebuilt_reason"] = (
            None if index is None else index.rebuilt_reason
        )
        traces[key] = _directional_trace(trace, "forward")

        if bidirectional:
            reverse_key = f"{key}{pairs_separator}REVERSE"
            # Reuse the same selected evidence in the opposite orientation so
            # the two logical directions differ only in direction, not in which
            # demonstrations happened to be retrieved.
            reverse_pairs = [
                (
                    render_fn(example.tgt_iri, example.src_iri, example.kind.value),
                    example.answer_text,
                )
                for example in result.examples
            ]
            per_query[reverse_key] = reverse_pairs
            reverse_trace = _directional_trace(trace, "reverse")
            # prompt_sha256 binds a trace record to the rendered demonstration
            # (bundle.py recomputes it per direction), so rebind it and the token
            # cost to the re-rendered reverse prompts.
            counter = retriever.token_counter
            reverse_trace["selected"] = tuple(
                {
                    **record,
                    "prompt_sha256": hashlib.sha256(
                        reverse_prompt.encode("utf-8")
                    ).hexdigest(),
                    "tokens": counter.count(reverse_prompt) + counter.count(answer_text),
                }
                for record, (reverse_prompt, answer_text) in zip(
                    reverse_trace["selected"], reverse_pairs
                )
            )
            traces[reverse_key] = reverse_trace

    return per_query, traces
