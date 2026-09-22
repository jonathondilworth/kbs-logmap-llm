"""
logmap_llm.pipeline.model_selection

Self-supervised automatic model selection (`[model_selection] automatic = true`).

LogMap's anchors, the initial-alignment equivalence mappings it did not escalate to M_ask, are
taken to be correct. They can therefore be turned into oracle questions with known answers
without any reference alignment: `build_ranking_set` samples up to `max_anchors` of them and
pairs each with one constructed negative (the anchor's source with another anchor's target).
Stage two renders their prompts with the very same templates as the M_ask prompts, every
permitted model configuration is asked them (orchestration.select_model), and
`rank_candidates` orders the candidates by how many they answered correctly; the best one
becomes the run's oracle. pandas-only, so it is testable without the JVM.
"""
from __future__ import annotations

import json
import random

import numpy as np
import pandas as pd

from logmap_llm.constants import M_ASK_COLUMNS, PAIRS_SEPARATOR
from logmap_llm.oracle.rag.types import EntityKind
from logmap_llm.utils.io import atomic_json_write_strict

RANKING_LABEL_COLUMN = "ranking_label"  # the pseudo-label, carried next to the M_ask columns
_RANKING_SCHEMA = 1


def _pair_key(row) -> frozenset:
    return frozenset({str(row.iloc[0]), str(row.iloc[1])})


def build_ranking_set(
    mappings: pd.DataFrame, m_ask_df: pd.DataFrame, max_anchors: int, seed: int,
) -> pd.DataFrame:
    """
    M_ask-shaped frame of up to `max_anchors` anchors (label True), each followed by one
    constructed negative (label False). Anchors are the pseudo-positives the RAG corpus also
    uses (oracle/rag/pipeline_adapter.build_typed_corpus_from_anchors): equivalence rows of the
    initial alignment outside M_ask with a recognised entity type. A negative keeps the anchor's
    source and takes the target of another anchor of the same kind; it is never a pair LogMap
    proposed itself (initial alignment or M_ask). Deterministic for a given seed.
    """
    asked = {_pair_key(row) for _, row in m_ask_df.iterrows()}
    proposed = {_pair_key(row) for _, row in mappings.iterrows()} | asked
    anchors: list[tuple[str, str, float, str]] = []
    for _, row in mappings.iterrows():
        if str(row.iloc[2]).strip() != "=" or _pair_key(row) in asked:
            continue
        try:
            kind = EntityKind.coerce(str(row.iloc[4]).strip() if len(row) > 4 else "CLS").value
        except ValueError:
            continue  # UNKNO / unexpected type: never mistype an anchor
        anchors.append((str(row.iloc[0]), str(row.iloc[1]), float(row.iloc[3]), kind))

    rng = random.Random(seed)
    sample = anchors if len(anchors) <= max_anchors else rng.sample(anchors, max_anchors)
    targets_by_kind: dict[str, list[str]] = {}
    for _src, tgt, _conf, kind in anchors:
        targets_by_kind.setdefault(kind, []).append(tgt)

    rows = []
    for src, tgt, conf, kind in sample:
        rows.append([src, tgt, "=", conf, kind, True])
        # sorted: set order depends on hash seeding and must not reach the RNG
        donors = sorted({t for t in targets_by_kind[kind]
                         if t != tgt and frozenset({src, t}) not in proposed})
        if donors:
            rows.append([src, rng.choice(donors), "=", conf, kind, False])
    return pd.DataFrame(rows, columns=[*M_ASK_COLUMNS, RANKING_LABEL_COLUMN])


def write_ranking_artifact(
    path, ranking_df: pd.DataFrame, prompts: dict, *, max_anchors: int, seed: int,
) -> pd.DataFrame:
    """Persist the questions stage two rendered. Rows without a forward prompt (unresolvable
    entities) are dropped so consultation coverage stays exact; returns the kept rows."""
    keys = ranking_df.iloc[:, 0].astype(str) + PAIRS_SEPARATOR + ranking_df.iloc[:, 1].astype(str)
    kept = ranking_df[keys.isin(prompts)].reset_index(drop=True)
    payload = {
        "schema": _RANKING_SCHEMA, "max_anchors": max_anchors, "seed": seed,
        "columns": list(kept.columns),
        "questions": json.loads(kept.to_json(orient="values")),
        "prompts": prompts,
    }
    atomic_json_write_strict(path, payload, indent=2)
    return kept


def load_ranking_artifact(path) -> tuple[pd.DataFrame, dict]:
    with open(path, encoding="utf-8") as fp:
        payload = json.load(fp)
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != _RANKING_SCHEMA
        or not all(key in payload for key in ("columns", "questions", "prompts"))
    ):
        raise ValueError(f"unsupported model-ranking artifact: {path}")
    return pd.DataFrame(payload["questions"], columns=payload["columns"]), payload["prompts"]


def score_candidate(index: int, oracle_cfg, ranking_df: pd.DataFrame, predictions) -> dict:
    """One ranking record. `predictions` is the consultation frame (the ranking rows plus
    Oracle_prediction) or None when the campaign aborted under the failure tolerance."""
    record = {
        "index": index,
        "model_name": oracle_cfg.model_name,
        "base_url": oracle_cfg.base_url,
        "interaction_style": str(getattr(oracle_cfg.interaction_style, "value",
                                         oracle_cfg.interaction_style)),
        "asked": int(len(ranking_df)),
        "correct": 0,
        "errors": int(len(ranking_df)),
        "status": "aborted",
    }
    if predictions is None:
        return record
    answered = [
        (bool(verdict), bool(label))
        for verdict, label in zip(predictions["Oracle_prediction"], predictions[RANKING_LABEL_COLUMN])
        if isinstance(verdict, (bool, np.bool_))  # "error" / "skipped" are not answers
    ]
    record.update(
        correct=sum(verdict == label for verdict, label in answered),
        errors=len(predictions) - len(answered),
        status="scored",
    )
    return record


def rank_candidates(records: list[dict]) -> list[dict]:
    """Best first: most correct answers, then fewest unanswered, then the configured order."""
    return sorted(records, key=lambda record: (-record["correct"], record["errors"], record["index"]))
