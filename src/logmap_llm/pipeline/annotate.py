"""
logmap_llm.pipeline.annotate

Annotate mode (22 Sep 2026, LOCAL_CHANGES.md §7): the oracle's verdict on every M_ask
candidate, written next to the candidate itself so a set of mappings can be validated by
the LLM oracle without any LogMap step (`pipeline.align_ontologies = "external"` feeds an
external mapping file in as M_ask; `pipeline.stop_after_consultation = true` ends the run
after Step 3). The files are written after every consultation, in every mode that produces
a predictions DataFrame (consult or reuse), so a normal LogMapLLM run also documents what
the oracle said about each candidate.

Two files, in M_ask order (the order of `<task>-logmap_mappings_to_ask_oracle_user_llm.txt`,
duplicates by (source, target) collapsed to the first occurrence as the pipeline does):

* `<task>-<template>-annotated.txt` — LogMap pipe format: the M_ask row's own five columns
  (`source|target|relation|confidence|entityType`, verbatim) followed by `LLM_annotation`
  (`True`, `False`, `ERROR` for a failed consultation, `SKIPPED` for a candidate without a
  verdict) and, for bidirectional (mutual-subsumption) runs, `LLM_forward_subsumption` and
  `LLM_reverse_subsumption` (the two directional verdicts the AND was taken over; `n/a` on
  the forward-only property/instance lanes).
* `<task>-<template>-annotated.tsv` — the same columns tab-separated with a header
  (`source target relation confidence type LLM_annotation [LLM_forward_subsumption
  LLM_reverse_subsumption] LLM_confidence`) plus a last `LLM_confidence` column.

**How LLM_confidence is computed.** The oracle is never asked for a confidence. The value is
the pipeline's `Oracle_confidence`: `oracle/consultation.py::calculate_logprobs_confidence`
reads the token logprobs returned with the completion (`request_logprobs = true`), finds the
first token that is one of the answer tokens (`true`/`false` or `yes`/`no`, case-insensitive,
leading whitespace ignored) and returns exp(max logprob) of the *parsed* answer's token
among that position's `top_logprobs`, clipped to [0, 1] — i.e. the model's probability of
the verdict it gave, not of the more likely of the two options. It is `nan` when the
endpoint returned no logprobs (e.g. Gemini on OpenRouter, or `request_logprobs = false`) or
when no answer token was found; in bidirectional mode it is the minimum of the forward and
reverse confidences. `nan` is written literally.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from logmap_llm.constants import (
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
    COL_RELATION,
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    PAIRS_SEPARATOR,
    EntityType,
)
from logmap_llm.evaluation.conventions import RELATIONS, normalise_relation
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.utils.io import atomic_write_text_strict

_RDF_SUFFIXES = frozenset({".rdf", ".xml", ".owl"})
_RDF_RESOURCE = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
_HEADER_FIRST_FIELDS = frozenset({"srcentity", "source", "src", "entity1", "source_entity_uri"})
_ENTITY_TYPES = frozenset(member.value for member in EntityType)
_ANNOTATION_ERROR = "ERROR"
_ANNOTATION_SKIPPED = "SKIPPED"
TSV_BASE_HEADER = ("source", "target", "relation", "confidence", "type", "LLM_annotation")
TSV_BIDIRECTIONAL_HEADER = ("LLM_forward_subsumption", "LLM_reverse_subsumption")
TSV_CONFIDENCE_HEADER = "LLM_confidence"


###
# External mappings -> M_ask rows
###


def _rdf_rows(path: Path) -> list[tuple[str, str, str, str, str]]:
    rows = []
    for element in ET.parse(path).getroot().iter():
        entity1 = entity2 = None
        relation = measure = None
        for child in element:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag == "entity1":
                entity1 = child.get(_RDF_RESOURCE) or child.get("resource")
            elif tag == "entity2":
                entity2 = child.get(_RDF_RESOURCE) or child.get("resource")
            elif tag == "relation":
                relation = child.text
            elif tag == "measure":
                measure = (child.text or "").strip()
        if entity1 is not None and entity2 is not None:
            rows.append((entity1, entity2, normalise_relation(relation), measure or "1.0", EntityType.CLASS.value))
    return rows


def load_external_mappings(path: str | Path) -> list[tuple[str, str, str, str, str]]:
    """
    The rows of an external mapping file as `(source, target, relation, confidence, entityType)`
    text tuples. Pipe `.txt` (LogMap format) and TSV files keep their columns verbatim: the
    relation is the third column when it is one of `= < > ?` (else `=`, and a numeric third
    column is read as the confidence), the confidence the next column (default `1.0`), the
    entity type the fifth column when it is a LogMap type symbol (default `CLS`). OAEI
    Alignment RDF cells give relation and measure; their entity type is `CLS`. A header line
    is skipped; blank lines are ignored; duplicate (source, target) rows are kept here and
    collapsed by the caller.
    """
    path = Path(path)
    if path.suffix.lower() in _RDF_SUFFIXES:
        return _rdf_rows(path)
    sep = PAIRS_SEPARATOR if path.suffix.lower() == ".txt" else "\t"
    rows = []
    first = True
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            fields = [field.strip() for field in line.split(sep)]
            if first:
                first = False
                if fields[0].lower() in _HEADER_FIRST_FIELDS:
                    continue
            if len(fields) < 2:
                raise ValueError(f"{path}: a mapping line needs at least two {sep!r}-separated fields: {line!r}")
            relation, confidence, entity_type = "=", "1.0", EntityType.CLASS.value
            rest = fields[2:]
            if rest and rest[0] in RELATIONS:
                relation = rest[0]
                rest = rest[1:]
            if rest:
                try:
                    float(rest[0])
                    confidence = rest[0]
                    rest = rest[1:]
                except ValueError:
                    pass
            if rest and rest[0] in _ENTITY_TYPES:
                entity_type = rest[0]
            rows.append((fields[0], fields[1], relation, confidence, entity_type))
    return rows


def external_m_ask_frame(rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """The M_ask DataFrame (the five `M_ASK_COLUMNS`) of external rows, duplicates by
    (source, target) collapsed to the first occurrence, confidence as float."""
    seen: set[tuple[str, str]] = set()
    kept = []
    for source, target, relation, confidence, entity_type in rows:
        if (source, target) in seen:
            continue
        seen.add((source, target))
        kept.append((source, target, relation, float(confidence), entity_type))
    return pd.DataFrame(
        kept,
        columns=[COL_SOURCE_ENTITY_URI, COL_TARGET_ENTITY_URI, COL_RELATION, COL_CONFIDENCE, COL_ENTITY_TYPE],
    )


def write_external_m_ask(run_paths: PipelinePaths, rows: list[tuple[str, str, str, str, str]]) -> pd.DataFrame:
    """
    Publish external rows as the run's initial alignment and M_ask (`<task>-logmap_mappings.txt`,
    `<task>-logmap_mappings.tsv` and `<task>-logmap_mappings_to_ask_oracle_user_llm.txt`, all the
    same rows, pipe/tab format, original column text kept) so stage two and every later step
    find what a LogMap alignment would have left behind. Returns the M_ask DataFrame.
    """
    seen: set[tuple[str, str]] = set()
    unique = []
    for row in rows:
        if (row[0], row[1]) in seen:
            continue
        seen.add((row[0], row[1]))
        unique.append(row)
    pipe_text = "".join(PAIRS_SEPARATOR.join(row) + "\n" for row in unique)
    tab_text = "".join("\t".join(row) + "\n" for row in unique)
    run_paths.initial_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text_strict(run_paths.logmap_mappings(), lambda fp: fp.write(pipe_text))
    atomic_write_text_strict(run_paths.logmap_mappings_tsv(), lambda fp: fp.write(tab_text))
    atomic_write_text_strict(run_paths.logmap_m_ask(), lambda fp: fp.write(pipe_text))
    return external_m_ask_frame(unique)


###
# Annotated files
###


def _render_verdict(value) -> str:
    """True/False -> 'True'/'False'; 'error' -> ERROR; 'skipped' -> SKIPPED; 'n/a' kept;
    anything else (NaN, None, unparsed text) -> ERROR."""
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            value = value.item()          # numpy scalars
        except (TypeError, ValueError):
            pass
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return "True"
        if lowered == "false":
            return "False"
        if lowered == "skipped":
            return _ANNOTATION_SKIPPED
        if lowered == "n/a":
            return "n/a"
        return _ANNOTATION_ERROR
    return _ANNOTATION_ERROR


def _render_confidence(value) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if math.isnan(number):
        return "nan"
    return repr(number)


def _m_ask_rows(run_paths: PipelinePaths, predictions: pd.DataFrame) -> list[list[str]]:
    """The M_ask rows as five text columns, in file order, duplicates collapsed; from the
    M_ask file when it exists (verbatim text), else from the predictions DataFrame."""
    m_ask_path = run_paths.logmap_m_ask()
    rows: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    if m_ask_path.is_file():
        with m_ask_path.open(encoding="utf-8") as handle:
            for raw in handle:
                line = raw.rstrip("\r\n")
                if not line.strip():
                    continue
                fields = line.split(PAIRS_SEPARATOR)
                if len(fields) < 2:
                    continue
                fields = [field.strip() for field in fields[:5]]
                fields += ["=", "1.0", EntityType.CLASS.value][len(fields) - 2:]
                key = (fields[0], fields[1])
                if key in seen:
                    continue
                seen.add(key)
                rows.append(fields)
        return rows
    for _, row in predictions.iterrows():
        source, target = str(row.iloc[0]).strip(), str(row.iloc[1]).strip()
        if (source, target) in seen:
            continue
        seen.add((source, target))
        relation = str(row.iloc[2]) if len(row) > 2 and str(row.iloc[2]) in RELATIONS else "="
        confidence = _render_confidence(row.iloc[3]) if len(row) > 3 else "1.0"
        entity_type = str(row.iloc[4]) if len(row) > 4 and str(row.iloc[4]) in _ENTITY_TYPES else EntityType.CLASS.value
        rows.append([source, target, relation, confidence, entity_type])
    return rows


def annotate_rows(
    m_ask_rows: list[list[str]], predictions: pd.DataFrame, bidirectional: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """(txt rows, tsv rows) for the M_ask rows given the predictions DataFrame."""
    by_pair: dict[tuple[str, str], pd.Series] = {}
    for _, row in predictions.iterrows():
        key = (str(row[COL_SOURCE_ENTITY_URI]).strip(), str(row[COL_TARGET_ENTITY_URI]).strip())
        by_pair.setdefault(key, row)
    txt_rows: list[list[str]] = []
    tsv_rows: list[list[str]] = []
    for fields in m_ask_rows:
        row = by_pair.get((fields[0], fields[1]))
        if row is None:
            annotation, forward, reverse, confidence = _ANNOTATION_SKIPPED, _ANNOTATION_SKIPPED, _ANNOTATION_SKIPPED, "nan"
        else:
            annotation = _render_verdict(row.get("Oracle_prediction"))
            forward = _render_verdict(row.get("Oracle_fwd_prediction", "n/a")) if bidirectional else ""
            reverse = _render_verdict(row.get("Oracle_rev_prediction", "n/a")) if bidirectional else ""
            confidence = _render_confidence(row.get("Oracle_confidence"))
        extra = [forward, reverse] if bidirectional else []
        txt_rows.append(list(fields) + [annotation] + extra)
        tsv_rows.append(list(fields) + [annotation] + extra + [confidence])
    return txt_rows, tsv_rows


def write_annotated_files(
    run_paths: PipelinePaths, predictions: pd.DataFrame, bidirectional: bool = False,
) -> tuple[Path, Path]:
    """Write `<task>-<template>-annotated.txt` and `.tsv` (see the module docstring)."""
    txt_rows, tsv_rows = annotate_rows(_m_ask_rows(run_paths, predictions), predictions, bidirectional)
    header = list(TSV_BASE_HEADER) + (list(TSV_BIDIRECTIONAL_HEADER) if bidirectional else []) + [TSV_CONFIDENCE_HEADER]
    txt_text = "".join(PAIRS_SEPARATOR.join(row) + "\n" for row in txt_rows)
    tsv_text = "\t".join(header) + "\n" + "".join("\t".join(row) + "\n" for row in tsv_rows)
    run_paths.output_dir.mkdir(parents=True, exist_ok=True)
    txt_path, tsv_path = run_paths.annotated_txt(), run_paths.annotated_tsv()
    atomic_write_text_strict(txt_path, lambda fp: fp.write(txt_text))
    atomic_write_text_strict(tsv_path, lambda fp: fp.write(tsv_text))
    return txt_path, tsv_path


__all__ = [
    "load_external_mappings",
    "external_m_ask_frame",
    "write_external_m_ask",
    "annotate_rows",
    "write_annotated_files",
    "TSV_BASE_HEADER",
    "TSV_BIDIRECTIONAL_HEADER",
    "TSV_CONFIDENCE_HEADER",
]
