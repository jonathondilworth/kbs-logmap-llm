"""
logmap_llm.evaluation.metrics

Pure functions for evaluating alignments; each accepts sets of mappings,
prediction lists, entity-type dicts, etc. and returns a metric dict
(see the individual function docstrings for the exact dict shapes).
Undefined ratios are returned as None with a reason in 'metric_notes',
never coerced to a measured 0.0.

The 'partial gold standard' semantics used by compute_kg_partial_prf are
sourced from the OAEI 2025 KG track page (see that function's docstring
for the FN-counting rule):
    https://oaei.ontologymatching.org/2025/results/knowledgegraph/
"""
from __future__ import annotations

###
# pure functions
###


def calc_precision(true_positives: int, false_positives: int) -> float:
    if (true_positives + false_positives) > 0:
        return float(true_positives / (true_positives + false_positives))
    # else:
    return float(0.0)



def calc_recall(true_positives: int, false_negatives: int) -> float:
    if (true_positives + false_negatives) > 0:
        return float(true_positives / (true_positives + false_negatives))
    # else:
    return float(0.0)



def calc_f1(precision: float, recall: float) -> float:
    if (precision + recall) > 0:
        return float((2 * precision * recall) / (precision + recall))
    # else:
    return float(0.0)



def calc_sensitivity(true_positives: int, false_negatives: int) -> float:
    if (true_positives + false_negatives) > 0:
        return float(true_positives / (true_positives + false_negatives))
    # else:
    return float(0.0)



def calc_specificity(true_negatives: int, false_positives: int) -> float:
    if (true_negatives + false_positives) > 0:
        return float(true_negatives / (true_negatives + false_positives))
    # else:
    return float(0.0)



def calc_youdens(sensitivity: float, specificity: float) -> float:
    return float(sensitivity + specificity - 1.0)



###
# composite functions (pure)
###


def _nullable_prf_from_counts(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
) -> tuple[float | None, float | None, float | None, dict[str, str]]:
    """Compute P/R/F1 without turning undefined ratios into measured zeroes."""
    metric_notes: dict[str, str] = {}

    if true_positives + false_positives:
        precision = calc_precision(true_positives, false_positives)
    else:
        precision = None
        metric_notes["precision"] = "undefined: no evaluated system mappings (tp+fp=0)"

    if true_positives + false_negatives:
        recall = calc_recall(true_positives, false_negatives)
    else:
        recall = None
        metric_notes["recall"] = "undefined: no reference mappings (tp+fn=0)"

    if precision is None or recall is None:
        f1 = None
        metric_notes["f1"] = "undefined: precision or recall undefined"
    elif precision + recall:
        f1 = calc_f1(precision, recall)
    else:
        f1 = None
        metric_notes["f1"] = "undefined: precision+recall=0"

    return precision, recall, f1, metric_notes



def compute_prf(system_alignment: set[tuple[str, str]], reference_alignment: set[tuple[str, str]]) -> dict:
    """
    standard (canonical) set-based precision / recall / F1

    An undefined ratio is represented as ``None`` and explained in
    ``metric_notes``.  This distinguishes, for example, an empty system
    alignment (precision is undefined) from measured precision of 0.0.
    """
    true_positives = len(system_alignment & reference_alignment)     # elems in both system and reference
    false_positives = len(system_alignment - reference_alignment)    # elems in system not in reference
    false_negatives = len(reference_alignment - system_alignment)    # elems in reference not in system

    precision, recall, f1, metric_notes = _nullable_prf_from_counts(
        true_positives, false_positives, false_negatives
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "system_size": len(system_alignment),
        "reference_size": len(reference_alignment),
        "metric_notes": metric_notes,
        "source": "custom",
    }



def compute_kg_partial_prf(system_alignment: set[tuple[str, str]], reference_alignment: set[tuple[str, str]]) -> dict:
    """
    P/R/F1 under the OAEI KG track partial gold standard semantics:

      - true positive: any system mapping (A, B) in the reference alignment
      - false positive: any system mapping (A, B) not in the reference, where A is a
        reference source entity or B is a reference target entity
      - false negative: any reference mapping not in the system mappings,
        i.e. |reference| - TP
      - ignored: any system mapping (A, B) where A is not a reference source entity
        and B is not a reference target entity

    the 'source' identifier in the returned dict is 'kg_partial'
    """
    # guard
    if not reference_alignment:
        precision, recall, f1, metric_notes = _nullable_prf_from_counts(0, 0, 0)
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "ignored": len(system_alignment),
            "system_size": len(system_alignment),
            "evaluated_size": 0,
            "reference_size": 0,
            "metric_notes": metric_notes,
            "source": "kg_partial",
        }

    reference_sources = set()
    for m_src, _m_tgt in reference_alignment:
        reference_sources.add(m_src)

    reference_targets = set()
    for _m_src, m_tgt in reference_alignment:
        reference_targets.add(m_tgt)

    true_positives = 0
    false_positives = 0
    number_of_ignored_mappings = 0

    for system_src_mapping, system_target_mapping in system_alignment:
        if (system_src_mapping in reference_sources) or (system_target_mapping in reference_targets):
            if (system_src_mapping, system_target_mapping) in reference_alignment:
                true_positives += 1
            else:
                false_positives += 1
        else:
            number_of_ignored_mappings += 1

    false_negatives = len(reference_alignment) - true_positives
    total_number_of_evaluated_mappings = len(system_alignment) - number_of_ignored_mappings

    precision, recall, f1, metric_notes = _nullable_prf_from_counts(
        true_positives, false_positives, false_negatives
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "ignored": number_of_ignored_mappings,
        "system_size": len(system_alignment),
        "evaluated_size": total_number_of_evaluated_mappings,
        "reference_size": len(reference_alignment),
        "metric_notes": metric_notes,
        "source": "kg_partial",
    }



def compute_oracle_metrics(predictions: list[dict], reference_alignment: set[tuple[str, str]], partial_reference: bool = False) -> dict:
    """
    oracle discrimination metrics (sensitivity, specificity, Youden's J) plus
    oracle precision / recall / F1, treating the oracle as a diagnostic test:
    for each input mapping, did the oracle accept or reject it, and was that
    decision correct wrt the reference alignment?

    also counted:
      - 'errors': consultations that failed (oracle returned a non-usable answer)
      - 'partial_scope_excluded': predictions whose source and target URIs fall
        entirely outside the reference alignment; only applied when
        partial_reference is true (the compute_kg_partial_prf protocol)
      - 'oracle_excluded': errors + partial_scope_excluded
      - 'false_mappings': list of FP/FN mappings, kept for diagnostics
    """
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    errors_encountered = 0
    partial_scope_excluded = 0
    false_mappings: list[dict] = []

    reference_sources = set()
    reference_targets = set()

    if partial_reference:

        for m_src, _m_tgt in reference_alignment:
            reference_sources.add(m_src)

        for _m_src, m_tgt in reference_alignment:
            reference_targets.add(m_tgt)

    # oracle predictions in _mappings to ask_
    for oracle_response in predictions:

        source_prediction_URI = oracle_response['source']
        target_prediction_URI = oracle_response['target']
        truthy_oracle_prediction: bool = oracle_response['prediction']

        # check whether the consultation failed:
        if truthy_oracle_prediction is None:
            errors_encountered += 1
            continue

        # check whether we need to account for the partial reference alignment:
        if partial_reference and (source_prediction_URI not in reference_sources) and (target_prediction_URI not in reference_targets):
            partial_scope_excluded += 1
            continue

        response_in_reference_alignment = (source_prediction_URI, target_prediction_URI) in reference_alignment

        if truthy_oracle_prediction and response_in_reference_alignment:
            true_positives += 1

        elif truthy_oracle_prediction and not response_in_reference_alignment:
            false_positives += 1
            false_mappings.append({
                "source_entity_uri": source_prediction_URI,
                "target_entity_uri": target_prediction_URI,
                "oracle_prediction": True, #truthy_oracle_prediction
                "oracle_confidence": oracle_response.get("confidence"),
                "in_reference": False, # response_in_reference_alignment
                "error_type": "FP",
            })

        elif not truthy_oracle_prediction and not response_in_reference_alignment:
            true_negatives += 1

        elif not truthy_oracle_prediction and response_in_reference_alignment:
            false_negatives += 1
            false_mappings.append({
                "source_entity_uri": source_prediction_URI,
                "target_entity_uri": target_prediction_URI,
                "oracle_prediction": False, #truthy_oracle_prediction
                "oracle_confidence": oracle_response.get("confidence"),
                "in_reference": True, # response_in_reference_alignment
                "error_type": "FN",
            })

    # diagnostics
    # undefined sensitivity/specificity/Youden's J are reported as None with a reason in
    # metric_notes, not coerced to 0.0; 0.0 is kept only when the denominator is defined
    sens_defined = (true_positives + false_negatives) > 0
    spec_defined = (true_negatives + false_positives) > 0
    metric_notes: dict = {}
    if sens_defined:
        sensitivity = calc_sensitivity(true_positives, false_negatives)
    else:
        sensitivity = None
        metric_notes["sensitivity"] = "undefined: no positive reference candidates (tp+fn=0)"
    if spec_defined:
        specificity = calc_specificity(true_negatives, false_positives)
    else:
        specificity = None
        metric_notes["specificity"] = "undefined: no negative reference candidates (tn+fp=0)"
    if sens_defined and spec_defined:
        youdens_j = calc_youdens(sensitivity, specificity)
    else:
        youdens_j = None
        metric_notes["youdens_j"] = "undefined: sensitivity or specificity undefined"

    oracle_precision = calc_precision(true_positives, false_positives)
    oracle_recall = calc_recall(true_positives, false_negatives)
    oracle_f1 = calc_f1(oracle_precision, oracle_recall)

    return {
        "tp": true_positives,
        "fp": false_positives,
        "tn": true_negatives,
        "fn": false_negatives,
        "errors": errors_encountered,
        "partial_scope_excluded": partial_scope_excluded,
        "oracle_excluded": errors_encountered + partial_scope_excluded,
        "total_candidates": len(predictions),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youdens_j": youdens_j,
        "metric_notes": metric_notes,
        "oracle_precision": oracle_precision,
        "oracle_recall": oracle_recall,
        "oracle_f1": oracle_f1,
        "false_mappings": false_mappings,
    }



def classify_uri_entity_type(uri: str) -> str:
    """
    classifies a URI as "class", "property", "instance", or "unknown" by
    substring-matching conventional URI path fragments; used by the KG-track
    stratified global metrics. '/class/', '/property/' and '/resource/' match
    the DBkWik URI conventions (and some other KGs); non-matches return 'unknown'
    """
    if "/class/" in uri:
        return "class"

    if "/property/" in uri:
        return "property"

    if "/resource/" in uri:
        return "instance"

    return "unknown"



def classify_mapping_pair(src_uri: str, tgt_uri: str) -> str:
    """
    classifies a mapping pair (src, tgt) as "class" / "property" / "instance"
    / "unknown" by URI convention, for KG-track diagnostic stratification.

    Uses the source URI's type; falls back to the target URI only when the
    source classifies as "unknown". This diagnostic stratifier powers per-type
    strata when explicit reference_{class,property,instance}.tsv files are
    unavailable; the official KG stratification (MELT kgEvalCli) is separate
    and typed by the reference itself. Keyed on the DBkWik-style path fragments
    (see classify_uri_entity_type); non-KG URIs classify as "unknown".
    """
    src_type = classify_uri_entity_type(src_uri)
    if src_type != "unknown":
        return src_type
    return classify_uri_entity_type(tgt_uri)



def classify_conference_pair(src_uri: str, tgt_uri: str, uri_types: dict[str, str]) -> str:
    """
    classifies a conference (track) mapping pair as either 'class' or 'property'
    based on the URI types provided to this function
    """
    src_type = uri_types.get(src_uri, "UNKNO")
    tgt_type = uri_types.get(tgt_uri, "UNKNO")
    if src_type == "CLS" and tgt_type == "CLS":
        return "class"
    if src_type in ("OPROP", "DPROP") and tgt_type in ("OPROP", "DPROP"):
        return "property"
    return "other"
