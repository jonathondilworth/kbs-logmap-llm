'''
MISC utils
'''
from pydantic import BaseModel
from pathlib import Path

from logmap_llm.evaluation.io import (
        load_entity_types_from_initial_alignment,
        load_mapping_pairs,
    )
from logmap_llm.evaluation.metrics import (
    classify_conference_pair,
    compute_prf,
)

###
# DEBUG HELPER (TYPE SWITCH FOR RESPONSE FMT)
###

def resolve_response_format_to_str(resp_fmt) -> str:
    '''
    ad-hoc type-switch: returns the resp_fmt value as str for debug logging
    '''
    if resp_fmt is None:
        return "Plain (None)"

    if isinstance(resp_fmt, type) and issubclass(resp_fmt, BaseModel):
        return resp_fmt.__name__

    if isinstance(resp_fmt, BaseModel):
        return f"UNEXPECTED instance of {type(resp_fmt).__name__} (expected the class itself, not an instance)"

    if isinstance(resp_fmt, dict):
        return f"raw dict schema (keys: {sorted(resp_fmt.keys())})"

    return f"UNKNOWN type={type(resp_fmt).__name__}"



def compute_conference_m1_m2_stratified(
    system_mappings_path: Path,
    reference_path: Path,
    initial_alignment_path: Path | None = None,
    class_reference_path: Path | None = None,
    property_reference_path: Path | None = None,
) -> dict | None:
    """
    Compute OAEI conference track M1 (class) and M2 (property) stratified metrics for a single
    task pair, applying compute_prf separately to class-only and property-only subsets.

    Suitable for calling directly from a notebook or analysis script to inspect M1/M2 breakdowns.
    Lives in misc.py because it belongs at the process-orchestration level, not the evaluation level.

    TODO: add a proper process-orchestration layer (results are currently aggregated per-pair with
    ad-hoc scripts, which makes ablations brittle), plus decent plotting scripts.

    Per-type reference files follow the conference track naming convention: for 'confOf-iasted.tsv'
    it expects 'confOf-iasted_class.tsv' and 'confOf-iasted_property.tsv' in the same directory,
    overridable via 'class_reference_path' / 'property_reference_path'. If neither per-type file
    exists and no overrides are given, returns None ("M1/M2 stratification not applicable").

    System pairs are classified class-class / property-property using LogMap entity type codes read
    from the initial alignment file (pipe-delimited m_ask format). If 'initial_alignment_path' is
    not provided, both buckets fall back to the full system pair set -- internally consistent but
    the split is effectively disabled; callers that care about M1/M2 must provide the path.

    Returns dict | None: zero, one, or two entries {"m1_class": ..., "m2_property": ...}, each a
    metric dict from compute_prf with its source field tagged "conference_stratified" so downstream
    JSON consumers can distinguish stratified from standard entries.
    """
    reference_path = Path(reference_path)
    ref_dir = reference_path.parent
    ref_stem = reference_path.stem

    class_ref_path = Path(
        class_reference_path
        if class_reference_path is not None
        else ref_dir / f"{ref_stem}_class.tsv"
    )

    property_ref_path = Path(
        property_reference_path
        if property_reference_path is not None
        else ref_dir / f"{ref_stem}_property.tsv"
    )

    if not class_ref_path.exists() and not property_ref_path.exists():
        return None

    system_pairs = load_mapping_pairs(Path(system_mappings_path))

    # build URI -> entity type code lookup - missing initial alignment
    # is the degenerate case where both buckets report the full pair set
    uri_types = {}
    if initial_alignment_path is not None:
        uri_types = load_entity_types_from_initial_alignment(
            Path(initial_alignment_path)
        )

    results: dict[str, dict] = {}

    if class_ref_path.exists():
        class_ref = load_mapping_pairs(class_ref_path)

        if uri_types:
            class_system = {
                (s, t) for s, t in system_pairs
                if classify_conference_pair(s, t, uri_types) == "class"
            }
            m1 = compute_prf(class_system, class_ref)
            m1["source"] = "conference_stratified"
        else:
            # Degenerate mode (no URI typing): the bucket is the full pair set,
            # tagged as an unstratified fallback so results don't claim a stratified metric.
            m1 = compute_prf(system_pairs, class_ref)
            m1["source"] = "conference_unstratified_fallback"
            m1["metric_notes"]["stratification"] = (
                "no URI typing available (initial alignment missing/unparsed); "
                "bucket is the full system pair set, NOT class-only"
            )
        results["m1_class"] = m1

    if property_ref_path.exists():
        property_ref = load_mapping_pairs(property_ref_path)

        if uri_types:
            property_system = {
                (s, t) for s, t in system_pairs
                if classify_conference_pair(s, t, uri_types) == "property"
            }
            m2 = compute_prf(property_system, property_ref)
            m2["source"] = "conference_stratified"
        else:
            m2 = compute_prf(system_pairs, property_ref)
            m2["source"] = "conference_unstratified_fallback"
            m2["metric_notes"]["stratification"] = (
                "no URI typing available (initial alignment missing/unparsed); "
                "bucket is the full system pair set, NOT property-only"
            )
        results["m2_property"] = m2

    return results if results else None

