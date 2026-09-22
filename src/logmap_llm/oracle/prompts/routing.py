"""
logmap_llm.oracle.prompts.routing

Pure (dependency-free) prompt-lane routing for M_ask candidates, importable without
owlready2 so the routing decision is unit-testable in a CPU-only environment.

These helpers prefer LogMap's authoritative entityType tag and fall back to the
ontology-derived type only when LogMap gave none: re-deriving the type from the
ontology collapses DPROP into the OPROP lane and mis-routes OWL2-punned
(class+property) URIs.
"""
from __future__ import annotations

# The tags LogMap writes in the 5th column of its M_ask file.
_LOGMAP_ENTITY_TYPES = ("CLS", "OPROP", "DPROP", "INST")


def mask_row_entity_type(row_series) -> str | None:
    """
    Return the authoritative LogMap entityType tag (CLS/OPROP/DPROP/INST) carried in the M_ask
    row's 5th column (index 4), or None when it is absent / blank / UNKNO / unrecognised.
    ``row_series`` is a pandas Series (or any sequence with ``.iloc``/len) for one M_ask row.
    """
    try:
        if len(row_series) > 4:
            et = str(row_series.iloc[4]).strip().upper()
            if et in _LOGMAP_ENTITY_TYPES:
                return et
    except Exception:
        return None
    return None


def resolve_pair_lane(mask_et: str | None, src_type: str, tgt_type: str) -> str:
    """
    Decide the prompt lane for a candidate: 'class' | 'property' | 'instance'.

    Prefers the authoritative LogMap entityType (``mask_et``); falls back to the
    ontology-derived resolve_entity types (``src_type``/``tgt_type``, one of
    'class'/'property'/'instance') only when LogMap gave no usable tag.
    """
    if mask_et == "INST":
        return "instance"
    if mask_et in ("OPROP", "DPROP"):
        return "property"
    if mask_et == "CLS":
        return "class"
    # fallback: derive the lane from the resolved entity types
    if src_type == "instance" or tgt_type == "instance":
        return "instance"
    if src_type == "property" or tgt_type == "property":
        return "property"
    return "class"
