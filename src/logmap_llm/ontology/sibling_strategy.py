"""
logmap_llm.ontology.sibling_strategy

The sibling-ranking strategy enum and its resolution rule, with no ontology dependency.

Split out of `sibling_retrieval.py` so configuration validation can determine the
strategy a run will use without importing owlready2. `sibling_retrieval` re-exports
`SiblingSelectionStrategy`, so existing imports keep working; this module is the single
definition of both the enum and the precedence rule.
"""
from __future__ import annotations

from enum import Enum


class SiblingSelectionStrategy(str, Enum):
    """Strategy for ranking the candidate sibling set."""

    ALPHANUMERIC = "alphanumeric"
    SHORTEST_LABEL = "shortest_label"
    CLS_TRANSFORMER = "cls_transformer"
    SBERT = "sbert"

    @property
    def is_embedding_based(self) -> bool:
        return self in (SiblingSelectionStrategy.CLS_TRANSFORMER, SiblingSelectionStrategy.SBERT)


#: Optional domain -> strategy auto-selection, consulted only when no strategy is configured
#: explicitly. Empty by default; a deployment registers mappings here keyed by a lowercase
#: substring of `alignmentTask.ontology_domain`. Prefer setting `prompts.sibling_strategy`
#: explicitly, which records the experimental condition in the config rather than module state.
DOMAIN_STRATEGY_OVERRIDES: dict[str, SiblingSelectionStrategy] = {}


def resolve_sibling_strategy(
    configured_strategy: str | None, ontology_domain: str | None,
) -> SiblingSelectionStrategy:
    """Pick a sibling selection strategy.

    Precedence: explicit config > registered domain override > generic auto.
    Pooling follows from the strategy: CLS_TRANSFORMER -> CLS, SBERT -> mean.
    """
    if configured_strategy is not None:
        return SiblingSelectionStrategy(configured_strategy)
    if ontology_domain and DOMAIN_STRATEGY_OVERRIDES:
        domain = ontology_domain.lower()
        for key, strategy in DOMAIN_STRATEGY_OVERRIDES.items():
            if key in domain:
                return strategy
    return SiblingSelectionStrategy.SBERT
