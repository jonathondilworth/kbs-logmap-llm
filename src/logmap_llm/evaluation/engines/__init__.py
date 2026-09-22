from logmap_llm.constants import (
    EVALUATION_ENGINE_NAMES,
    EXTRA_EVALUATION_ENGINES,
    PRIMARY_EVALUATION_ENGINES,
)
from logmap_llm.evaluation.engines.base import EvaluationEngine
from logmap_llm.evaluation.engines.bioml import BioMLEvaluationEngine
from logmap_llm.evaluation.engines.custom import CustomEvaluationEngine
from logmap_llm.evaluation.engines.deeponto import DeepOntoEvaluationEngine
from logmap_llm.evaluation.engines.logmap_oaei import LogMapOAEIEvaluationEngine
from logmap_llm.evaluation.engines.partial_reference import (
    PartialReferenceEvaluationEngine,
)


def build_engine(name: str, options: dict | None = None) -> EvaluationEngine:
    """
    Instantiate an engine by its registry name (`constants.EVALUATION_ENGINE_NAMES`).
    The primary engines take no options; the track-faithful engines take the option
    table of their `[evaluation.<name>]` config section. `deeponto` is constructed
    directly here — the harness's `select_engine` remains responsible for the JVM heap
    and the availability fallback of the primary block.
    """
    options = dict(options or {})
    if name == "custom":
        return CustomEvaluationEngine()
    if name == "partial_reference":
        return PartialReferenceEvaluationEngine()
    if name == "deeponto":
        return DeepOntoEvaluationEngine()
    if name == "logmap_oaei":
        return LogMapOAEIEvaluationEngine(**options)
    if name == "bioml":
        return BioMLEvaluationEngine(**options)
    raise ValueError(
        f"unknown evaluation engine {name!r}; known engines: {list(EVALUATION_ENGINE_NAMES)}"
    )


__all__ = [
    "EvaluationEngine",
    "CustomEvaluationEngine",
    "DeepOntoEvaluationEngine",
    "PartialReferenceEvaluationEngine",
    "LogMapOAEIEvaluationEngine",
    "BioMLEvaluationEngine",
    "build_engine",
    "EVALUATION_ENGINE_NAMES",
    "PRIMARY_EVALUATION_ENGINES",
    "EXTRA_EVALUATION_ENGINES",
]
