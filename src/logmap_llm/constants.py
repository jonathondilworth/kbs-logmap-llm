from __future__ import annotations
from typing import Literal
from pydantic import BaseModel
from enum import Enum
import os

###
# Debugging
###

VERBOSE = True
VERY_VERBOSE = False

###
# Oracle-related pydantic (schema) models
###

class BinaryOutputFormat(BaseModel):
    answer: bool

class BinaryOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: bool

class YesNoOutputFormat(BaseModel):
    answer: Literal["Yes", "No"]

class YesNoOutputFormatWithReasoning(BaseModel):
    reasoning: str
    answer: Literal["Yes", "No"]

class TokensUsage(BaseModel):
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None = None
    finish_reason: str | None = None
    native_finish_reason: str | None = None
    provider: str | None = None
    response_model: str | None = None
    raw_response: str | None = None

class LLMCallOutput(BaseModel):
    message: str | bool
    usage: TokensUsage
    logprobs: list | None
    parsed: BaseModel | None

###
# Oracle-related enums
###

class AnswerFormat(str, Enum):
    TRUE_FALSE = "true_false"
    YES_NO     = "yes_no"

class PositiveToken(str, Enum):
    TRUE = "true"
    YES  = "yes"

class NegativeToken(str, Enum):
    FALSE = "false"
    NO    = "no"

class ResponseModes(str, Enum):
    STRUCTURED = "structured"
    PLAIN  = "plain"

class InteractionStyle(str, Enum):
    AUTOMATIC = 'auto'
    OPEN_ROUTER = 'openrouter'
    OPEN_AI_CHAT_COMPLETIONS_PARSE = 'openai_chat_completions_parse_structured_output'
    LOCAL_VLLM = 'vllm'
    LOCAL_SG_LANG = 'sglang'
    LOCAL_GENERIC = 'local'


###
# Oracle-related 'constants'
###

PAIRS_SEPARATOR = "|"
ORACLE_PREDICTION_COLUMN = "Oracle_prediction"

###
# positive & negative tokens for answer format
# (used by logprobs extraction, text-fallback parsing, and CSV normalisation)
###

POSITIVE_TOKENS = frozenset({
    PositiveToken.TRUE,
    PositiveToken.YES,
})

NEGATIVE_TOKENS = frozenset({
    NegativeToken.FALSE,
    NegativeToken.NO,
})

###
# Answer format and response mode
# -------------------------------
# answer_format controls the answer vocabulary (true/false vs yes/no);
# response_mode controls the output container (structured JSON vs plain text).
# response_format follows from the combination: None for unstructured output,
# otherwise one of the (optionally reasoning-carrying) pydantic models above.
# Note: there is no support for unstructured responses with reasoning at present.
###

ANSWER_FORMATS = frozenset({
    AnswerFormat.TRUE_FALSE,
    AnswerFormat.YES_NO,
})

RESPONSE_MODES = frozenset({
    ResponseModes.STRUCTURED,
    ResponseModes.PLAIN,
})

RESPONSE_INSTRUCTION = {
    (AnswerFormat.TRUE_FALSE, ResponseModes.STRUCTURED): 'Respond with a JSON object: {"answer": true} or {"answer": false}.',
    (AnswerFormat.TRUE_FALSE, ResponseModes.PLAIN     ): 'Respond with "True" or "False".',
    (AnswerFormat.YES_NO,     ResponseModes.STRUCTURED): 'Respond with a JSON object: {"answer": "Yes"} or {"answer": "No"}.',
    (AnswerFormat.YES_NO,     ResponseModes.PLAIN     ): 'Respond with "Yes" or "No".',
}

RESPONSE_FORMAT_FOR_ANSWER = {
    (AnswerFormat.TRUE_FALSE , False): BinaryOutputFormat,
    (AnswerFormat.TRUE_FALSE , True ): BinaryOutputFormatWithReasoning,
    (AnswerFormat.YES_NO     , False): YesNoOutputFormat,
    (AnswerFormat.YES_NO     , True ): YesNoOutputFormatWithReasoning,
}

RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE = None

DEFAULT_ANSWER_FORMAT = AnswerFormat.TRUE_FALSE
DEFAULT_RESPONSE_MODE = ResponseModes.STRUCTURED

###
# Ontology-related enums
###

class EntityType(Enum):
    '''
    Entity type symbols as used by LogMap in its output files; LogMap-LLM
    uses the same symbols internally and in its own output files, so think
    carefully before making changes.
    '''
    CLASS = 'CLS'
    DATAPROPERTY = 'DPROP'
    OBJECTPROPERTY = 'OPROP'
    INSTANCE = 'INST'
    UNKNOWN = 'UNKNO'


class EntityRelation(Enum):
    '''
    Entity relation symbols as used by LogMap in its output files; LogMap-LLM
    uses the same symbols internally and in its own output files, so think
    carefully before making changes.
    '''
    SUBCLASSOF = '<'
    SUPERCLASSOF = '>'
    EQUIVALENCE = '='


###
# Pipeline configurations
# -----------------------
# The pipeline (see `orchestration.py`) is composed of steps where the output
# of fn_i is the input to fn_i+1; each step dispatches via match-case on the
# enums below. Original implementation:
#   https://github.com/jonathondilworth/logmap-llm/tree/jd-extended
###

# Step one (initial alignment)

class AlignMode(str, Enum):
    ALIGN = 'align'
    REUSE = 'reuse'
    BYPASS = 'bypass'
    # M_ask supplied by an external file (alignmentTask.external_mappings_filepath): no
    # LogMap alignment; the oracle annotates every mapping of that file (see
    # pipeline/annotate.py). Added 22 Sep 2026 (LOCAL_CHANGES.md §7).
    EXTERNAL = 'external'

# Step two (build prompts)

class PromptBuildMode(str, Enum):
    BUILD = 'build'
    REUSE = 'reuse'
    BYPASS = 'bypass'

# Step three (consult oracle)

class ConsultMode(str, Enum):
    CONSULT = 'consult'
    REUSE = 'reuse'
    LOCAL = 'local'
    BYPASS = 'bypass'

# Step four (refinement)

class RefineMode(str, Enum):
    REFINE = 'refine'
    BYPASS = 'bypass'

# Step four (refinement strategy)

class RefinementStrategy(str, Enum):
    '''
    How refinement runs when RefineMode.REFINE is matched in `runner.py`.
    Refinement via LogMap resolves any remaining conflicts, but can be slow
    (and some LogMap versions cannot complete it for instances, e.g. the
    OAEI 2025 KG task), so we also offer an approximate refinement in Python
    via a set union operation — handy for quickly testing changes; see
    `_kg_refine_in_python` in `logmap_llm.pipeline.orchestration`.
    '''
    LOGMAP = 'logmap'
    PYTHON_SETUNION = 'python'


###
# Constants (related to bridging.py)
# ----------------------------------
# M_ASK_COLUMNS is an immutable tuple; bridging.py provides a function that
# casts it to a list where one is needed.
###

COL_SOURCE_ENTITY_URI = 'source_entity_uri'
COL_TARGET_ENTITY_URI = 'target_entity_uri'
COL_RELATION = 'relation'
COL_CONFIDENCE = 'confidence'
COL_ENTITY_TYPE = 'entityType'

M_ASK_COLUMNS = (
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    COL_RELATION,
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
)

###
# Constants - oracle manager / consultation
###

DEFAULT_FAILURE_TOLERANCE_FLOOR       = 5
DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE = 5


###
# Constants - sibling selection
# -----------------------------
# defaults for SiblingSelector: some are (likely) tunable; see
# logmap_llm.ontology.sibling_retrieval for the strategy enum
###

# pretrained encoders (choices):
# The CLS-pooled strategy has no in-source default checkpoint: which specialised
# encoder a run uses is part of its experimental identity, so it is declared in
# config (prompts.sibling_model / few_shot.rag_encoder_model) rather than
# inherited silently from this module.
DEFAULT_CLS_ENCODER_MODEL: str | None = None
DEFAULT_GENERAL_MODEL = "sentence-transformers/all-MiniLM-L12-v2"

# cost cap on the candidate sibling set before scoring/embedding; set it to
# max(sibling_count) measured over your ontologies to 'disable' the cap, at the
# price of embedding cost that scales with the largest child count per class
# (potentially expensive for unclassified or 'flat' ontologies).
DEFAULT_MAX_SIBLING_CANDIDATES = 50

# default top-k siblings injected into prompts; keep small (1–3) to avoid
# bloating prompts that require siblings.
DEFAULT_TOP_K = 2

# default strategy name (ie. string form of SiblingSelectionStrategy)
# select from: "alphanumeric", "shortest_label", "cls_transformer", "sbert"
DEFAULT_SIBLING_STRATEGY = "cls_transformer"

###
# Constants - few shot
# --------------------
###

# sampled examples can collide with mappings in M_ask; we retry until a
# collision-free example set is obtained, capped to avoid an infinite loop in
# the (rare) case where that is impossible to satisfy.
DEFAULT_MAX_SAMPLE_RETRIES = 50

###
# Constants - evaluation
# ----------------------
###

# used to identify files that satisfy the DeepOnto TSV convention for ontology
# matching tasks, as used by DeepOnto's own evaluation implementation (which we
# re-use and borrow from): https://github.com/KRR-Oxford/DeepOnto
DEEPONTO_TSV_HEADER_PREFIXES: tuple[str, ...] = ("SrcEntity",)

# Evaluation engines selectable through `[evaluation] engines = [...]`. The first three
# produce the plain `global` block exactly as before (custom = set-based P/R/F1 on '='
# reference cells, partial_reference = the OAEI Knowledge Graph partial-gold-standard
# rule, deeponto = DeepOnto's AlignmentEvaluator); the last two are the track-faithful
# engines added in September 2026 (evaluation/engines/{logmap_oaei,bioml}.py) and can
# only follow a primary engine, each contributing a `global_<engine>` block.
PRIMARY_EVALUATION_ENGINES: tuple[str, ...] = ("custom", "partial_reference", "deeponto")
EXTRA_EVALUATION_ENGINES: tuple[str, ...] = ("logmap_oaei", "bioml")
EVALUATION_ENGINE_NAMES: tuple[str, ...] = PRIMARY_EVALUATION_ENGINES + EXTRA_EVALUATION_ENGINES


###
# Constants - bridging
# --------------------
# Confidence coercion at the Java boundary: when the oracle accepts a mapping
# but no usable logprob token was emitted, calculate_logprobs_confidence
# returns float('nan'); NaN is not a semantically safe value for LogMap's
# MappingObjectStr(..., double conf, ...), so it is coerced to this default
# (1.0, i.e. confident 'True'). This only fires for oracle predictions that
# accept the mapping.
###

DEFAULT_CONFIDENCE_FALLBACK = 1.0

###
# Constants - caching
# -------------------
###

DEFAULT_PROJECT_CACHE_ROOT = os.path.join(
    os.environ.get(
        'XDG_CACHE_HOME',
        os.path.expanduser('~/.cache')
    ),
    'logmap-llm',
)

DEFAULT_OWLREADY2_CACHE_DIR = os.path.join(
    DEFAULT_PROJECT_CACHE_ROOT,
    'owlready2',
)

DEFAULT_ENTROPY_CACHE_DIR = os.path.join(
    DEFAULT_PROJECT_CACHE_ROOT,
    'entropies',
)

JSON_DATA = (
    dict
    | list
    | str
    | int
    | float
    | bool
    | None
)

###
# Constants - ontology access (OBDA layer)
# ----------------------------------------
###

CANONICAL_RDFS_LABEL_URI_REF_STR = "http://www.w3.org/2000/01/rdf-schema#label"
