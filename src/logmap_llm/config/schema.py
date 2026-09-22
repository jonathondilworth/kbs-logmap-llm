from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal, Optional, ClassVar
from logmap_llm.constants import (
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
    RefinementStrategy,
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    RESPONSE_FORMAT_FOR_ANSWER,
    ResponseModes,
    AnswerFormat,
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_RESPONSE_MODE,
    InteractionStyle,
    EVALUATION_ENGINE_NAMES,
    PRIMARY_EVALUATION_ENGINES,
)
from logmap_llm.ontology.sibling_strategy import resolve_sibling_strategy
from logmap_llm.ontology.vocabularies import (
    OntologyConventionVocabulary,
    get_preset,
)
from logmap_llm.utils.logging import warn


class StrictConfigModel(BaseModel):
    """Base for user-authored configuration: typos must never be ignored."""

    model_config = ConfigDict(extra="forbid")


class AlignmentTaskConfig(StrictConfigModel):
    task_name: str = Field(
        min_length=1,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    )
    onto_source_filepath: str = Field(min_length=1)
    onto_target_filepath: str = Field(min_length=1)
    generate_extended_mappings_to_ask_oracle: bool = False
    # owl:imports IRIs that owlready2 must not follow. Empty for every track except
    # circular-economy, whose CEON.rdf imports an unpublished module (.../qudt/0.1/) that 404s.
    stub_import_iris: list[str] = Field(default_factory=list)
    logmap_parameters_dirpath: str = ""
    logmap_jvm_memory: str = Field(default="8g", pattern=r"^[1-9][0-9]*[mMgG]$")
    ontology_domain: str | None = None
    ontology_vocabulary: str = "default"
    # pipeline.align_ontologies = "external": the mappings to annotate (LogMap pipe .txt,
    # TSV, or OAEI Alignment RDF); required in that mode and rejected in every other.
    external_mappings_filepath: Optional[str] = None

    @model_validator(mode="after")
    def _validate_vocabulary_preset(self) -> "AlignmentTaskConfig":
        """fail fast on unknown vocabulary names at config-load time"""
        get_preset(self.ontology_vocabulary)  # raises ValueError if unknown
        return self

    @property
    def resolved_vocabulary(self) -> OntologyConventionVocabulary:
        """the actual vocabulary instance, resolved from the preset name"""
        return get_preset(self.ontology_vocabulary)


class PromptTemplateConfig(StrictConfigModel):
    '''
    For controlling which templates are used during oracle consultation.
    Every track is assumed to include class-based alignment, so the cls dev/user
    templates default to `class_equivalence` and `synonyms_only`.
    '''
    cls_dev_prompt_template_name: str            = "class_equivalence"
    cls_usr_prompt_template_name: str            = "synonyms_only"
    prop_dev_prompt_template_name: Optional[str] = "property_equivalence"
    prop_usr_prompt_template_name: Optional[str] = None
    # Optional dedicated data-property user template. When unset, data-property candidates
    # use the datatype-aware object-property template `prop_usr_prompt_template_name`.
    dprop_usr_prompt_template_name: Optional[str] = None
    inst_dev_prompt_template_name: Optional[str] = "instance_equivalence"
    inst_usr_prompt_template_name: Optional[str] = None

    # (strategy) select from 'alphanumeric', 'shortest_label', 'cls_transformer', 'sbert'
    # when unset, resolves via any registered domain override, else 'sbert'
    # (see ontology.sibling_strategy.DOMAIN_STRATEGY_OVERRIDES)
    sibling_strategy: Optional[
        Literal["alphanumeric", "shortest_label", "cls_transformer", "sbert"]
    ] = None

    # optional override of the embedding model; only used when sibling_strategy
    # is 'cls_transformer' or 'sbert'; when None, the default checkpoint is used
    sibling_model: Optional[str] = None

    # An immutable checkpoint commit for the sibling embedding model. Required whenever
    # `few_shot.rag_negative_layout = 'paired-sibling-v2'` resolves to an embedding strategy:
    # an unpinned model can silently change the experiment, and an unpinned `from_pretrained`
    # resolves `refs/main`, which an offline cache holding only the pinned snapshot cannot serve.
    sibling_model_revision: Optional[str] = None

    # the cost cap on the candidate sibling set prior to ranking
    # None -> uses DEFAULT_MAX_SIBLING_CANDIDATES from constants.py
    sibling_max_candidates: Optional[int] = Field(default=None, ge=1)

    # Compute device for the sibling ranker. None -> cuda if available, else cpu.
    # Declare it when running on more than one host: cuda and cpu matmuls can resolve a
    # near-tie between candidate siblings differently, and the device is not part of
    # condition_id (`rag_encoder_device` exists for the same reason on the retrieval side).
    sibling_encoder_device: Optional[str] = None

    @model_validator(mode="after")
    def validate_prompt_lanes(self) -> "PromptTemplateConfig":
        property_user_configured = any(
            (value or "").strip()
            for value in (
                self.prop_usr_prompt_template_name,
                self.dprop_usr_prompt_template_name,
            )
        )
        if property_user_configured and not (
            self.prop_dev_prompt_template_name or ""
        ).strip():
            raise ValueError(
                "prop_dev_prompt_template_name must be non-empty when a property "
                "or data-property user prompt is configured"
            )
        if (self.inst_usr_prompt_template_name or "").strip() and not (
            self.inst_dev_prompt_template_name or ""
        ).strip():
            raise ValueError(
                "inst_dev_prompt_template_name must be non-empty when an instance "
                "user prompt is configured"
            )
        return self


#: Strategies whose resolved RAG mode is QUERY_RAG or STATIC_HARD — the only two the
#: retriever's paired branch accepts. Duplicated rather than imported because `config` must
#: not depend on `oracle.rag`; tests/test_negative_layout_config.py pins this set in
#: agreement with `oracle.rag.pipeline_adapter._STRATEGY_TO_MODE`.
_PAIRED_STRATEGIES = frozenset({"query-rag", "hard-similar", "static-hard", "hard"})


class FewShotConfig(StrictConfigModel):
    '''
    For controlling the few-shot prompting mode during oracle consultation.
    The default `few_shot_k` of 0 disables few-shot prompting entirely.
    '''
    few_shot_k: int                 = Field(default=0, ge=0)
    few_shot_seed: int              = 42
    few_shot_negative_strategy: str = "hard" # hard negatives produce near-miss contrastive examples
    rag_encoder_kind: Literal["cls_transformer", "sbert", "hashing"] = "cls_transformer"
    # No in-source default: the encoder checkpoint is part of the run's experimental identity
    # and is declared here by the campaign. _build_rag_encoder refuses an empty value for any
    # semantic encoder kind.
    rag_encoder_model: str = ""
    rag_encoder_revision: Optional[str] = None
    rag_encoder_device: Optional[str] = None
    rag_encoder_max_length: int = Field(default=64, ge=1)
    rag_failure_policy: Literal["error", "record_zero_shot"] = "error"
    rag_cache_dir: Optional[str] = None
    prebuilt_few_shot_bundle_path: Optional[str] = None
    # Which anchors a prebuilt bundle may draw its demonstrations from: 'leave-one-task-out'
    # (every example comes from another task; the ISWC campaigns) or 'pooled' (the receiver's
    # own anchors are eligible as well). Unset means leave-one-task-out, so frozen configs and
    # the bundles they bind stay unchanged; the loader refuses a bundle built under the other
    # policy (see pipeline/rag_fewshot.PREBUILT_SELECTION_POLICIES).
    prebuilt_anchor_pool: Optional[Literal["leave-one-task-out", "pooled"]] = None

    # How the k/2 pseudo-negatives are constructed:
    #
    #   'paired-sibling-v2'  Ni = (Pi.src, highest-ranked eligible ontology sibling of Pi.tgt)
    #   'paired-donor-v2'    Ni = (Pi.src, target of the next eligible ranked donor after Pi)
    #   'donor-cross-v1'     Frozen: reproduces the completed campaigns byte-for-byte, retained
    #                        only so their demonstration blocks can be re-rendered under a
    #                        newer core. New campaign specs must not select it.
    #
    # Required whenever few_shot_k > 0, but deliberately not enforced by a validator here:
    # frozen per-job configs inside completed batches carry few_shot_k > 0 with no layout and
    # must stay loadable by aggregation and import tooling (a schema-level rejection would make
    # aggregate._job_row mark them status='invalid_plan' and overwrite the completed batch's
    # aggregate output). The requirement is instead
    # enforced where campaigns are created and executed: experiments/plan.py at generation,
    # pipeline/stage_two.py at execution.
    rag_negative_layout: Optional[
        Literal["paired-sibling-v2", "paired-donor-v2", "donor-cross-v1"]
    ] = None

    # few_shot_negative_strategy is validated against the known strategies / RAG modes so a
    # typo or an unimplemented mode fails loudly at config load instead of silently
    # downgrading to random.
    VALID_NEGATIVE_STRATEGIES: ClassVar[set] = {
        "hard", "random", "hard-similar",          # legacy sampler strategy names
        "query-rag", "static-hard", "static-random", "zero-shot",  # RAG mode names (oracle/rag)
    }

    @model_validator(mode="before")
    @classmethod
    def migrate_sapbert_encoder_kind(cls, data: dict) -> dict:
        """Accept the legacy `rag_encoder_kind = "sapbert"` (the CLS-pooled encoder before it
        was generalised) so the frozen configs of completed campaigns, whose sealed alignments
        the bundle builder reads, still load."""
        if isinstance(data, dict) and data.get("rag_encoder_kind") == "sapbert":
            warn("few_shot.rag_encoder_kind 'sapbert' is DEPRECATED; use 'cls_transformer' instead.")
            return {**data, "rag_encoder_kind": "cls_transformer"}
        return data

    @model_validator(mode="after")
    def _validate_few_shot(self):
        strat = self.few_shot_negative_strategy
        if strat not in self.VALID_NEGATIVE_STRATEGIES:
            raise ValueError(
                f"few_shot_negative_strategy={strat!r} is not recognised; expected one of "
                f"{sorted(self.VALID_NEGATIVE_STRATEGIES)}. Refusing to silently fall back to random."
            )
        if (
            self.few_shot_k > 0
            and self.rag_encoder_kind in {"cls_transformer", "sbert"}
            and not (self.rag_encoder_revision or "").strip()
        ):
            raise ValueError(
                "rag_encoder_revision is required when semantic few-shot retrieval is enabled; "
                "pin an immutable "
                "model revision or explicitly select rag_encoder_kind='hashing' as a baseline."
            )
        layout = self.rag_negative_layout
        # Safe to enforce at load: every frozen config from a completed campaign has
        # rag_negative_layout absent, so these rules cannot reject one.
        if layout is not None and self.few_shot_k <= 0:
            raise ValueError(
                "few_shot.rag_negative_layout is meaningless with few_shot_k = 0; "
                "remove it, or enable few-shot retrieval"
            )
        if layout in {"paired-sibling-v2", "paired-donor-v2"} and self.few_shot_k % 2 != 0:
            raise ValueError(
                f"few_shot.rag_negative_layout={layout!r} pairs one constructed negative with "
                f"each retrieved positive, so few_shot_k must be even; got {self.few_shot_k}"
            )
        # The retriever only takes the paired branch when the mode is QUERY_RAG or
        # STATIC_HARD. Under any other strategy a v2 layout would be recorded in the trace
        # as `negative_layout` and then ignored — a job whose trace claims one method while
        # its prompts contain another.
        if layout in {"paired-sibling-v2", "paired-donor-v2"} and strat not in _PAIRED_STRATEGIES:
            raise ValueError(
                f"few_shot.rag_negative_layout={layout!r} builds one negative per retrieved "
                f"positive, which requires a ranked retrieval mode, but "
                f"few_shot_negative_strategy={strat!r} does not provide one. Use one of "
                f"{sorted(_PAIRED_STRATEGIES)}, or select rag_negative_layout='donor-cross-v1'."
            )
        if self.prebuilt_few_shot_bundle_path is not None:
            if not self.prebuilt_few_shot_bundle_path.strip():
                raise ValueError("prebuilt_few_shot_bundle_path must not be empty")
            if layout is not None and layout != "donor-cross-v1":
                raise ValueError(
                    "prebuilt_few_shot_bundle_path carries already-rendered cross-task "
                    "donor negatives, so it requires "
                    "few_shot.rag_negative_layout='donor-cross-v1'"
                )
            if self.few_shot_k <= 0:
                raise ValueError(
                    "prebuilt_few_shot_bundle_path requires few_shot_k > 0"
                )
            if self.few_shot_k != 4:
                raise ValueError(
                    "prebuilt_few_shot_bundle_path currently requires few_shot_k = 4"
                )
            if strat != "query-rag":
                raise ValueError(
                    "prebuilt_few_shot_bundle_path requires "
                    "few_shot_negative_strategy='query-rag'"
                )
            if self.rag_failure_policy != "error":
                raise ValueError(
                    "prebuilt_few_shot_bundle_path requires rag_failure_policy='error'"
                )
        elif self.prebuilt_anchor_pool is not None:
            raise ValueError(
                "few_shot.prebuilt_anchor_pool describes a prebuilt bundle; set "
                "prebuilt_few_shot_bundle_path or remove it"
            )
        return self


class OracleConfig(StrictConfigModel):
    model_name: str = Field(min_length=1)  # required; provide in the config.toml
    api_key: str = "EMPTY" # use 'EMPTY' for vLLM or the actual API key for your selected service

    # LLM parameters
    base_url: Optional[str] = "https://openrouter.ai/api/v1" # "http://localhost:8000/v1" (for vLLM & SGLang)
    supports_chat_template_kwargs: Optional[bool] = None
    failure_tolerance: Optional[int] = Field(default=None, ge=1)

    max_workers: int                      = Field(default=24, ge=1)
    enable_thinking: Optional[bool]       = False            # toggles thinking mode on/off for supported models
    max_completion_tokens: int            = Field(default=2048, ge=1)
    request_timeout_seconds: float         = Field(default=120.0, gt=0.0)
    connect_timeout_seconds: float         = Field(default=15.0, gt=0.0)
    transient_retries: int                 = Field(default=2, ge=0, strict=True)
    seed: Optional[int]                    = Field(default=None, strict=True)
    request_logprobs: bool                 = True
    openrouter_provider: Optional[str]      = Field(default=None, min_length=1)
    openrouter_allow_fallbacks: bool        = False
    openrouter_require_parameters: bool     = True
    temperature: float                    = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float                          = Field(default=1.0, gt=0.0, le=1.0)
    reasoning_effort: Optional[str]       = None             # set explicitly only for models that support reasoning controls
    reasoning_token_budget: Optional[int] = Field(default=None, ge=1)
    local_oracle_predictions_dirpath: str = ""

    interaction_style: Literal[
        InteractionStyle.AUTOMATIC,
        InteractionStyle.OPEN_AI_CHAT_COMPLETIONS_PARSE,
        InteractionStyle.OPEN_ROUTER,
        InteractionStyle.LOCAL_GENERIC,
        InteractionStyle.LOCAL_VLLM,
        InteractionStyle.LOCAL_SG_LANG,
    ] = InteractionStyle.AUTOMATIC

    '''
    Note, at present there is some _delicate_ global state in templates.py that depends on
    DEFAULT_ANSWER_FORMAT and DEFAULT_RESPONSE_MODE from constants.py, this is why we use
    these defaults here, just to ensure that the schema is aligned to the specified defaults
    when these values may be changed by a user. TODO: prepare a more appropriate implementation.
    '''
    answer_format: Literal[AnswerFormat.TRUE_FALSE, AnswerFormat.YES_NO]  = DEFAULT_ANSWER_FORMAT    # 'true_false'
    response_mode: Literal[ResponseModes.STRUCTURED, ResponseModes.PLAIN] = DEFAULT_RESPONSE_MODE    # 'structured'

    @property
    def response_format(self) -> BinaryOutputFormat | BinaryOutputFormatWithReasoning | YesNoOutputFormat | YesNoOutputFormatWithReasoning | None:
        if self.response_mode not in (ResponseModes.STRUCTURED, ResponseModes.PLAIN):
            raise ValueError("The specified `response_mode` is not supported.")
        if self.response_mode == "plain":
            return None
        # bool(): enable_thinking=None is schema-valid and means "send no thinking control"
        # — no reasoning output is expected, so the response shape is the same as False.
        return RESPONSE_FORMAT_FOR_ANSWER[
            (self.answer_format, bool(self.enable_thinking))
        ]

    _NON_KWARG_FIELDS: ClassVar[frozenset[str]] = frozenset({
        "local_oracle_predictions_dirpath",
        "api_key",
    })

    @property
    def consult_kwargs(self) -> dict:
        kwargs_dict = self.model_dump(exclude=self._NON_KWARG_FIELDS)
        kwargs_dict["response_format"] = self.response_format
        return kwargs_dict

    @model_validator(mode="before")
    @classmethod
    def migrate_openrouter_model_name(cls, data: dict) -> dict:
        """Accept legacy `openrouter_model_name` key with a deprecation warning."""
        if isinstance(data, dict) and "openrouter_model_name" in data:
            migrated = data.copy()
            legacy_value = migrated.pop("openrouter_model_name")
            if "model_name" in migrated and migrated["model_name"] != legacy_value:
                raise ValueError(
                    "Conflicting oracle keys 'openrouter_model_name' and 'model_name'"
                )
            warn(
                "Config key 'openrouter_model_name' is DEPRECATED; use 'model_name' instead."
            )
            migrated["model_name"] = legacy_value
            return migrated
        return data

    @model_validator(mode="before")
    @classmethod
    def migrate_openrouter_apikey(cls, data: dict) -> dict:
        """Accept legacy `openrouter_apikey` key with a deprecation warning."""
        if isinstance(data, dict) and "openrouter_apikey" in data:
            migrated = data.copy()
            legacy_value = migrated.pop("openrouter_apikey")
            if "api_key" in migrated and migrated["api_key"] != legacy_value:
                raise ValueError(
                    "Conflicting oracle keys 'openrouter_apikey' and 'api_key'"
                )
            warn("Config key 'openrouter_apikey' is DEPRECATED; use 'api_key' instead.")
            migrated["api_key"] = legacy_value
            return migrated
        return data

    @model_validator(mode="after")
    def validate_api_key_reference(self) -> "OracleConfig":
        if self.api_key.startswith("ENV:") and not self.api_key[4:].strip():
            raise ValueError("oracle.api_key ENV reference must name an environment variable")
        if self.reasoning_token_budget is not None and self.reasoning_effort not in (None, ""):
            raise ValueError(
                "Set either oracle.reasoning_effort or oracle.reasoning_token_budget, not both"
            )
        if (
            self.reasoning_token_budget is not None
            and self.reasoning_token_budget >= self.max_completion_tokens
        ):
            raise ValueError(
                "oracle.reasoning_token_budget must be smaller than max_completion_tokens "
                "so the structured answer retains an output allowance"
            )
        return self


class OutputsConfig(StrictConfigModel):
    """Configuration for output directory paths."""
    logmapllm_output_dirpath: str = Field(min_length=1)
    logmap_initial_alignment_output_dirpath: str = Field(min_length=1)
    logmap_refined_alignment_output_dirpath: str = Field(min_length=1)


class PipelineConfig(StrictConfigModel):
    """Configuration for pipeline step modes."""
    align_ontologies: AlignMode = AlignMode.ALIGN
    build_oracle_prompts: PromptBuildMode = PromptBuildMode.BUILD
    consult_oracle: ConsultMode = ConsultMode.CONSULT
    refine_alignment: RefineMode = RefineMode.REFINE
    refinement_strategy: RefinementStrategy = RefinementStrategy.LOGMAP
    # Stop after Step 3: the oracle verdicts and the annotated M_ask files are the run's
    # product; refinement and evaluation are skipped and run_result.json records
    # stopped_after = "consultation". Requires consult_oracle consult/reuse and
    # evaluation.evaluate = false.
    stop_after_consultation: bool = False


class LogMapOAEIEvaluationOptions(StrictConfigModel):
    """`[evaluation.logmap_oaei]`: LogMap's own OAEI evaluator (relation-aware,
    orientation-insensitive, reference cells flagged `?` ignored); see
    evaluation/engines/logmap_oaei.py."""
    # the reference with every relation kept (reference.rdf or reference_full.tsv);
    # defaults to evaluation.reference_alignment_path
    reference_path: Optional[str] = None
    # report the 3-dp values LogMap prints as the block's P/R/F1 (printed_3dp always present)
    rounded: bool = False
    # oracle metrics: a candidate pair is positive when the reference holds it either way round
    orientation_insensitive: bool = True


class BioMLEvaluationOptions(StrictConfigModel):
    """`[evaluation.bioml]`: the OAEI Bio-ML protocols by edition; see
    evaluation/engines/bioml.py for the edition x setting table."""
    edition: Literal[2022, 2023, 2024, 2025, 2026] = 2025
    setting: Literal["unsupervised", "semi_supervised", "codabench"] = "unsupervised"
    reference_path: Optional[str] = None            # complete reference; defaults to reference_alignment_path
    reference_repaired_path: Optional[str] = None   # 2026: coherence-repaired reference with '?' flags
    test_reference_path: Optional[str] = None       # <= 2025 semi-supervised: refs_equiv/test.tsv
    train_alignment_path: Optional[str] = None      # <= 2025 semi-supervised: refs_equiv/train.tsv
    ignored_classes_path: Optional[str] = None      # >= 2023: use_in_alignment=false IRIs, one per line
    deprecated_classes_path: Optional[str] = None   # 2026: owl:deprecated IRIs, one per line
    split_path: Optional[str] = None                # 2026 CodaBench: split.tsv (SrcEntity, TgtEntity, split)

    @model_validator(mode="after")
    def _validate_edition_setting(self) -> "BioMLEvaluationOptions":
        if self.setting == "codabench":
            if self.edition != 2026:
                raise ValueError("evaluation.bioml.setting='codabench' exists for edition 2026 only")
            if not (self.split_path or "").strip():
                raise ValueError("evaluation.bioml.setting='codabench' requires evaluation.bioml.split_path")
        if self.setting == "semi_supervised" and self.edition == 2026:
            raise ValueError("evaluation.bioml edition 2026 has no semi_supervised setting; use codabench")
        if self.edition == 2026 and not (self.reference_repaired_path or "").strip():
            raise ValueError(
                "evaluation.bioml edition 2026 requires reference_repaired_path (the "
                "coherence-repaired reference is the track's headline)"
            )
        return self


class EvaluationConfig(StrictConfigModel):
    """Configuration for the optional evaluation step."""
    evaluate: bool = False
    reference_alignment_path: Optional[str] = None
    train_alignment_path: Optional[str] = None
    test_cands_path: Optional[str] = None
    metrics: list[str] | str = Field(default_factory=lambda: ["global", "oracle"])
    force_custom_eval: bool = True
    partial_reference: bool = False                 # for kg track: true
    stratified_by_entity_type: bool = False
    stratified_class_property: bool = False
    jvm_memory: str = Field(default="8g", pattern=r"^[1-9][0-9]*[mMgG]$")
    # Evaluation engines. Unset (the default) keeps the historical selection for the
    # `global` block: partial_reference -> partial_reference, else force_custom_eval ->
    # custom, else deeponto when importable. When set, the first entry is that primary
    # engine (it must be 'partial_reference' iff partial_reference is true; 'deeponto'
    # here overrides force_custom_eval, which is kept for backward compatibility only) and
    # every further entry adds a `global_<engine>` (and `oracle_<engine>`) block computed
    # under that engine's reference convention; options live in the per-engine tables.
    engines: Optional[list[str]] = None
    logmap_oaei: Optional[LogMapOAEIEvaluationOptions] = None
    bioml: Optional[BioMLEvaluationOptions] = None

    @model_validator(mode="after")
    def normalise_metrics(self):
        """Accept comma-separated string or list for metrics."""
        if isinstance(self.metrics, str):
            self.metrics = [m.strip() for m in self.metrics.split(",")]
        else:
            self.metrics = [str(m).strip() for m in self.metrics]
        if not self.metrics or any(not metric for metric in self.metrics):
            raise ValueError("evaluation.metrics must contain at least one metric name")
        unsupported = sorted(set(self.metrics) - {"global", "oracle"})
        if unsupported:
            raise ValueError(
                f"Unsupported evaluation metric(s): {unsupported}; supported metrics are "
                "['global', 'oracle']."
            )
        self.metrics = list(dict.fromkeys(self.metrics))
        if self.evaluate and not (self.reference_alignment_path or "").strip():
            raise ValueError(
                "evaluation.reference_alignment_path is required when evaluation.evaluate=true"
            )
        if self.stratified_by_entity_type and self.stratified_class_property:
            raise ValueError(
                "evaluation stratification modes are mutually exclusive; choose "
                "stratified_by_entity_type or stratified_class_property"
            )
        return self

    @model_validator(mode="after")
    def validate_engines(self) -> "EvaluationConfig":
        """The engine list is explicit and consistent with the legacy flags."""
        if self.engines is None:
            return self
        names = [str(name).strip() for name in self.engines]
        if not names or any(not name for name in names):
            raise ValueError("evaluation.engines must list at least one engine name")
        unknown = sorted(set(names) - set(EVALUATION_ENGINE_NAMES))
        if unknown:
            raise ValueError(
                f"Unknown evaluation engine(s): {unknown}; known engines are "
                f"{list(EVALUATION_ENGINE_NAMES)}"
            )
        if len(set(names)) != len(names):
            raise ValueError("evaluation.engines must not repeat an engine")
        if names[0] not in PRIMARY_EVALUATION_ENGINES:
            raise ValueError(
                f"evaluation.engines[0] must be a primary engine {list(PRIMARY_EVALUATION_ENGINES)} "
                f"(it produces the plain 'global' block); got {names[0]!r}"
            )
        if any(name in PRIMARY_EVALUATION_ENGINES for name in names[1:]):
            raise ValueError("only evaluation.engines[0] may be a primary engine")
        if self.partial_reference and names[0] != "partial_reference":
            raise ValueError(
                "evaluation.partial_reference=true requires evaluation.engines[0]='partial_reference'"
            )
        if names[0] == "partial_reference" and not self.partial_reference:
            raise ValueError(
                "evaluation.engines[0]='partial_reference' requires evaluation.partial_reference=true"
            )
        if "bioml" in names:
            options = self.bioml or BioMLEvaluationOptions()
            if options.setting == "semi_supervised" and not (
                (options.train_alignment_path or "").strip() or (self.train_alignment_path or "").strip()
            ):
                raise ValueError(
                    "evaluation.bioml.setting='semi_supervised' requires evaluation.bioml."
                    "train_alignment_path (or evaluation.train_alignment_path)"
                )
        self.engines = names
        return self

    def engine_options(self) -> dict[str, dict]:
        """Per-engine option tables for `evaluation.harness.evaluate_alignment`
        (unset options omitted so engine defaults apply)."""
        options: dict[str, dict] = {}
        for name in ("logmap_oaei", "bioml"):
            table = getattr(self, name)
            options[name] = {} if table is None else table.model_dump(exclude_none=True)
        return options


class ModelSelectionConfig(StrictConfigModel):
    """`[model_selection]`: automatic, self-supervised model selection; off unless
    `automatic = true`. LogMap's anchors are assumed correct, so every candidate is asked the
    prompts of up to `max_anchors` anchors plus one constructed negative each, and the one
    answering most correctly becomes the run's oracle (pipeline/model_selection.py). No
    reference alignment is read."""
    automatic: bool = False
    max_anchors: int = Field(default=10, ge=1)
    seed: int = 42
    # [[model_selection.candidates]]: partial [oracle] tables (model_name, base_url, api_key,
    # interaction_style, ...) merged over the base [oracle]; see candidate_oracle_configs()
    candidates: list[dict] = Field(default_factory=list)


class LogMapLLMConfig(StrictConfigModel):
    """
    Top-level configuration schema for LogMap-LLM; validates the entire TOML
    config structure on construction.
    TODO: should we not make all these default factories?
    """
    alignmentTask: AlignmentTaskConfig
    oracle: OracleConfig
    prompts: PromptTemplateConfig = Field(default_factory=PromptTemplateConfig)
    few_shot: FewShotConfig = Field(default_factory=FewShotConfig)
    outputs: OutputsConfig
    pipeline: PipelineConfig
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    # Optional, so configs (and the frozen job dumps derived from them) that never mention it
    # are unchanged.
    model_selection: Optional[ModelSelectionConfig] = None

    @property
    def automatic_model_selection(self) -> bool:
        return self.model_selection is not None and self.model_selection.automatic

    def candidate_oracle_configs(self) -> list[OracleConfig]:
        """The permitted model configurations: each candidate merged over the base [oracle]."""
        base = self.oracle.model_dump()
        candidates = self.model_selection.candidates if self.model_selection else []
        return [OracleConfig.model_validate({**base, **candidate}) for candidate in candidates]

    @model_validator(mode="after")
    def validate_model_selection(self) -> "LogMapLLMConfig":
        """Automatic selection needs at least two candidates, each a valid [oracle] table
        naming a model, and a consultation to hand the winner to."""
        if not self.automatic_model_selection:
            return self
        candidates = self.model_selection.candidates
        if len(candidates) < 2:
            raise ValueError(
                "model_selection.automatic=true requires at least two [[model_selection.candidates]]"
            )
        for index, candidate in enumerate(candidates):
            if not str(candidate.get("model_name", "")).strip():
                raise ValueError(f"model_selection.candidates[{index}] must name a model_name")
            try:
                OracleConfig.model_validate({**self.oracle.model_dump(), **candidate})
            except ValueError as exc:
                raise ValueError(
                    f"model_selection.candidates[{index}] is not a valid [oracle] table: {exc}"
                ) from exc
        if self.pipeline.consult_oracle != ConsultMode.CONSULT:
            raise ValueError(
                "model_selection.automatic=true requires pipeline.consult_oracle='consult' "
                "(the selected model must be the one consulted)"
            )
        return self

    @model_validator(mode="after")
    def validate_sibling_negative_requirements(self) -> "LogMapLLMConfig":
        """`paired-sibling-v2` must not run on an accidentally-defaulted sibling strategy.

        Pydantic already validates the strategy value; this checks the auto-resolution
        path: with no explicit strategy, the selector consults any registered domain
        override for `ontology_domain` and falls back to SBERT, silently running a
        different experiment under the intended one's name. Cross-section rule, so it
        lives here rather than on FewShotConfig; unreachable for any frozen historical
        config, since none can carry the layout.
        """
        if self.few_shot.rag_negative_layout != "paired-sibling-v2":
            return self
        if (self.prompts.sibling_strategy or "").strip() or (
            self.alignmentTask.ontology_domain or ""
        ).strip():
            resolved = resolve_sibling_strategy(
                self.prompts.sibling_strategy, self.alignmentTask.ontology_domain,
            )
            if resolved.is_embedding_based and not (
                self.prompts.sibling_model_revision or ""
            ).strip():
                raise ValueError(
                    f"few_shot.rag_negative_layout='paired-sibling-v2' resolves to the "
                    f"'{resolved.value}' embedding strategy, so prompts.sibling_model_revision "
                    "must pin an immutable model commit. An unpinned checkpoint can silently "
                    "change the experiment, and cannot be loaded at all from an offline cache "
                    "holding only the pinned snapshot."
                )
            return self
        raise ValueError(
            "few_shot.rag_negative_layout='paired-sibling-v2' ranks sibling candidates with "
            "the configured sibling strategy, but neither prompts.sibling_strategy nor "
            "alignmentTask.ontology_domain is set, so the strategy would silently default to "
            "'sbert'. Set prompts.sibling_strategy explicitly (alphanumeric, shortest_label, "
            "cls_transformer, sbert), or declare alignmentTask.ontology_domain so the resolution is "
            "recorded."
        )

    @model_validator(mode="after")
    def validate_pipeline_state(self) -> "LogMapLLMConfig":
        pipeline = self.pipeline
        if (
            self.few_shot.prebuilt_few_shot_bundle_path is not None
            and pipeline.consult_oracle == ConsultMode.CONSULT
            and pipeline.build_oracle_prompts != PromptBuildMode.BUILD
        ):
            raise ValueError(
                "few_shot.prebuilt_few_shot_bundle_path requires "
                "pipeline.build_oracle_prompts='build' so the bundle is strictly "
                "validated before consultation"
            )
        if (
            pipeline.build_oracle_prompts == PromptBuildMode.REUSE
            and pipeline.align_ontologies != AlignMode.REUSE
        ):
            raise ValueError(
                "pipeline.build_oracle_prompts='reuse' requires "
                "pipeline.align_ontologies='reuse' so prompts cannot be paired with a fresh alignment"
            )
        if (
            pipeline.build_oracle_prompts == PromptBuildMode.BUILD
            and pipeline.align_ontologies == AlignMode.BYPASS
        ):
            raise ValueError(
                "pipeline.build_oracle_prompts='build' requires an alignment (align or reuse)"
            )
        if (
            pipeline.consult_oracle == ConsultMode.CONSULT
            and pipeline.build_oracle_prompts == PromptBuildMode.BYPASS
        ):
            raise ValueError(
                "pipeline.consult_oracle='consult' requires prompts (build or reuse)"
            )
        if (
            self.evaluation.evaluate
            and "oracle" in self.evaluation.metrics
            and pipeline.consult_oracle not in {ConsultMode.CONSULT, ConsultMode.REUSE}
        ):
            raise ValueError(
                "evaluation metric 'oracle' requires pipeline.consult_oracle to be "
                "'consult' or 'reuse'; use metrics=['global'] for a plain-LogMap baseline"
            )
        external = (self.alignmentTask.external_mappings_filepath or "").strip()
        if pipeline.align_ontologies == AlignMode.EXTERNAL and not external:
            raise ValueError(
                "pipeline.align_ontologies='external' requires "
                "alignmentTask.external_mappings_filepath (the mappings to annotate)"
            )
        if external and pipeline.align_ontologies != AlignMode.EXTERNAL:
            raise ValueError(
                "alignmentTask.external_mappings_filepath is only read when "
                "pipeline.align_ontologies='external'; remove it or select that mode"
            )
        if pipeline.stop_after_consultation:
            if pipeline.consult_oracle not in {ConsultMode.CONSULT, ConsultMode.REUSE}:
                raise ValueError(
                    "pipeline.stop_after_consultation=true requires pipeline.consult_oracle "
                    "'consult' or 'reuse' (there is no consultation to stop after otherwise)"
                )
            if self.evaluation.evaluate:
                raise ValueError(
                    "pipeline.stop_after_consultation=true skips refinement and evaluation; "
                    "set evaluation.evaluate=false (a run cannot claim an evaluation it does not perform)"
                )
        return self


def validate_config(config_dict: dict) -> LogMapLLMConfig:
    """Validate the config dict and return a typed config object."""
    return LogMapLLMConfig.model_validate(config_dict)
