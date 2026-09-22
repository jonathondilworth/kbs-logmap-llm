# Configuration reference

A single TOML file drives every run: `python -m logmap_llm --config <path>` (default `configs/default_config.toml`). The schema is a Pydantic model tree in `logmap_llm/config/schema.py` (`LogMapLLMConfig`, one model per TOML table); loading lives in `logmap_llm/config/loader.py`. Every model inherits `StrictConfigModel` with `extra="forbid"`, so a misspelt key anywhere in the file is a hard validation error, never silently ignored.

`load_config()` resolves exactly the path it is given (no search path, no env expansion), parses with stdlib `tomllib`, applies the CLI reuse overrides onto the `pipeline` table, and validates. The CLI wrapper `load_and_validate_config()` renders validation errors (one line per error, `section x field: message`) and exits 1. `--reuse-align` forces `pipeline.align_ontologies = "reuse"`; `--reuse-prompts` forces `pipeline.build_oracle_prompts = "reuse"` *and* `pipeline.align_ontologies = "reuse"` — prompts can never be reused against a fresh alignment. See [running the pipeline](pipeline.md) for the other CLI flags (`--no-cache`, `--run-root`), which are applied outside the loader.

## Minimal example

Only `[alignmentTask]`, `[oracle]`, and `[outputs]` need to appear; `[prompts]`, `[few_shot]`, `[evaluation]` are optional tables with defaults, and the loader inserts an empty `[pipeline]` table if it is absent. (If you call `schema.validate_config()` on a hand-built dict, bypassing the loader, `pipeline` is a required key.)

```toml
[alignmentTask]
task_name = "mouse-human"
onto_source_filepath = "data/mouse.owl"
onto_target_filepath = "data/human.owl"

[oracle]
model_name = "qwen/qwen3-32b"
api_key = "ENV:OPENROUTER_API_KEY"

[outputs]
logmapllm_output_dirpath = "output/logmapllm"
logmap_initial_alignment_output_dirpath = "output/initial-alignment"
logmap_refined_alignment_output_dirpath = "output/refined-alignment"

[evaluation]
evaluate = true
reference_alignment_path = "data/reference.rdf"
```

## [alignmentTask]

Required. Note the camelCase table name — this is the one table that is not snake_case; `[alignment_task]` is rejected as an unknown key.

| Field | Type / default | Controls |
|---|---|---|
| `task_name` | str, required | Run identifier; must match `^[A-Za-z0-9][A-Za-z0-9_-]*$` |
| `onto_source_filepath` | str, required | Source ontology path |
| `onto_target_filepath` | str, required | Target ontology path |
| `generate_extended_mappings_to_ask_oracle` | bool, `false` | Extended candidate generation for M_ask |
| `stub_import_iris` | list[str], `[]` | `owl:imports` IRIs owlready2 must not follow (needed only where an import 404s) |
| `logmap_parameters_dirpath` | str, `""` | Directory of LogMap parameter files |
| `logmap_jvm_memory` | str, `"8g"` | LogMap JVM heap; must match `^[1-9][0-9]*[mMgG]$` |
| `ontology_domain` | str or null, null | Free-text domain label consulted by sibling-strategy auto-resolution |
| `ontology_vocabulary` | str, `"default"` | Annotation/URI convention preset; see the [ontology access layer](ontology.md) |
| `external_mappings_filepath` | str or null, null | The mappings to annotate when `pipeline.align_ontologies = "external"` (pipe `.txt`, TSV or OAEI RDF); required in that mode, rejected otherwise |

`ontology_vocabulary` is resolved against the preset registry at load time (`_validate_vocabulary_preset` calls `get_preset`), so an unknown name fails immediately with the list of known presets: `default`, `dbpedia_family` (OAEI Knowledge Graph track), `multilingual_skos` (OAEI Digital Humanities track).

## [oracle]

Required. Endpoint, sampling, and output-shape settings for the [LLM oracle](oracle.md).

| Field | Type / default | Controls |
|---|---|---|
| `model_name` | str, required | Model identifier |
| `api_key` | str, `"EMPTY"` | `"EMPTY"` for local vLLM, a literal key, or `"ENV:VARNAME"` |
| `base_url` | str, `"https://openrouter.ai/api/v1"` | API endpoint; e.g. `http://localhost:8000/v1` for vLLM/SGLang |
| `interaction_style` | str, `"auto"` | Client style: `auto`, `openai_chat_completions_parse_structured_output`, `openrouter`, `local`, `vllm`, `sglang` |
| `answer_format` | `"true_false"` \| `"yes_no"`, `"true_false"` | Answer vocabulary |
| `response_mode` | `"structured"` \| `"plain"`, `"structured"` | Structured JSON vs plain-text answers |
| `enable_thinking` | bool or null, `false` | Thinking mode for supported models; null sends no thinking control |
| `max_completion_tokens` | int ≥ 1, `2048` | Completion cap |
| `temperature` / `top_p` | float, `0.0` / `1.0` | Sampling (0–2 / (0, 1]) |
| `seed` | int or null, null | API sampling seed (strict int, no coercion) |
| `max_workers` | int ≥ 1, `24` | Concurrent request threads |
| `request_timeout_seconds` / `connect_timeout_seconds` | float, `120.0` / `15.0` | Per-request / connect timeouts |
| `transient_retries` | int ≥ 0, `2` | Retries on transient failure (strict int) |
| `failure_tolerance` | int ≥ 1 or null, null | Floor on the cumulative-failure abort threshold (effective threshold is at least 5% of the batch; default floor 5) |
| `request_logprobs` | bool, `true` | Request token logprobs (used for confidence) |
| `reasoning_effort` | str or null, null | Reasoning-effort control |
| `reasoning_token_budget` | int ≥ 1 or null, null | Reasoning token budget |
| `openrouter_provider` | str or null, null | Pin an OpenRouter provider |
| `openrouter_allow_fallbacks` | bool, `false` | Allow provider fallbacks |
| `openrouter_require_parameters` | bool, `true` | Require providers to honour parameters |
| `supports_chat_template_kwargs` | bool or null, null | Whether the endpoint accepts `chat_template_kwargs` |
| `local_oracle_predictions_dirpath` | str, `""` | Precomputed predictions for `pipeline.consult_oracle = "local"` |

An `api_key` of the form `ENV:VARNAME` is a sentinel: the schema only checks that a variable is named, and the secret is read from the environment at consultation time (`oracle/consultation.py::_resolve_api_key`), so the real key is never written into config files, manifests, or results. `reasoning_effort` and `reasoning_token_budget` are mutually exclusive, and the budget must be strictly smaller than `max_completion_tokens` so the answer retains an output allowance. The `(answer_format, enable_thinking)` pair selects the structured `response_format` model when `response_mode = "structured"`; plain mode sends none. Legacy keys `openrouter_model_name` and `openrouter_apikey` are still migrated with a deprecation warning; supplying both old and new keys with different values is an error.

## [prompts]

Optional. Selects the [prompt templates](oracle.md) per entity lane and configures sibling ranking.

| Field | Type / default | Controls |
|---|---|---|
| `cls_dev_prompt_template_name` | str, `"class_equivalence"` | Class-lane developer (system) template |
| `cls_usr_prompt_template_name` | str, `"synonyms_only"` | Class-lane user template |
| `prop_dev_prompt_template_name` | str or null, `"property_equivalence"` | Property-lane developer template |
| `prop_usr_prompt_template_name` | str or null, null | Property-lane user template |
| `dprop_usr_prompt_template_name` | str or null, null | Dedicated data-property user template; when unset, data properties use `prop_usr_prompt_template_name` |
| `inst_dev_prompt_template_name` | str or null, `"instance_equivalence"` | Instance-lane developer template |
| `inst_usr_prompt_template_name` | str or null, null | Instance-lane user template |
| `sibling_strategy` | `"alphanumeric"` \| `"shortest_label"` \| `"cls_transformer"` \| `"sbert"` or null, null | Sibling-ranking strategy |
| `sibling_model` | str or null, null | Embedding-model override (embedding strategies only) |
| `sibling_model_revision` | str or null, null | Immutable checkpoint commit for the sibling model |
| `sibling_max_candidates` | int ≥ 1 or null, null | Cap on candidate siblings before ranking; null uses `DEFAULT_MAX_SIBLING_CANDIDATES = 50` |
| `sibling_encoder_device` | str or null, null | Ranker device; null is cuda if available, else cpu |

Configuring a property or data-property *user* template requires a non-empty `prop_dev_prompt_template_name`; likewise an instance user template requires `inst_dev_prompt_template_name`. When `sibling_strategy` is unset, `ontology/sibling_strategy.py::resolve_sibling_strategy` consults any registered domain override for `alignmentTask.ontology_domain` (the override table is empty in a stock install) and falls back to `sbert`. Declare `sibling_encoder_device` on multi-host runs: cuda and cpu can resolve near-ties differently and the device is not part of the run identity.

## [few_shot]

Optional. Governs few-shot demonstration retrieval for oracle prompts; see [RAG few-shot retrieval](rag.md).

| Field | Type / default | Controls |
|---|---|---|
| `few_shot_k` | int ≥ 0, `0` | Number of examples per query; 0 disables few-shot entirely |
| `few_shot_seed` | int, `42` | Sampling seed |
| `few_shot_negative_strategy` | str, `"hard"` | One of `hard`, `random`, `hard-similar` (legacy samplers) or `query-rag`, `static-hard`, `static-random`, `zero-shot` (RAG modes) |
| `rag_negative_layout` | `"paired-sibling-v2"` \| `"paired-donor-v2"` \| `"donor-cross-v1"` or null, null | How the k/2 pseudo-negatives are constructed |
| `rag_encoder_kind` | `"cls_transformer"` \| `"sbert"` \| `"hashing"`, `"cls_transformer"` | Retrieval encoder family |
| `rag_encoder_model` | str, `""` | Encoder checkpoint; no in-source default — it is part of the run's experimental identity |
| `rag_encoder_revision` | str or null, null | Immutable model revision |
| `rag_encoder_device` | str or null, null | Retrieval-side device |
| `rag_encoder_max_length` | int ≥ 1, `64` | Encoder max sequence length |
| `rag_failure_policy` | `"error"` \| `"record_zero_shot"`, `"error"` | Behaviour when retrieval fails |
| `rag_cache_dir` | str or null, null | RAG cache directory |
| `prebuilt_few_shot_bundle_path` | str or null, null | Already-rendered cross-task donor-negative bundle |

This table carries most of the interacting rules. An unknown negative strategy fails loudly at load — the schema refuses to fall back to random. Semantic retrieval (`few_shot_k > 0` with a `cls_transformer` or `sbert` encoder) requires a pinned `rag_encoder_revision`; the alternative is explicitly choosing `rag_encoder_kind = "hashing"` as a baseline. A `rag_negative_layout` with `few_shot_k = 0` is rejected; the `paired-*` layouts pair one constructed negative with each retrieved positive, so they require an even `few_shot_k` and a ranked-retrieval strategy (`hard`, `hard-similar`, `query-rag`, `static-hard`). `donor-cross-v1` is frozen: it exists only to re-render completed campaigns byte-for-byte, and new campaign specs must not select it.

The converse rule — a layout is required whenever `few_shot_k > 0` — is deliberately *not* in the schema, because frozen per-job configs inside completed batches carry `few_shot_k > 0` with no layout and must stay loadable by aggregation tooling. It is enforced where campaigns are created and executed instead: `experiments/plan.py` at generation and `pipeline/stage_two.py` at run time. Similarly, an empty `rag_encoder_model` passes the schema but is refused later by `pipeline/rag_fewshot.py::build_rag_encoder` for semantic encoder kinds.

`prebuilt_few_shot_bundle_path` requires `few_shot_k = 4` exactly, strategy `query-rag`, `rag_failure_policy = "error"`, layout `donor-cross-v1` or unset, and — cross-section — `pipeline.build_oracle_prompts = "build"` whenever `consult_oracle = "consult"`, so the bundle is strictly validated before consultation.

## [outputs]

Required. Three non-empty directory paths, no defaults: `logmapllm_output_dirpath`, `logmap_initial_alignment_output_dirpath`, `logmap_refined_alignment_output_dirpath`. The `--run-root` CLI flag re-roots all three under one directory.

## [pipeline]

Per-step run modes; values are the enums in `logmap_llm/constants.py`. Omittable via the loaders since every field has a default.

| Field | Values (default first) | Step |
|---|---|---|
| `align_ontologies` | `align`, `reuse`, `external`, `bypass` | 1: classical LogMap alignment; `external` = an external mapping file is the M_ask (annotate mode) |
| `build_oracle_prompts` | `build`, `reuse`, `bypass` | 2: prompt construction |
| `consult_oracle` | `consult`, `reuse`, `local`, `bypass` | 3: LLM consultation |
| `refine_alignment` | `refine`, `bypass` | 4: alignment refinement |
| `refinement_strategy` | `logmap`, `python` | 4: full LogMap refinement vs approximate Python set-union (`orchestration.py::_kg_refine_in_python`, for tasks LogMap refinement cannot complete, e.g. OAEI 2025 KG instances) |
| `stop_after_consultation` | `false`, `true` | end the run after Step 3 and the annotated M_ask files; requires `consult_oracle` consult/reuse and `evaluation.evaluate = false` |

The top-level validator `validate_pipeline_state` enforces step consistency: `build_oracle_prompts = "reuse"` requires `align_ontologies = "reuse"`; `build_oracle_prompts = "build"` needs an alignment (`align` or `reuse`, not `bypass`); `consult_oracle = "consult"` needs prompts (`build` or `reuse`).

## [evaluation]

Optional; see [evaluation](evaluation.md).

| Field | Type / default | Controls |
|---|---|---|
| `evaluate` | bool, `false` | Run Step 5 at all |
| `reference_alignment_path` | str or null, null | Gold reference; required non-empty when `evaluate = true` |
| `train_alignment_path` / `test_cands_path` | str or null, null | Train split / test candidates for split-aware scoring |
| `metrics` | list or comma-separated str, `["global", "oracle"]` | Only `global` and `oracle` are supported |
| `force_custom_eval` | bool, `true` | Force the custom evaluator over DeepOnto |
| `partial_reference` | bool, `false` | Treat the reference as partial (KG track: true) |
| `stratified_by_entity_type` / `stratified_class_property` | bool, `false` | Result stratification; mutually exclusive |
| `jvm_memory` | str, `"8g"` | JVM heap for evaluation tooling; same pattern as `logmap_jvm_memory` |
| `engines` | list[str] or null, null | Engine list: a primary engine first, then track-faithful engines (`logmap_oaei`, `bioml`); see [evaluation](evaluation.md) |
| `[evaluation.logmap_oaei]` | table, unset | `reference_path`, `rounded`, `orientation_insensitive` |
| `[evaluation.bioml]` | table, unset | `edition`, `setting`, `reference_path`, `reference_repaired_path`, `test_reference_path`, `train_alignment_path`, `ignored_classes_path`, `deprecated_classes_path`, `split_path` |

`metrics` is normalised to a stripped, order-preserving, deduplicated list. The `oracle` metric scores the LLM as a binary classifier and therefore requires `pipeline.consult_oracle` to be `consult` or `reuse`; use `metrics = ["global"]` for a plain-LogMap baseline.

## Cross-section rules at a glance

Two `@model_validator` checks on `LogMapLLMConfig` tie sections together. `validate_pipeline_state` covers the step-mode consistency and prebuilt-bundle rules listed above, plus the annotate-mode rules (`external` needs `alignmentTask.external_mappings_filepath` and vice versa; `stop_after_consultation` needs a consultation to stop after and `evaluation.evaluate = false`); `EvaluationConfig.validate_engines` checks the engine list (`partial_reference` first exactly when `partial_reference = true`, a semi-supervised `bioml` setting needs a training split). `validate_sibling_negative_requirements` guards `rag_negative_layout = "paired-sibling-v2"`: either `prompts.sibling_strategy` or `alignmentTask.ontology_domain` must be set (otherwise the strategy would silently default to `sbert` and run a different experiment under the intended one's name), and if the resolved strategy is embedding-based, `prompts.sibling_model_revision` must pin an immutable commit.

For programmatic use, `config/loader.py` exposes `load_config()` (errors propagate — the library seam, used by the RAG bundle planner in `oracle/rag/bundle.py`) and `print_config_summary()` (flattened dotted keys, `oracle.api_key` masked).
