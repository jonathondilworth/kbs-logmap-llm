# The LLM oracle

LogMap hands the pipeline a set of uncertain candidate mappings, M_ask, that it could neither accept nor reject on its own. The oracle subsystem turns each candidate into a binary question — is the source entity equivalent to (or subsumed by) the target entity? — sends it to an LLM over an OpenAI-compatible chat-completions API, and appends the verdicts to the M_ask DataFrame as `Oracle_*` columns. Two modules split the work: `logmap_llm/oracle/manager.py` owns the API client, request construction and answer parsing; `logmap_llm/oracle/consultation.py` owns the campaign — coverage guards, threading, retries, abort policy and result assembly. The prompts themselves are built earlier, in the Stage 2 subprocess (see [running the pipeline](pipeline.md)), and consultation only replays them.

## Consultation flow

Step 3 of the pipeline (`logmap_llm/pipeline/orchestration.py::consult_oracle`) takes the prompt artifact (`mappings_to_ask_oracle_user_prompts.json`) loaded in Step 2, resolves developer prompts and any few-shot artifact, then calls `consult_oracle_for_mappings_to_ask` or, when the artifact's keys carry a direction suffix (built from a bidirectional template), `consult_oracle_bidirectional`. Prompts are keyed `SRC_URI|TGT_URI` (bidirectional runs add a `SRC_URI|TGT_URI|REVERSE` companion per candidate). Before any request is sent, coverage guards check that the prompt keys exactly match the M_ask rows — and, in bidirectional mode, that every forward key has its reverse — raising `ValueError` otherwise rather than consulting a partial candidate set.

A single `OracleConsultationManager` is shared by all worker threads. Its message prefix — the developer message plus any shared few-shot turns — is frozen (`freeze_messages()`) before fan-out, after which it is immutable; each call assembles its own message list from local copies. Consultations run on a `ThreadPoolExecutor` with `oracle.max_workers` threads (default 24), keeping a constant in-flight window: the pool is primed with one future per worker and each completion submits the next prompt.

Results are reassembled in M_ask row order and written atomically to `mappings_to_ask_with_oracle_predictions.csv`. A unidirectional run records, per candidate: `Oracle_prediction`, `Oracle_confidence`, token counts, finish reasons, the serving provider and model, and the raw reply text. A bidirectional run asks both subsumption directions and accepts a candidate only when both come back true; `Oracle_confidence` is the minimum of the two directional confidences, and `Oracle_fwd_prediction` / `Oracle_rev_prediction` (with their confidences) are kept so the conjunction can be decomposed afterwards.

## Providers and models

Any endpoint speaking the OpenAI chat-completions protocol works: OpenRouter, vLLM, SGLang, or OpenAI itself. `oracle.model_name` is required; `oracle.base_url` defaults to `https://openrouter.ai/api/v1`. `oracle.interaction_style` selects the request dialect:

| Style | Behaviour |
| --- | --- |
| `auto` (default) | `openrouter` when `openrouter.ai` occurs in the base URL, else the vLLM path |
| `openrouter` | `create()` with JSON-schema guidance on the wire, reply parsed locally |
| `vllm`, `sglang` | `create()` with strict guided JSON-schema decoding |
| `openai_chat_completions_parse_structured_output` | `client.chat.completions.parse()` structured outputs |
| `local` | plain text only; combining it with structured mode raises at construction |

The SDK client is built with `max_retries=0` — retry policy lives entirely in `consultation.py` — and a timeout of `oracle.request_timeout_seconds` (default 120) with `oracle.connect_timeout_seconds` (default 15) to connect. Sampling is controlled by `oracle.temperature` (default 0.0), `oracle.top_p` (default 1.0) and `oracle.seed`.

Provider-specific extras go in `extra_body`. On OpenRouter: reasoning controls (`oracle.reasoning_effort`, or `oracle.reasoning_token_budget`; the literal effort `"none"` sends `{"enabled": false}`) and route pinning (`oracle.openrouter_provider`, with `openrouter_allow_fallbacks` defaulting to false and `openrouter_require_parameters` to true, so a run cannot silently migrate to a provider that ignores its parameters). On vLLM/SGLang: `chat_template_kwargs = {"enable_thinking": ...}` carrying `oracle.enable_thinking` (the config default `false` is sent too), unless the model cannot accept the parameter (`supports_chat_template_kwargs`, auto-detected as false for Mistral-family model names).

## Credentials

`oracle.api_key` accepts a literal key, `"EMPTY"` for local servers, or the sentinel `ENV:VARNAME`. The sentinel is resolved from `os.environ` only at manager construction (`_resolve_api_key` in `consultation.py`), so the real key never appears in `config.toml`, in batch manifests, or in run artifacts: `_configured_oracle_params` (the oracle parameter block sealed into the run result) omits `api_key` entirely, `OracleConfig` excludes it from `consult_kwargs`, and the config summary printer masks it. If the named variable is unset the run fails immediately with a message advising you to `source <root>/.secrets/env` before running cloud experiments.

## Response modes, formats and logprobs

Two orthogonal settings shape the answer. `oracle.response_mode` picks the container: `structured` (default) requests a JSON object and parses it against a Pydantic format; `plain` sends no `response_format` and parses free text. `oracle.answer_format` picks the vocabulary: `true_false` (default, `{"answer": true}`) or `yes_no`. With `oracle.enable_thinking` the `WithReasoning` format variants are used instead (see `RESPONSE_FORMAT_FOR_ANSWER` in `logmap_llm/constants.py`).

Structured replies that fail JSON or schema validation on the `create()` paths are not discarded: the manager falls back to `_parse_plain_text_answer`, a staged cascade that tries an exact answer token, then JSON (bare boolean or a single-key `{"answer": ...}` object), then a clause scan with negation-scope handling. A reply the cascade cannot resolve raises `ValueError` ("Ambiguous LLM response") and becomes an error record — it is never guessed.

When `oracle.request_logprobs` is true (the default), every request asks for `logprobs` with `top_logprobs=3`, and `calculate_logprobs_confidence` derives `Oracle_confidence` as the probability the model assigned to the answer it actually gave (so values below 0.5 are possible). Not every provider accepts the parameter: the first request rejected specifically for `logprobs` trips a run-wide downgrade latch — logprobs are disabled for the remainder of the run, the failed call is retried once without them, and the downgrade is recorded in `df.attrs["oracle_capabilities"]` and in the sealed oracle parameters (`logprobs_requested` vs `logprobs_effective`). After a downgrade, `Oracle_confidence` is NaN; refinement decisions rest on the boolean verdict, not on this diagnostic column.

## Errors, retries and skips

The per-mapping worker (`consult_oracle_for_mapping`) never lets an exception escape: every failure becomes an error record `(key, "error", NaN, unknown tokens)`. Only transient failures are retried — connection errors and HTTP 429/500/502/503/504 — up to `oracle.transient_retries` times (default 2, so three attempts), with exponential backoff whose jitter is derived from `sha256(seed:key:attempt)`, making delays reproducible under a fixed `oracle.seed`. A `BadRequestError` or an ambiguous reply fails immediately.

A campaign-level circuit breaker aborts the whole consultation after 5 consecutive errors, or when cumulative errors reach `max(oracle.failure_tolerance or 5, 5% of the campaign)`. On abort the entry points return `None` and the runner refuses to report a partial condition as successful. Candidates that finish the campaign without any result are marked `skipped` in `Oracle_prediction`; the same fate awaits candidates the prompt builder could not render (unresolvable URIs, mixed-type pairs, or a lane with no template configured).

Downstream, both `error` and `skipped` count as rejects: the accept mask (`prediction_is_true_mask` in `logmap_llm/utils/data.py`) treats anything that is not `True` as not accepted, so refinement never adds those mappings to the final alignment. The [evaluation](evaluation.md) harness additionally maps them to `prediction: None`, counts them under `errors`, and excludes them from the oracle confusion matrix rather than scoring them as deliberate rejections.

## Prompt assembly

User prompts are built in the Stage 2 subprocess (`logmap_llm/pipeline/stage_two.py`) by `build_oracle_user_prompts` (or `_bidirectional`) in `logmap_llm/oracle/prompts/templates.py`. Templates live in a decorated registry — `registry.register(name, entity_type=..., bidirectional=..., requires_siblings=...)` — and are selected per lane in the `[prompts]` config section:

- `cls_usr_prompt_template_name` — class lane (default `synonyms_only`)
- `prop_usr_prompt_template_name` / `dprop_usr_prompt_template_name` — object- and data-property lanes (default unset)
- `inst_usr_prompt_template_name` — instance lane (default unset)

A lane with no configured template skips its candidates (recorded, and rejected downstream as above). When no dedicated data-property template is set, `DPROP` candidates use the object-property template, which renders a datatype-aware domain/range clause instead.

Routing (`logmap_llm/oracle/prompts/routing.py`) trusts LogMap's own entity-type tag in the fifth M_ask column (`CLS`/`OPROP`/`DPROP`/`INST`) over types re-derived from the ontology, because re-derivation collapses data properties into the object-property lane and mis-routes OWL2-punned URIs; ontology-derived types are only a fallback when the tag is absent. Entities are resolved through the [ontology access layer](ontology.md), property-first for property-tagged rows so undeclared ABox predicates still resolve.

Each template renders both entities' context with the deterministic helpers in `logmap_llm/oracle/prompts/formatting.py`: parents and synonyms, sibling context ("Other \"X\" concepts include ..."), OWL restrictions verbalised ("It relates to some ..."), domain/range clauses for properties, and types plus attribute clauses for instances — with shared-predicate selection optionally ranked by predicate entropy. Name choices use `min()`/`sorted()` throughout so prompt text is byte-stable under `PYTHONHASHSEED`. A frozen `PromptContext` (`logmap_llm/oracle/prompts/context.py`), built once per run from the config, supplies the answer format, response mode and the optional `alignmentTask.ontology_domain` preamble to every template. An experimental natural-language verbaliser for instance attributes exists in `oracle/prompts/nl_verbaliser.py` ("whose homeworld is \"Tatooine\"" rather than "is \"homeworld\" \"Tatooine\"") but is not wired into the pipeline; it can be injected via `PromptContext.from_config(instance_fmt_fn=...)`.

### Developer messages

The system prompt comes from the registry in `logmap_llm/oracle/prompts/developer.py`: `get_developer_prompt(name, answer_format, response_mode)` concatenates a named persona (`class_equivalence`, `property_equivalence`, `instance_equivalence`, `domain_expert`, ...) with the response-format instruction matching the configured mode; unknown names fail loudly. The class-lane message (`prompts.cls_dev_prompt_template_name`) becomes the frozen default; when property or instance lanes are configured, orchestration also builds a map from entity type to developer text, and consultation substitutes the matching message per call based on the candidate's M_ask tag. The message is sent under role `system` for vLLM/SGLang and `developer` otherwise.

### Few-shot examples: shared vs per-query

With `few_shot.few_shot_k > 0`, Step 3 requires the `few_shot_examples.json` artifact written by Stage 2 and refuses to run without it. The artifact has two shapes, dispatched by type:

- a flat list of `[user, assistant]` pairs — legacy static examples, baked once into the frozen message prefix and shared by every consultation;
- a dict keyed by M_ask key — query-specific examples from [RAG retrieval](rag.md), looked up per candidate and spliced between the frozen prefix and the final user message on each call, never stored on the manager.

Both kinds are rendered with the same template function as the live query, so a demonstration looks exactly like the question that follows it, and assistant answers come from the same `(answer_format, response_mode)` vocabulary the oracle is instructed to use. The static builder (`FewShotExampleBuilder` in `oracle/prompts/few_shot.py`) draws positives from the task's `train.tsv` and constructs hard negatives by replacing the target with one of its ontology siblings (falling back to a random column swap), excluding any pair that appears in M_ask so demonstrations cannot leak the answer to a live candidate.
