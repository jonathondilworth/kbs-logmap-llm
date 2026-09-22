'''
logmap_llm.oracle.consultation
This module contains functionality that supports consulting
(interacting with) LLM Oracles.
'''
from __future__ import annotations

import hashlib
import time
from logmap_llm.oracle.manager import (
    OracleConsultationManager,
    OracleTransientResponseError,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from logmap_llm.config.schema import OracleConfig
from logmap_llm.constants import (
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    TokensUsage,
    PAIRS_SEPARATOR,
    POSITIVE_TOKENS,
    NEGATIVE_TOKENS,
    DEFAULT_FAILURE_TOLERANCE_FLOOR,
    DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.logging import (
    warn,
    warning,
    critical,
    debug,
)
from tqdm import tqdm
import numpy as np
from openai import APIConnectionError, APIStatusError, BadRequestError
import pandas as pd

_TRANSIENT_HTTP_STATUSES = frozenset({429, 500, 502, 503, 504})


def _require_forward_prompt_coverage(
    m_ask_prompts: dict[str, str],
    m_ask_init_alignment_df: pd.DataFrame,
) -> None:
    """Fail before inference unless prompt keys exactly cover the M_ask candidates.

    Reverse keys used by bidirectional consultation are deliberately ignored here; their
    corresponding forward key is the candidate identity shared by both consultation modes.
    """
    expected = {
        str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        for _, row in m_ask_init_alignment_df.iterrows()
    }
    reverse_suffix = PAIRS_SEPARATOR + "REVERSE"
    actual = {
        str(key)
        for key in m_ask_prompts
        if not str(key).endswith(reverse_suffix)
    }
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        def preview(values: list[str]) -> list[str]:
            suffix = [f"... ({len(values)} total)"] if len(values) > 3 else []
            return values[:3] + suffix

        raise ValueError(
            "Prompt coverage does not match M_ask; refusing to send any oracle requests. "
            f"missing={preview(missing)}, unexpected={preview(unexpected)}"
        )


def _require_reverse_prompt_coverage(m_ask_prompts: dict[str, str],
                                     forward_only_keys: set[str] | None = None) -> None:
    """Bidirectional consultation needs both directions per candidate; fail before inference.

    The forward guard deliberately ignores reverse keys, so this validates them: a
    forward-only candidate would be consulted (and paid for) and then silently marked
    'skipped' by the aggregation. Same contract as the forward guard: refuse to send
    anything.
    """
    reverse_suffix = PAIRS_SEPARATOR + "REVERSE"
    keys = {str(key) for key in m_ask_prompts}
    forward = {key for key in keys if not key.endswith(reverse_suffix)}
    reverse_bases = {key[: -len(reverse_suffix)] for key in keys if key.endswith(reverse_suffix)}
    # Hybrid lanes: property/instance candidates are consulted once (forward only) and are
    # exempt from the reverse requirement; a reverse key on one of them is still an error.
    exempt = set(forward_only_keys or ())
    missing = sorted(forward - reverse_bases - exempt)
    orphaned = sorted((reverse_bases - forward) | (reverse_bases & exempt))
    if missing or orphaned:
        raise ValueError(
            "Bidirectional prompt coverage is asymmetric; refusing to send any oracle "
            f"requests. forward-without-reverse={missing[:3]} ({len(missing)} total), "
            f"reverse-without-forward={orphaned[:3]} ({len(orphaned)} total)"
        )


def _is_transient_oracle_error(exc: BaseException) -> bool:
    """Return true only for transport failures, the campaign's HTTP allowlist and a
    200 response without choices (OpenRouter's in-body provider error)."""
    if isinstance(exc, (APIConnectionError, OracleTransientResponseError)):
        return True
    return (
        isinstance(exc, APIStatusError)
        and exc.status_code in _TRANSIENT_HTTP_STATUSES
    )


def _retry_delay_seconds(key: str, retry_number: int, seed: int | None) -> float:
    """Deterministic exponential backoff with up to 25% keyed jitter."""
    base = 0.5 * (2 ** min(retry_number - 1, 4))
    material = f"{0 if seed is None else seed}:{key}:{retry_number}".encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    return base * (1.0 + 0.25 * fraction)


def _consultation_failure_record(key: str, exc: BaseException):
    warning(f"Consultation for {key} failed: {type(exc).__name__}: {exc}")
    return key, "error", float('nan'), TokensUsage(input_tokens=None, output_tokens=None)


def _sum_optional_tokens(first: int | None, second: int | None) -> int | None:
    """None + None stays None (unknown); otherwise sum, treating a lone None as 0."""
    if first is None and second is None:
        return None
    return (first or 0) + (second or 0)



def _debug_conversation_history(llm_oracle):
    print()
    debug("Conversation History:")
    for n_message, message in enumerate(llm_oracle.messages):
        debug(f"Message {n_message}: {message}")
    print()



def get_llm_mapping_prediction(response):
    """Extract the boolean prediction from a structured LLM response."""
    if isinstance(response.parsed, (BinaryOutputFormat, BinaryOutputFormatWithReasoning)):
        return response.parsed.answer
    if isinstance(response.parsed, (YesNoOutputFormat, YesNoOutputFormatWithReasoning)):
        answer_str = response.parsed.answer.strip().lower()
        if answer_str in POSITIVE_TOKENS:
            return True
        elif answer_str in NEGATIVE_TOKENS:
            return False
        else:
            raise ValueError(f"YesNoOutputFormat answer not recognised: '{response.parsed.answer}'")
    raise NotImplementedError()



def calculate_logprobs_confidence(
    log_probs: list,
    prediction: bool | None = None,
) -> float:
    """
    Extract confidence from logprobs for a binary true/false prediction.
    Searches token logprobs for the true/false decision token, handling
    both bare tokens ("true", "false") and tokens with leading whitespace
    (" true", " false") as (possibly) produced by structured JSON output;
    returns the probability assigned to the parsed answer, rather than the
    larger probability of either binary option.
    Returns NaN if logprobs are unavailable or contain no true/false token.
    """
    if not log_probs:
        if VERBOSE:
            debug("(calculate_logprobs_confidence) invalid argument, returning NaN (cast to float).")
        return float('nan')

    for token_info in log_probs:
        token_text = token_info["token"].strip().lower()
        if token_text not in POSITIVE_TOKENS and token_text not in NEGATIVE_TOKENS:
            if VERBOSE and VERY_VERBOSE:
                debug("(calculate_logprobs_confidence) token_text not in POSITIVE_TOKENS or NEGATIVE_TOKENS.")
            continue

        if prediction is True:
            answer_tokens = POSITIVE_TOKENS
        elif prediction is False:
            answer_tokens = NEGATIVE_TOKENS
        else:
            answer_tokens = (
                POSITIVE_TOKENS if token_text in POSITIVE_TOKENS else NEGATIVE_TOKENS
            )

        answer_logprobs: list[float] = []
        if token_text in answer_tokens and isinstance(token_info.get("logprob"), (int, float)):
            answer_logprobs.append(float(token_info["logprob"]))
        for entry in token_info.get("top_logprobs", []):
            entry_token = str(entry.get("token", "")).strip().lower()
            if entry_token in answer_tokens and isinstance(entry.get("logprob"), (int, float)):
                answer_logprobs.append(float(entry["logprob"]))
        if answer_logprobs:
            probability = float(np.exp(max(answer_logprobs)))
            return min(1.0, max(0.0, probability))
        return float("nan")

    if VERBOSE:
        debug("(calculate_logprobs_confidence) No relevant tokens found at any position.")
    return float('nan')



def _check_failure_abort(prediction_status, consecutive_failures, cumulative_failures, failure_tolerance,
                         consecutive_limit=DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE) -> tuple[bool, int, int]:
    if prediction_status != "error":
        return False, 0, cumulative_failures
    # else (an error has occured):
    warn("A consultation failure has been encountered.")
    consecutive_failures += 1
    cumulative_failures += 1
    should_abort: bool = (consecutive_failures >= consecutive_limit or cumulative_failures >= failure_tolerance)
    return should_abort, consecutive_failures, cumulative_failures



def _resolve_failure_tolerance(oracle_cfg: OracleConfig, n_total: int) -> int:
    """failure tolerance / abort policy"""
    failure_tolerance_floor = DEFAULT_FAILURE_TOLERANCE_FLOOR
    if oracle_cfg.failure_tolerance is not None:
        failure_tolerance_floor = oracle_cfg.failure_tolerance
    effective_faulure_tolerance = max(failure_tolerance_floor, int(n_total * 0.05))
    if VERBOSE:
        debug(f"Effective failure tolerance is set to: {effective_faulure_tolerance}.")
    return effective_faulure_tolerance



def _print_abort_message(consecutive_failures: int, cumulative_failures: int, failure_tolerance: int) -> None:
    critical("\nABORTING Oracle consultations prematurely! Error report:")
    if consecutive_failures >= DEFAULT_CONSECUTIVE_FAILURE_TOLERANCE:
        critical(f"  {consecutive_failures} consecutive failures. Check your system setup.")
    critical(f"  {cumulative_failures} cumulative failures (threshold: {failure_tolerance})")
    warning("Pending consultations will be cancelled.")
    warning("Running consultations will run to completion.\n")



def _resolve_api_key(api_key: str) -> str:
    """Resolve an ``ENV:VARNAME`` sentinel to the secret in that environment variable, so the
    real key is NEVER written into config.toml / manifests / results. A plain value (``EMPTY``
    for local vLLM, or a literal key) passes through unchanged."""
    import os
    if isinstance(api_key, str) and api_key.startswith("ENV:"):
        var = api_key[4:].strip()
        val = os.environ.get(var, "")
        if not val:
            raise ValueError(
                f"oracle.api_key requests environment variable {var!r} but it is unset — "
                f"source <root>/.secrets/env before running cloud oracle experiments."
            )
        return val
    return api_key


def _build_oracle_manager(oracle_cfg: OracleConfig, developer_prompt_text: str | None,
                          few_shot_examples: list | None) -> OracleConsultationManager:
    llm_oracle = OracleConsultationManager(
        api_key=_resolve_api_key(oracle_cfg.api_key),
        model_name=oracle_cfg.model_name,
        interaction_style=oracle_cfg.interaction_style,
        base_url=oracle_cfg.base_url,
        temperature=oracle_cfg.temperature,
        top_p=oracle_cfg.top_p,
        reasoning_effort=oracle_cfg.reasoning_effort,
        reasoning_token_budget=oracle_cfg.reasoning_token_budget,
        max_completion_tokens=oracle_cfg.max_completion_tokens,
        enable_thinking=oracle_cfg.enable_thinking,
        supports_chat_template_kwargs=oracle_cfg.supports_chat_template_kwargs,
        response_format=oracle_cfg.response_format,
        request_timeout_seconds=oracle_cfg.request_timeout_seconds,
        connect_timeout_seconds=oracle_cfg.connect_timeout_seconds,
        transient_retries=oracle_cfg.transient_retries,
        seed=oracle_cfg.seed,
        request_logprobs=oracle_cfg.request_logprobs,
        openrouter_provider=oracle_cfg.openrouter_provider,
        openrouter_allow_fallbacks=oracle_cfg.openrouter_allow_fallbacks,
        openrouter_require_parameters=oracle_cfg.openrouter_require_parameters,
    )
    if developer_prompt_text is not None:
        llm_oracle.add_developer_message(developer_prompt_text)
    # Shared few-shot (a list) is baked into the frozen developer prefix. Query-specific
    # RAG few-shot (a dict keyed by M_ask key) is not baked here — it is passed
    # per-consultation, so the shared/frozen state stays example-free and concurrency-safe.
    if isinstance(few_shot_examples, list) and few_shot_examples:
        llm_oracle.add_few_shot_examples(few_shot_examples)
    if VERBOSE:
        _debug_conversation_history(llm_oracle=llm_oracle)
    return llm_oracle



def _build_pair_entity_types(m_ask_init_alignment_df: pd.DataFrame) -> dict[str, str]:
    """
    Build src|tgt -> entityType lookup from the m_ask DataFrame, used to select the class
    (CLS), property (OPROP), or instance (INST) dev/system + user prompts during
    consultation. ie. dict: { "SRC_URI|TGT_URI": "OPROP" || "INST" || "CLS", ... }
    """
    pair_entity_types = {}
    for _, row in m_ask_init_alignment_df.iterrows():
        base_key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        etype = str(row.iloc[4]).strip() if len(row) > 4 else "CLS"
        pair_entity_types[base_key] = etype
        if VERBOSE and VERY_VERBOSE:
            debug(f"Attached Entity Type '{etype}' to Mapping '{base_key}'")

    return pair_entity_types



def _resolve_developer_override(full_key: str, developer_prompt_map: dict | None, pair_entity_types: dict[str, str]) -> str | None:
    if not developer_prompt_map:
        return None
    base_key = PAIRS_SEPARATOR.join(full_key.split(PAIRS_SEPARATOR)[:2])
    etype = pair_entity_types.get(base_key, "CLS")
    return developer_prompt_map.get(etype)



def _run_consultations(llm_oracle: OracleConsultationManager, m_ask_prompts: dict[str, str],
                       pair_entity_types: dict[str, str], developer_prompt_map: dict | None,
                       max_workers: int, failure_tolerance: int, desc: str,
                       per_query_examples: dict | None = None) -> dict | None:
    """
    Dispatches all prompts in parallel; returns None if aborted.
    ``per_query_examples``: optional {m_ask_key -> [(user,assistant), ...]} so each
    consultation carries its own query-specific few-shot examples (no shared conversation
    state).
    """
    results = {}
    consecutive_failures = 0
    cumulative_failures = 0

    items = iter(m_ask_prompts.items())
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = set()
        future_keys: dict = {}  # future -> m_ask key, for the defensive raised-future branch

        def submit_next() -> bool:
            try:
                key, prompt = next(items)
            except StopIteration:
                return False
            future = executor.submit(
                consult_oracle_for_mapping,
                key,
                prompt,
                llm_oracle,
                developer_override=_resolve_developer_override(
                    key, developer_prompt_map, pair_entity_types
                ),
                few_shot_examples=(
                    per_query_examples.get(key) if per_query_examples else None
                ),
            )
            pending.add(future)
            future_keys[future] = key
            return True

        for _ in range(min(max_workers, len(m_ask_prompts))):
            submit_next()

        progress = tqdm(total=len(m_ask_prompts), desc=desc)
        while pending:
            future = next(as_completed(pending))
            pending.remove(future)
            try:
                key, prediction, confidence, usage = future.result()
                results[key] = (prediction, confidence, usage)
                status = prediction  # "error" on a failed consultation; bool otherwise
            except Exception as e:
                # Defensive: consult_oracle_for_mapping catches its own exceptions and
                # returns error tuples, so this only fires on a bug in that error path.
                # Record the failure against its actual key so the mapping does not
                # surface downstream as 'skipped'.
                failed_key = future_keys.get(future)
                warning(f"Consultation for {failed_key} raised unexpectedly: "
                        f"{type(e).__name__}: {e}")
                if failed_key is not None:
                    results[failed_key] = (
                        "error", float("nan"),
                        TokensUsage(input_tokens=None, output_tokens=None),
                    )
                status = "error"

            # Drive the failure-abort policy from the returned status: consultation
            # failures and the BadRequestError branch return ("error", ...) tuples instead
            # of raising, so future.result() almost never raises. Passing `status` makes
            # both error tuples and raised exceptions count toward the abort policy.
            should_abort, consecutive_failures, cumulative_failures = _check_failure_abort(
                status, consecutive_failures, cumulative_failures, failure_tolerance,
            )
            if should_abort:
                _print_abort_message(consecutive_failures, cumulative_failures, failure_tolerance)
                for queued in pending:
                    queued.cancel()
                progress.close()
                return None
            progress.update(1)
            submit_next()
        progress.close()

    return results



def consult_oracle_for_mapping(key, prompt, llm_oracle, developer_override=None, few_shot_examples=None):
    """obtain prediction, confidence and token usage for a given mapping (id'd via key).
    ``few_shot_examples`` = this mapping's own query-specific RAG examples."""
    max_retries = getattr(llm_oracle, "transient_retries", 2)
    seed = getattr(llm_oracle, "seed", None)
    for attempt in range(max_retries + 1):
        try:
            response = llm_oracle.consult_oracle(
                prompt, developer_override, few_shot_examples
            )
            prediction = get_llm_mapping_prediction(response)
            confidence = calculate_logprobs_confidence(response.logprobs, prediction)
            return key, prediction, confidence, response.usage
        except BadRequestError as exc:
            try:
                body = getattr(exc, "body", None) or {}
                error_msg = body.get("error", {}).get("message") or str(exc)
            except (AttributeError, TypeError):
                error_msg = str(exc)
            warning(f'BadRequestError: {error_msg} (for mapping: {key})')
            return _consultation_failure_record(key, exc)
        except Exception as exc:
            if not _is_transient_oracle_error(exc):
                return _consultation_failure_record(key, exc)
            if attempt == max_retries:
                return _consultation_failure_record(key, exc)
            retry_number = attempt + 1
            delay = _retry_delay_seconds(key, retry_number, seed)
            if VERBOSE:
                debug(
                    f"Transient retry {retry_number}/{max_retries} for mapping {key}: "
                    f"{type(exc).__name__}: {exc}; waiting {delay:.3f}s"
                )
            time.sleep(delay)

    raise AssertionError("unreachable retry state")



def consult_oracle_for_mappings_to_ask(
    m_ask_prompts: dict[str, str],
    m_ask_init_alignment_df: pd.DataFrame,
    oracle_cfg: OracleConfig,
    developer_prompt_text: str | None = None,
    developer_prompt_map: dict | None = None,
    few_shot_examples: list | None = None,
    **kwargs
) -> pd.DataFrame | None:

    _require_forward_prompt_coverage(m_ask_prompts, m_ask_init_alignment_df)
    failure_tolerance = _resolve_failure_tolerance(oracle_cfg, len(m_ask_prompts))
    llm_oracle = _build_oracle_manager(oracle_cfg, developer_prompt_text, few_shot_examples)
    llm_oracle.freeze_messages() # frozen prefix = developer msg (+ legacy shared few-shot list only)
    pair_entity_types = _build_pair_entity_types(m_ask_init_alignment_df) if developer_prompt_map else {}
    # A dict of few-shot examples is query-specific -> passed per-consultation.
    per_query_examples = few_shot_examples if isinstance(few_shot_examples, dict) else None

    results = _run_consultations(
        llm_oracle=llm_oracle,
        m_ask_prompts=m_ask_prompts,
        pair_entity_types=pair_entity_types,
        developer_prompt_map=developer_prompt_map,
        max_workers=oracle_cfg.max_workers,
        failure_tolerance=failure_tolerance,
        desc="Oracle consultations",
        per_query_examples=per_query_examples,
    )

    if results is None:
        return None

    ordered_pred, ordered_conf, ordered_in, ordered_out = [], [], [], []
    ordered_reasoning, ordered_finish, ordered_native_finish = [], [], []
    ordered_provider, ordered_response_model = [], []
    ordered_raw_response = []
    skipped_count = 0

    for _, row in m_ask_init_alignment_df.iterrows():
        key = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        if key in results:
            pred, conf, usage = results[key]
            ordered_pred.append(pred)
            ordered_conf.append(conf)
            ordered_in.append(usage.input_tokens)
            ordered_out.append(usage.output_tokens)
            ordered_reasoning.append(usage.reasoning_tokens)
            ordered_finish.append(usage.finish_reason)
            ordered_native_finish.append(usage.native_finish_reason)
            ordered_provider.append(usage.provider)
            ordered_response_model.append(usage.response_model)
            ordered_raw_response.append(usage.raw_response)
        else:
            skipped_count += 1
            ordered_pred.append("skipped")
            ordered_conf.append(float("nan"))
            ordered_in.append(None)
            ordered_out.append(None)
            ordered_reasoning.append(None)
            ordered_finish.append(None)
            ordered_native_finish.append(None)
            ordered_provider.append(None)
            ordered_response_model.append(None)
            ordered_raw_response.append(None)

    if skipped_count > 0:
        warn(f"{skipped_count} mappings 'skipped' (no prompt built due to unresolvable class URIs).")

    m_ask_df_ext = m_ask_init_alignment_df.copy()

    m_ask_df_ext['Oracle_prediction'] = ordered_pred # bool | "error" | "skipped"
    m_ask_df_ext['Oracle_confidence'] = pd.Series(ordered_conf, dtype="float64")
    m_ask_df_ext['Oracle_input_tokens'] = pd.Series(ordered_in, dtype="Int64")      # nullable
    m_ask_df_ext['Oracle_output_tokens'] = pd.Series(ordered_out, dtype="Int64")    # nullable
    m_ask_df_ext['Oracle_reasoning_tokens'] = pd.Series(ordered_reasoning, dtype="Int64")
    m_ask_df_ext['Oracle_finish_reason'] = pd.Series(ordered_finish, dtype="string")
    m_ask_df_ext['Oracle_native_finish_reason'] = pd.Series(ordered_native_finish, dtype="string")
    m_ask_df_ext['Oracle_provider'] = pd.Series(ordered_provider, dtype="string")
    m_ask_df_ext['Oracle_response_model'] = pd.Series(ordered_response_model, dtype="string")
    m_ask_df_ext['Oracle_raw_response'] = pd.Series(ordered_raw_response, dtype="string")
    m_ask_df_ext.attrs["oracle_capabilities"] = {
        "logprobs_requested": llm_oracle.logprobs_requested,
        "logprobs_effective": llm_oracle.logprobs,
        "downgrades": list(llm_oracle.capability_downgrades),
    }

    return m_ask_df_ext



def _forward_only_keys(m_ask_init_alignment_df: pd.DataFrame) -> set[str]:
    """M_ask keys that LogMap types as property/instance candidates: consulted once, never AND-ed."""
    return {
        key for key, etype in _build_pair_entity_types(m_ask_init_alignment_df).items()
        if etype in ("OPROP", "DPROP", "INST")
    }


def _aggregate_bidirectional(results: dict, m_ask_init_alignment_df: pd.DataFrame,
                             forward_only_keys: set[str] | None = None) -> dict:
    """Fold per-direction oracle results into one verdict per M_ask row.

    Class candidates: accepted iff the forward AND the reverse subsumption hold; confidence is
    the minimum of the two; tokens are summed. Forward-only (property/instance) candidates carry
    their single verdict through unchanged, with the reverse columns marked "n/a". A row without
    the results it needs is "skipped" (a reject downstream). Pure, so it is unit-testable.
    """
    forward_only_keys = forward_only_keys or set()
    out = {k: [] for k in ("pred", "conf", "in", "out", "fwd", "rev", "fwd_conf", "rev_conf")}
    skipped = 0

    def _f(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float("nan")

    for _, row in m_ask_init_alignment_df.iterrows():
        base = str(row.iloc[0]) + PAIRS_SEPARATOR + str(row.iloc[1])
        rev_key = base + PAIRS_SEPARATOR + "REVERSE"
        if base in forward_only_keys and base in results:
            pred, conf, usage = results[base]
            out["pred"].append(pred); out["conf"].append(_f(conf))
            out["in"].append(usage.input_tokens); out["out"].append(usage.output_tokens)
            out["fwd"].append(pred); out["rev"].append("n/a")
            out["fwd_conf"].append(_f(conf)); out["rev_conf"].append(float("nan"))
            continue
        if base not in results or rev_key not in results:
            skipped += 1
            out["pred"].append("skipped"); out["conf"].append(float("nan"))
            out["in"].append(None); out["out"].append(None)
            out["fwd"].append("skipped"); out["rev"].append("skipped")
            out["fwd_conf"].append(float("nan")); out["rev_conf"].append(float("nan"))
            if VERBOSE and VERY_VERBOSE:
                debug(f"(consult_oracle_bidirectional) skipping mapping in M_ask: {base} <> {rev_key}.")
            continue
        fwd_pred, fwd_conf, fwd_usage = results[base]
        rev_pred, rev_conf, rev_usage = results[rev_key]
        out["fwd"].append(fwd_pred); out["rev"].append(rev_pred)
        out["fwd_conf"].append(_f(fwd_conf)); out["rev_conf"].append(_f(rev_conf))
        # true iff both hold (equivalence via mutual subsumption)
        if fwd_pred == "error" or rev_pred == "error":
            agg_pred = "error"
            warn("Encountered an 'error' when computing logical AND in bidirectional mode.")
        else:
            agg_pred = bool(fwd_pred) and bool(rev_pred)
        try:  # the confidence is the minimum of the two directional confidences
            agg_conf = min(float(fwd_conf), float(rev_conf))
        except (ValueError, TypeError):
            warn("Confidence value can not be resolved for an M_ask mapping in bidirectional mode.")
            agg_conf = float("nan")
        out["pred"].append(agg_pred); out["conf"].append(agg_conf)
        # Sum only when at least one direction reported usage; keep None (nullable Int64)
        # when neither did, matching unidirectional mode.
        out["in"].append(_sum_optional_tokens(fwd_usage.input_tokens, rev_usage.input_tokens))
        out["out"].append(_sum_optional_tokens(fwd_usage.output_tokens, rev_usage.output_tokens))
    out["skipped"] = skipped
    return out


def consult_oracle_bidirectional(
    m_ask_prompts: dict[str, str],
    m_ask_init_alignment_df: pd.DataFrame,
    oracle_cfg: OracleConfig,
    developer_prompt_text: str | None = None,
    developer_prompt_map: dict | None = None,
    few_shot_examples: list | None = None,
    **kwargs
) -> pd.DataFrame | None:
    """
    Consult an LLM Oracle with bidirectional subsumption prompts.
    --------------------------------------------------------------
    Sends forward and reverse subsumption queries, then aggregates into
    per-candidate equivalence predictions. A candidate is predicted as
    equivalent (ie. true) if and only if both forward and reverse subsumption
    hold; otherwise false.
    """
    _require_forward_prompt_coverage(m_ask_prompts, m_ask_init_alignment_df)
    forward_only_keys = _forward_only_keys(m_ask_init_alignment_df)
    _require_reverse_prompt_coverage(m_ask_prompts, forward_only_keys)
    failure_tolerance = _resolve_failure_tolerance(oracle_cfg, len(m_ask_prompts))
    llm_oracle = _build_oracle_manager(oracle_cfg, developer_prompt_text, few_shot_examples)
    llm_oracle.freeze_messages() # frozen prefix = developer msg (+ legacy shared few-shot list only)
    pair_entity_types = _build_pair_entity_types(m_ask_init_alignment_df) if developer_prompt_map else {}
    per_query_examples = few_shot_examples if isinstance(few_shot_examples, dict) else None

    results = _run_consultations(
        llm_oracle=llm_oracle,
        m_ask_prompts=m_ask_prompts,
        pair_entity_types=pair_entity_types,
        developer_prompt_map=developer_prompt_map,
        max_workers=oracle_cfg.max_workers,
        failure_tolerance=failure_tolerance,
        desc="Oracle consultations (bidirectional)",
        per_query_examples=per_query_examples,
    )

    if results is None:
        return None

    agg = _aggregate_bidirectional(results, m_ask_init_alignment_df, forward_only_keys)
    ordered_pred, ordered_conf, ordered_in, ordered_out = agg["pred"], agg["conf"], agg["in"], agg["out"]
    ordered_fwd, ordered_rev, ordered_fwd_conf, ordered_rev_conf = agg["fwd"], agg["rev"], agg["fwd_conf"], agg["rev_conf"]
    if agg["skipped"] > 0:
        warning(f"{agg['skipped']} mappings marked as 'skipped'.")

    m_ask_df_ext = m_ask_init_alignment_df.copy()
    m_ask_df_ext['Oracle_prediction'] = ordered_pred # bool | "error" | "skipped"
    m_ask_df_ext['Oracle_confidence'] = pd.Series(ordered_conf, dtype="float64")
    m_ask_df_ext['Oracle_input_tokens'] = pd.Series(ordered_in, dtype="Int64")      # nullable
    m_ask_df_ext['Oracle_output_tokens'] = pd.Series(ordered_out, dtype="Int64")    # nullable
    # The two directional decisions the AND was taken over; written only on bidirectional runs.
    m_ask_df_ext['Oracle_fwd_prediction'] = ordered_fwd
    m_ask_df_ext['Oracle_rev_prediction'] = ordered_rev
    m_ask_df_ext['Oracle_fwd_confidence'] = pd.Series(ordered_fwd_conf, dtype="float64")
    m_ask_df_ext['Oracle_rev_confidence'] = pd.Series(ordered_rev_conf, dtype="float64")
    m_ask_df_ext.attrs["oracle_capabilities"] = {
        "logprobs_requested": llm_oracle.logprobs_requested,
        "logprobs_effective": llm_oracle.logprobs,
        "downgrades": list(llm_oracle.capability_downgrades),
    }

    return m_ask_df_ext
