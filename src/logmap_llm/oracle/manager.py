'''
logmap_llm.oracle.manager
Contains OracleConsultationManager/s which manages LLM interactions via the OpenAI SDK.
Supporting both OpenRouter and local endpoints (vLLM, SGLang).
'''
from __future__ import annotations
from openai import OpenAI, Timeout
from pydantic import BaseModel, ValidationError
from logmap_llm.constants import (
    BinaryOutputFormat,
    BinaryOutputFormatWithReasoning,
    YesNoOutputFormat,
    YesNoOutputFormatWithReasoning,
    LLMCallOutput,
    TokensUsage,
    POSITIVE_TOKENS,
    NEGATIVE_TOKENS,
    InteractionStyle,
    RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.misc import resolve_response_format_to_str
from logmap_llm.utils.logging import (
    warning,
    debug,
)
import threading
import json
import re


###
# Plain-answer parsing
# --------------------
# Helpers for OracleConsultationManager._parse_plain_text_answer, kept at module level
# so they are unit-testable in isolation.
###

#: Trimmed from both ends before an exact-token match: whitespace, quoting, markdown
#: emphasis and terminal punctuation.
_ANSWER_STRIP_CHARS = " \t\r\n\"'`*_.!,;:()[]{}"

#: Words that invert the polarity of a following answer token within the same clause.
#: Apostrophes are stripped before matching, so "isn't" arrives as "isnt". Note "no" is both
#: a negator and a negative token: "no, not true" resolves to False either way.
_NEGATORS = frozenset({
    "not", "nt", "never", "cannot", "cant", "dont", "doesnt", "didnt",
    "isnt", "arent", "wasnt", "werent", "no", "none", "nor", "neither",
    "without", "nope", "false",
})

#: How many preceding word tokens a negator may govern. Three covers "is not true",
#: "cannot really be false", "no, they are not" without reaching across a clause.
_NEGATION_WINDOW = 3

#: Words a negator may reach across to govern an answer token. Anything else blocks it.
#: The rule is grammatical rather than positional: a negator reaches an answer token only
#: through copulas, auxiliaries and intensifiers — the words that chain "cannot" to "false"
#: in "cannot be false". A content word in between means the negator is negating that word,
#: and the answer token stands unflipped ("not identical - False" reads False).
_NEGATION_CARRIERS = frozenset({
    "be", "been", "being", "is", "are", "was", "were", "am",
    "the", "a", "an", "it", "this", "that", "these", "those", "they",
    "really", "necessarily", "actually", "certainly", "definitely", "strictly",
    "exactly", "entirely", "quite", "simply", "just", "always", "then", "so",
    "would", "could", "should", "can", "may", "might", "must", "do", "does", "did",
})

#: Clause boundaries. A comma splits too, because "not equivalent, so false" carries two
#: independent verdicts and the second is the terminal one. Dashes, arrows and pipes are
#: verdict separators in practice ("not identical -> False") and split as well; "/" does
#: not, because "true/false" is a genuine ambiguity that must not resolve to a verdict.
_CLAUSE_SPLIT = re.compile(r"[.!?;:\n|–—→]+|,|--+|->|=>|(?<=\s)-(?=\s)")
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def _token_verdict(token: str) -> bool | None:
    """True/False for a supported answer token, else None."""
    if token in POSITIVE_TOKENS:
        return True
    if token in NEGATIVE_TOKENS:
        return False
    return None


def _json_verdict(text: str) -> bool | None:
    """The verdict of a supported JSON answer document, else None."""
    try:
        loaded = json.loads(text)
    except (ValueError, TypeError):
        return None
    if isinstance(loaded, bool):
        return loaded
    if isinstance(loaded, dict) and set(loaded) == {"answer"}:
        value = loaded["answer"]
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return _token_verdict(value.strip(_ANSWER_STRIP_CHARS).lower())
    return None


def _clause_verdicts(text: str) -> list[list[bool]]:
    """Effective verdicts per clause, with negation scope resolved.

    Returns one list per clause that contains at least one answer token. A token is flipped
    when a negator governs it: within `_NEGATION_WINDOW` words, in the same clause, reached
    only across `_NEGATION_CARRIERS`.
    """
    clauses: list[list[bool]] = []
    for clause in _CLAUSE_SPLIT.split(text.lower()):
        if not clause or not clause.strip():
            continue
        words = [word.replace("'", "") for word in _WORD.findall(clause)]
        found: list[bool] = []
        for index, word in enumerate(words):
            polarity = _token_verdict(word)
            if polarity is None:
                continue
            found.append((not polarity) if _is_negated(words, index) else polarity)
        if found:
            clauses.append(found)
    return clauses


def _is_negated(words: list[str], index: int) -> bool:
    """Does a negator govern the answer token at `index`?

    Walk back at most `_NEGATION_WINDOW` words. A negator flips; a carrier is transparent;
    anything else is the thing the negator was actually about, and stops the walk.
    """
    for position in range(index - 1, max(-1, index - 1 - _NEGATION_WINDOW), -1):
        candidate = words[position]
        if candidate in _NEGATORS:
            return True
        if candidate not in _NEGATION_CARRIERS:
            return False
    return False


_UNSUPPORTED_PARAMETER_MARKERS = (
    "not supported",
    "unsupported",
    "does not support",
    "unknown parameter",
    "unrecognized",
    "unexpected keyword",
    "not allowed",
)


def _is_unsupported_logprobs_error(error: BaseException) -> bool:
    """Recognise an explicit provider capability refusal, not any logprob failure."""
    message = str(error).lower()
    return "logprob" in message and any(
        marker in message for marker in _UNSUPPORTED_PARAMETER_MARKERS
    )


class OracleTransientResponseError(RuntimeError):
    """
    A 200 response without any choice. OpenRouter (and some upstream providers) answer a
    failed upstream call with a well-formed completion object whose ``choices`` is ``None``
    and an ``error`` member (e.g. ``{"code": 502, "message": "Provider returned error"}``)
    instead of an HTTP error. Indexing the first choice raised a ``TypeError`` that the
    consultation loop did not classify as transient, so one burst of five such responses
    tripped the circuit breaker (observed on OpenRouter/Google AI Studio, Sept 2026). The
    consultation loop treats this error like a connection error: retried under
    ``transient_retries``, then recorded as an ``error`` verdict.
    """

    def __init__(self, error: object = None, provider: object = None):
        self.error = error
        self.provider = provider
        detail = f"{error!r}" if error is not None else "no error object"
        super().__init__(
            f"provider response carried no choices ({detail}"
            + (f", provider={provider!r}" if provider else "")
            + ")"
        )


def _first_choice(response):
    """The first choice of a completion, guarded against the choices-less error shape above."""
    choices = getattr(response, "choices", None)
    if not choices:
        error = getattr(response, "error", None)
        provider = getattr(response, "provider", None)
        warning(
            "(consult_oracle) provider returned a response without choices; treating it as "
            f"transient. error={error!r} provider={provider!r}"
        )
        raise OracleTransientResponseError(error, provider)
    return choices[0]


class OracleConsultationManager:
    """
    Manages consultations with an LLM Oracle via OpenAI-compatible API.
    Supports OpenRouter, vLLM, SGLang, and any OpenAI-compatible endpoint.
    """
    def __init__(self, api_key: str, model_name: str, interaction_style: str, base_url: str, temperature: float, top_p: float,
                 reasoning_effort: str | None, max_completion_tokens: int, enable_thinking: bool,
                 reasoning_token_budget: int | None = None,
                 supports_chat_template_kwargs: bool | None = None,
                 response_format: BinaryOutputFormat | BinaryOutputFormatWithReasoning | YesNoOutputFormat | YesNoOutputFormatWithReasoning | None = None,
                 request_timeout_seconds: float = 120.0, connect_timeout_seconds: float = 15.0,
                 transient_retries: int = 2, seed: int | None = None,
                 request_logprobs: bool = True, openrouter_provider: str | None = None,
                 openrouter_allow_fallbacks: bool = False,
                 openrouter_require_parameters: bool = True):

        if VERBOSE:
            debug("Initialising OracleConsultationManager ... with args:")
            debug(f"model={model_name}")
            debug(f"base_url={base_url}")
            debug(f"temp={str(temperature)}")
            debug(f"reasoning={str(enable_thinking)}")
            debug(f"response_format={resolve_response_format_to_str(response_format)}")

        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url

        self.interaction_style = self._resolve_interaction_style(
            interaction_style, self.base_url,
        )

        # _dispatch_consult has no structured-mode branch for 'local', so refuse the
        # combination up front as a clear config error. (Plain response mode still works
        # with 'local': the unstructured branch dispatches before the style switch.)
        if (
            self.interaction_style == InteractionStyle.LOCAL_GENERIC
            and response_format is not RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE
        ):
            raise ValueError(
                "interaction_style='local' supports only the plain (unstructured) response "
                "mode; for structured output select 'vllm', 'sglang', 'openrouter' or "
                "'openai_chat_completions_parse_structured_output' explicitly."
            )

        if VERBOSE:
            debug(f"(resolved argument) interaction_style={InteractionStyle(self.interaction_style)}")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            # Consultation owns one explicit retry policy; avoid multiplying it
            # by a second opaque SDK retry loop.
            max_retries=0,
            timeout=Timeout(request_timeout_seconds, connect=connect_timeout_seconds),
        )

        self.messages = []
        self._frozen = False
        self._frozen_messages = ()

        self.response_format = response_format

        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.reasoning_token_budget = reasoning_token_budget
        self.max_completion_tokens = max_completion_tokens
        self.enable_thinking = enable_thinking
        self.seed = seed
        self.transient_retries = transient_retries
        self.openrouter_provider = openrouter_provider
        self.openrouter_allow_fallbacks = openrouter_allow_fallbacks
        self.openrouter_require_parameters = openrouter_require_parameters

        self.logprobs_requested = request_logprobs
        self.logprobs = request_logprobs
        self.top_logprobs = 3
        # One manager is shared across max_workers consultation threads; the logprobs
        # latch is written from whichever thread is refused first.
        self._logprobs_lock = threading.Lock()
        self.capability_downgrades: list[dict[str, str]] = []

        # chat_template_kwargs compatibility: Mistral models served via vLLM use the
        # proprietary "tekken" tokenizer, which rejects chat_template /
        # chat_template_kwargs. Resolution order: explicit config override, else
        # auto-detection from model_name, else default True.
        explicit = supports_chat_template_kwargs

        if explicit is not None:
            self.supports_chat_template_kwargs = explicit
        else:
            self.supports_chat_template_kwargs = (
                not self._is_mistral_family(self.model_name)
            )

        # TODO: maintain an official 'supported models' list


    @staticmethod
    def _is_mistral_family(model_name: str) -> bool:
        """Check whether a model name belongs to the Mistral family."""
        return "mistral" in model_name.lower()


    @staticmethod
    def _resolve_interaction_style(requested: str, base_url: str | None) -> InteractionStyle:
        """
        Translates the configured interaction_style into a concrete enum value (see constants.py).
        """
        if requested.lower() == InteractionStyle.AUTOMATIC:
            if 'openrouter.ai' in (base_url or ''):
                return InteractionStyle.OPEN_ROUTER
            # else (not openrouter):
            return InteractionStyle.LOCAL_VLLM # compatible \w vLLM & SGLang
        # else (not auto):
        try:
            return InteractionStyle(requested)
        except ValueError:
            raise ValueError(f"Unknown interaction_style '{requested}'.")


    def _resolve_system_role(self) -> str:
        """
        Return the appropriate role name for system-level instructions.
        OpenAI/OpenRouter accept 'developer'; vLLM expects 'system'.
        """
        # add the neccesary configuration options here when required (hence: `in`)
        if self.interaction_style in (InteractionStyle.LOCAL_VLLM, InteractionStyle.LOCAL_SG_LANG):
            return 'system'
        return 'developer'

    # -------------------
    # Message management:
    # -------------------

    def freeze_messages(self) -> None:
        """
        Freeze messages — no further modifications allowed.

        Call before entering multithreaded consultation to ensure the
        shared message list is not mutated during concurrent reads.
        """
        self._frozen = True
        self._frozen_messages = tuple(self.messages)


    def add_developer_message(self, message: str) -> None:
        """Add or replace the developer (system) message."""
        if self._frozen:
            raise RuntimeError("Cannot modify messages after freeze_messages().")

        role = self._resolve_system_role()
        system_roles = {"developer", "system"}

        if len(self.messages) == 0 or self.messages[0]["role"] not in system_roles:
            self.messages.insert(0, self.build_api_message(role, message))
        else:
            self.messages[0] = self.build_api_message(role, message)


    def add_message(self, role: str, message: str) -> None:
        """Add a message to the conversation history."""
        if self._frozen:
            raise RuntimeError("Cannot modify messages after freeze_messages().")
        self.messages.append(self.build_api_message(role, message))


    def add_few_shot_examples(self, examples: list) -> None:
        """
        Add few-shot example pairs as user/assistant message turns.
        Call after add_developer_message() and before freeze_messages().
        Each example is a (user_prompt, assistant_response) tuple.
        """
        for user_prompt, assistant_response in examples:
            self.add_message("user", user_prompt)
            self.add_message("assistant", assistant_response)


    def set_response_format(self, response_format: BaseModel | dict) -> None:
        self.response_format = response_format


    def build_api_message(self, role: str, message: str) -> dict:
        return {"role": role, "content": message}


    def set_interaction_style(self, interaction_style) -> None:
        self.interaction_style = interaction_style


    def clear_messages(self) -> None:
        """Clear all messages and unfreeze."""
        self._frozen = False
        self._frozen_messages = ()
        self.messages = []

    # --------------------
    # Oracle consultation
    # --------------------

    def consult_oracle(self, message, developer_override=None, few_shot_examples=None):
        """
        Consult the LLM Oracle, dispatching to the appropriate method.

        ``few_shot_examples``: an optional per-query list of (user_prompt, assistant_answer)
        pairs, assembled into this request locally and never stored on the manager — so
        concurrent consultations each carry their own query-specific examples with no shared
        conversation state. When None, few-shot examples (if any) are already baked into the
        frozen developer prefix.
        """
        try:
            return self._dispatch_consult(message, developer_override, few_shot_examples)
        except Exception as e:
            # A provider that refuses logprobs outright (e.g. HTTP 400 "logprobs are not
            # supported with reasoning models") would otherwise fail every call: drop
            # logprobs once, latch it for the rest of the run, and retry. The condition
            # deliberately does not test `self.logprobs`: consultations run on max_workers
            # threads against one shared manager, so threads already in flight when the
            # first refusal latches self.logprobs=False must still retry here rather than
            # re-raise. Retry is bounded by construction: at most once per call, and the
            # retry cannot raise a logprobs error because logprobs are no longer sent.
            if getattr(self, "logprobs_requested", True) and _is_unsupported_logprobs_error(e):
                with self._logprobs_lock:
                    if self.logprobs:
                        warning("(consult_oracle) provider rejected logprobs "
                                f"({type(e).__name__}); disabling logprobs for the remainder of this "
                                "run. Oracle_confidence becomes NaN -> DEFAULT_CONFIDENCE_FALLBACK. "
                                "Verified inert: min_conf_pro_map does not gate oracle-accepted "
                                "mappings (99.06% vs 99.20% retention across the 0.80 boundary).")
                        if not hasattr(self, "capability_downgrades"):
                            self.capability_downgrades = []
                        self.capability_downgrades.append({
                            "capability": "logprobs",
                            "requested": "enabled",
                            "effective": "disabled",
                            "reason": "provider explicitly rejected the parameter",
                        })
                    self.logprobs = False
                    self.top_logprobs = None
                return self._dispatch_consult(message, developer_override, few_shot_examples)
            raise


    def _dispatch_consult(self, message, developer_override=None, few_shot_examples=None):
        if self.response_format is RESPONSE_FORMAT_FOR_UNSTRUCTURED_RESPONSE:
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via plain.")
            return self._consult_via_plain(message, developer_override, few_shot_examples)

        elif self.interaction_style == InteractionStyle.OPEN_ROUTER:
            # OpenRouter providers do not all honour response_format strictly.
            # Keep schema guidance on the wire, but parse the visible response
            # locally so an unambiguous Yes/No or True/False is not discarded.
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting OpenRouter via create + flexible binary parser.")
            return self._consult_via_create(message, developer_override, few_shot_examples)

        elif self.interaction_style == InteractionStyle.OPEN_AI_CHAT_COMPLETIONS_PARSE:
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via parse.")
            return self._consult_via_parse(message, developer_override, few_shot_examples)

        elif self.interaction_style in (InteractionStyle.LOCAL_VLLM, InteractionStyle.LOCAL_SG_LANG):
            if VERBOSE and VERY_VERBOSE:
                debug("(consult_oracle) Consulting via create.")
            return self._consult_via_create(message, developer_override, few_shot_examples)

        else:
            raise ValueError(f"Interaction style not recognised: {self.interaction_style}")


    def _build_base_kwargs(self, prompt, developer_override=None, few_shot_examples=None):

        msgs = list(self._frozen_messages if self._frozen else self.messages)

        # Per-call developer prompt substitution (entity-type-aware)
        if developer_override is not None and msgs:
            system_roles = {"developer", "system"}
            if msgs[0]["role"] in system_roles:
                msgs[0] = self.build_api_message(msgs[0]["role"], developer_override)

        # Per-query few-shot pairs are expanded into user/assistant turns inserted after
        # the developer message and before the target query — assembled locally so
        # concurrent requests never mutate or read shared conversation state.
        fewshot_msgs = []
        for pair in (few_shot_examples or []):
            fewshot_msgs.append(self.build_api_message("user", pair[0]))
            fewshot_msgs.append(self.build_api_message("assistant", pair[1]))

        messages = [*msgs, *fewshot_msgs, self.build_api_message("user", prompt)]

        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed

        # self.logprobs is cleared by the runtime fallback in consult_oracle() the first
        # time a provider refuses the parameter. Dropping logprobs is safe: the derived
        # Oracle_confidence is handed to the Java refiner but never read there, so it does
        # not gate which oracle-accepted mappings survive refinement. (min_conf_pro_map
        # thresholds LogMap's own ISUB-derived confidence, not the oracle's.)
        if self.logprobs:
            kwargs["logprobs"] = self.logprobs
            kwargs["top_logprobs"] = self.top_logprobs

        extra_body = {}
        # vLLM/SGLang expose tokenizer thinking controls through chat_template_kwargs.
        if (
            self.interaction_style in (
                InteractionStyle.LOCAL_VLLM,
                InteractionStyle.LOCAL_SG_LANG,
            )
            and self.enable_thinking is not None
            and self.supports_chat_template_kwargs
        ):
            extra_body["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking
            }
        if self.interaction_style == InteractionStyle.OPEN_ROUTER:
            # Use OpenRouter's unified nested control. Effort and an exact token
            # allocation are alternative controls and are validated as mutually exclusive.
            if self.reasoning_token_budget is not None:
                extra_body["reasoning"] = {
                    "max_tokens": self.reasoning_token_budget
                }
            elif self.reasoning_effort == "none":
                extra_body["reasoning"] = {"enabled": False}
            elif self.reasoning_effort:
                extra_body["reasoning"] = {"effort": self.reasoning_effort}
            provider = {
                "allow_fallbacks": self.openrouter_allow_fallbacks,
                "require_parameters": self.openrouter_require_parameters,
            }
            if self.openrouter_provider is not None:
                provider["order"] = [self.openrouter_provider]
            extra_body["provider"] = provider
        if extra_body:
            kwargs["extra_body"] = extra_body

        return kwargs


    def _extract_response(self, response, parsed_output):
        """Extract common fields from an API response into LLMCallOutput."""
        output_message = _first_choice(response).message.content
        raw_usage = getattr(response, "usage", None)
        completion_details = getattr(raw_usage, "completion_tokens_details", None)
        if isinstance(completion_details, dict):
            reasoning_tokens = completion_details.get("reasoning_tokens")
        else:
            reasoning_tokens = getattr(completion_details, "reasoning_tokens", None)
        choice = _first_choice(response)
        usage = TokensUsage(
            input_tokens=getattr(raw_usage, "prompt_tokens", None),
            output_tokens=getattr(raw_usage, "completion_tokens", None),
            reasoning_tokens=reasoning_tokens,
            finish_reason=getattr(choice, "finish_reason", None),
            native_finish_reason=(
                getattr(choice, "native_finish_reason", None)
                or getattr(response, "native_finish_reason", None)
            ),
            provider=getattr(response, "provider", None),
            response_model=getattr(response, "model", None),
            raw_response=output_message,
        )
        try:
            logprobs = choice.logprobs.model_dump()["content"]
        except (AttributeError, KeyError, TypeError):
            logprobs = []

        return LLMCallOutput(
            message=output_message,
            usage=usage,
            logprobs=logprobs,
            parsed=parsed_output,
        )


    @staticmethod
    def _parse_plain_text_answer(raw_content: str) -> BinaryOutputFormat:
        """
        Parse a plain-text LLM response into a BinaryOutputFormat.

        Used both as the primary parser in plain mode and as the fallback inside
        _consult_via_create when guided JSON decoding fails.

        A staged cascade, first match wins:

          1. the whole response, stripped of whitespace, quotes and trailing punctuation,
             is a supported token  ->  that verdict;
          2. it is JSON of the supported shape  ->  that verdict;
          3. clause scan: for each polarity token compute an effective verdict, flipped by
             a negator within the preceding few words of the same clause. All effective
             verdicts agreeing -> that verdict; otherwise the terminal verdict (the last
             clause carrying one) if that clause is internally consistent; otherwise
             ambiguous (raises).

        This resolves negation scope, not meaning ("true is wrong" still reads as True).
        The compliance guarantee is the strict raw-format audit each campaign publishes,
        not this parser.
        """
        if raw_content is None:
            raise ValueError("LLM returned empty response")

        stripped = str(raw_content).strip()

        # (1) exact supported token, possibly quoted, emphasised or punctuated
        verdict = _token_verdict(stripped.strip(_ANSWER_STRIP_CHARS).lower())
        if verdict is not None:
            return BinaryOutputFormat(answer=verdict)

        # (2) JSON of the supported shape (the guided-decoding fallback path)
        verdict = _json_verdict(stripped)
        if verdict is not None:
            return BinaryOutputFormat(answer=verdict)

        # (3) clause scan with negation scope
        clause_verdicts = _clause_verdicts(stripped)
        if not clause_verdicts:
            raise ValueError(f"Could not parse LLM response: {raw_content[:200]!r}")

        flat = [value for clause in clause_verdicts for value in clause]
        if all(value == flat[0] for value in flat):
            return BinaryOutputFormat(answer=flat[0])

        terminal = clause_verdicts[-1]
        if all(value == terminal[0] for value in terminal):
            return BinaryOutputFormat(answer=terminal[0])

        raise ValueError(
            f"Ambiguous LLM response (conflicting verdicts): {raw_content[:200]!r}"
        )


    def _consult_via_plain(self, prompt, developer_override=None, few_shot_examples=None):
        """
        Consult via create() without a response_format constraint; free-form text is
        parsed locally against POSITIVE_TOKENS / NEGATIVE_TOKENS. Works identically
        against any OpenAI-compatible endpoint (OpenRouter, vLLM, SGLang) because no
        provider-specific schema mechanism is involved.
        """
        kwargs = self._build_base_kwargs(prompt, developer_override, few_shot_examples)
        kwargs["max_tokens"] = self.max_completion_tokens
        # deliberately no response_format — that is the point of plain mode
        response = self.client.chat.completions.create(**kwargs)
        raw_content = _first_choice(response).message.content
        parsed_output = self._parse_plain_text_answer(raw_content)

        return self._extract_response(response, parsed_output)


    def _consult_via_parse(self, prompt, developer_override=None, few_shot_examples=None):
        """
        Consult via OpenAI's parse() method with structured outputs
        """
        kwargs = self._build_base_kwargs(prompt, developer_override, few_shot_examples)
        # OpenRouter's provider capability registry exposes this control as
        # ``max_tokens``. Sending the OpenAI-specific alias
        # ``max_completion_tokens`` with provider.require_parameters=true can
        # therefore make an otherwise compatible pinned route ineligible.
        token_key = (
            "max_tokens"
            if self.interaction_style == InteractionStyle.OPEN_ROUTER
            else "max_completion_tokens"
        )
        kwargs[token_key] = self.max_completion_tokens
        kwargs["response_format"] = self.response_format

        response = self.client.chat.completions.parse(**kwargs)
        parsed_output = _first_choice(response).message.parsed

        return self._extract_response(response, parsed_output)


    def _consult_via_create(self, prompt, developer_override=None, few_shot_examples=None):
        """
        Consult via create() with JSON schema guided decoding (vLLM, SGLang)
        """
        kwargs = self._build_base_kwargs(prompt, developer_override, few_shot_examples)
        kwargs["max_tokens"] = self.max_completion_tokens

        if hasattr(self.response_format, 'model_json_schema'):
            schema = self.response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_format.__name__,
                    "strict": True,
                    "schema": schema,
                },
            }
        else:
            kwargs["response_format"] = self.response_format

        response = self.client.chat.completions.create(**kwargs)
        raw_content = _first_choice(response).message.content

        try:
            parsed_dict = json.loads(raw_content)
            if not isinstance(parsed_dict, dict):
                # Valid JSON but not an object — a provider that ignores response_format
                # and answers a bare `true`, a quoted string or an array. Raise here so
                # the plain-text fallback runs instead of the ** splat's TypeError.
                raise TypeError(
                    f"guided decoding returned non-object JSON ({type(parsed_dict).__name__})"
                )
            parsed_output = self.response_format(**parsed_dict)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            warning(f"_consult_via_create JSON-schema parse failed ({type(exc).__name__}: {exc}); "
                    f"falling back to _parse_plain_text_answer.")
            parsed_output = self._parse_plain_text_answer(raw_content)

        return self._extract_response(response, parsed_output)
