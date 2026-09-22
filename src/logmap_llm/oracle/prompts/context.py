"""
logmap_llm.oracle.prompts.context

The immutable per-run prompt context: a frozen dataclass constructed once per run and
passed explicitly to every template, so no rendered prompt depends on mutable
module-level state.

The derived accessors must reproduce the rendered prompt text byte for byte, including
the single-space fallback in `forced_domain_string`; `tests/goldens/prompt_templates.json`
pins all registered templates across both domain settings and all four
(answer_format, response_mode) combinations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from logmap_llm.constants import (
    ANSWER_FORMATS,
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_RESPONSE_MODE,
    RESPONSE_INSTRUCTION,
    RESPONSE_MODES,
)
from logmap_llm.oracle.prompts.formatting import format_instance_attribute_clause


@dataclass(frozen=True)
class PromptContext:
    """Everything a prompt template needs beyond the two entities it is rendering."""

    answer_format: str = DEFAULT_ANSWER_FORMAT
    response_mode: str = DEFAULT_RESPONSE_MODE
    ontology_domain: str | None = None
    #: Formats one instance attribute clause. Only the instance templates consult it, and
    #: only when the caller has not passed an explicit `fmt_fn`.
    instance_fmt_fn: Callable = format_instance_attribute_clause

    def __post_init__(self) -> None:
        if self.answer_format not in ANSWER_FORMATS:
            raise ValueError(
                f"Unknown answer_format '{self.answer_format}'. "
                f"Valid options: {sorted(ANSWER_FORMATS)}"
            )
        if self.response_mode not in RESPONSE_MODES:
            raise ValueError(
                f"Unknown response_mode '{self.response_mode}'. "
                f"Valid options: {sorted(RESPONSE_MODES)}"
            )

    @classmethod
    def from_config(cls, config, *, instance_fmt_fn: Callable | None = None) -> "PromptContext":
        """Build the context for one run from its validated LogMapLLMConfig.

        The two response fields are passed through unconverted: `AnswerFormat` and
        `ResponseModes` are `(str, Enum)` members that compare and hash equal to their
        values, satisfying both the `ANSWER_FORMATS`/`RESPONSE_MODES` membership checks
        and the `RESPONSE_INSTRUCTION` tuple lookup. `str()` on such a member returns
        `'AnswerFormat.TRUE_FALSE'`, not `'true_false'`, so coercing here would make
        `__post_init__` reject every valid config. See
        tests/test_prompt_context_from_config.py.
        """
        return cls(
            answer_format=config.oracle.answer_format,
            response_mode=config.oracle.response_mode,
            ontology_domain=config.alignmentTask.ontology_domain,
            instance_fmt_fn=instance_fmt_fn or format_instance_attribute_clause,
        )

    # --- derived accessors; rendered text must stay byte-stable ------

    @property
    def response_instruction(self) -> str:
        """The instruction appended to a prompt; depends on the (format, mode) pair."""
        return RESPONSE_INSTRUCTION[(self.answer_format, self.response_mode)]

    @property
    def domain_preamble(self) -> str:
        if self.ontology_domain:
            return f"We have two entities from different {self.ontology_domain} ontologies."
        return "We have two entities from different ontologies."

    @property
    def forced_domain_string(self) -> str:
        """A space-padded domain qualifier, or a single space.

        Some legacy prompts hard-code a domain qualifier ("two <domain> ontologies") and differ
        from the expected domain preamble. For those the domain string is "forced", even if none
        has been set, so the surrounding spacing stays correct.
        """
        if self.ontology_domain:
            return str(f" {self.ontology_domain} ")
        return " "
