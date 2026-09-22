"""
Ontology-mediation arm: third-ontology mediated mapping composition plus LLM-oracle
diagnosis, reusing the baseline oracle consultation and set-union refinement unmodified.

The module is quarantined and off by default: `run_mediation_arm` raises
`MediationQuarantinedError` before any I/O, and `MediationConfig.enable_mediation` is False.
The composed fixture's `gold` column is restricted to the evaluation join; it never reaches
prompt construction or the oracle path.
A hard scope guard refuses any track outside the configured allowlist.
The set-union reuse is exact because the mediation M_ask is disjoint from the initial LogMap
alignment, so `_kg_refine_in_python`'s {initial - M_ask} ∪ {accepted} equals
initial ∪ {accepted} — a pure recall-expansion channel.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
import contextlib
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, model_validator

from logmap_llm.constants import (
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    COL_RELATION,
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
    PAIRS_SEPARATOR,
)
from logmap_llm.config.schema import OracleConfig
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.contracts import OracleResult
from logmap_llm.pipeline.orchestration import _kg_refine_in_python  # baseline set-union, reused unmodified
import logmap_llm.oracle.consultation as oc                          # baseline consultation, reused unmodified
from logmap_llm.evaluation.metrics import compute_prf, compute_oracle_metrics
from logmap_llm.utils.data import filter_accepted_predictions


class MediationQuarantinedError(RuntimeError):
    """Raised before any mediation I/O, inference, scoring, or provider access.

    The historical mediation evidence is reference-contaminated and
    is not a valid experimental condition. A future appendix must use fresh,
    provenance-disjoint inputs behind a separate entry point.
    """


MEDIATION_QUARANTINE_MESSAGE = (
    "MEDIATION_QUARANTINED: historical mediation evidence is "
    "reference-contaminated, historical-only, and excluded from the main campaign. "
    "No execution is permitted through this legacy entry point; a future independent "
    "appendix requires new provenance-disjoint inputs and explicit authorization."
)


# ---------------------------------------------------------------------------
# Track scope guard
# ---------------------------------------------------------------------------
# The mediation channel is only defined for tracks with a composed-candidate fixture; which
# tracks those are is a property of the shipped data, so the allowlist comes from configuration
# (`MediationConfig.scoped_tracks` / `scoped_track_prefixes`) rather than being hard-coded.
# Prefix entries admit decorated task labels (e.g. "<track>_body") under their track.


class MediationScopeError(ValueError):
    """Raised when the mediation arm is pointed at an out-of-scope track (hard stop)."""


def assert_mediation_scope(
    track: str,
    scoped_tracks: tuple[str, ...] | frozenset[str] = (),
    scoped_track_prefixes: tuple[str, ...] = (),
) -> None:
    """Refuse any track outside the configured allowlist; a no-op when none is configured."""
    if not scoped_tracks and not scoped_track_prefixes:
        return
    t = (track or "").strip().lower()
    allowed = frozenset(x.strip().lower() for x in scoped_tracks)
    prefixes = tuple(x.strip().lower() for x in scoped_track_prefixes)
    if t in allowed:
        return
    head = t.split("_")[0].split("-")[0]
    if (prefixes and t.startswith(prefixes)) or head in prefixes:
        return
    raise MediationScopeError(
        f"mediation arm refused for out-of-scope track {track!r}: the mediating-ontology "
        f"composition channel is defined only for tracks with a composed-candidate fixture. "
        f"Configured tracks: {sorted(allowed)}; configured prefixes: {sorted(prefixes)}."
    )


# ---------------------------------------------------------------------------
# Config (add-only; deliberately not a field of the baseline LogMapLLMConfig —
# the mediation arm has its own entry point).
# ---------------------------------------------------------------------------

class MediationConfig(BaseModel):
    """Configuration for the off-by-default ontology-mediation arm."""
    # the gate — must be explicitly True to run anything
    enable_mediation: bool = False

    # what to align over
    track: str
    composed_candidates_path: str            # copied fixture (src,tgt,N,gold)
    n_min: int = 1                           # support (vote-count) threshold — the precision lever
    initial_alignment_path: Optional[str] = None   # plain-LogMap initial alignment (pipe-sep, 5 col);
                                             # None allowed at construction, but run_mediation_arm
                                             # refuses it at run time

    # where artifacts go (its own tree). Isolation into a content-addressed {run_id} subdir is
    # hard-pinned on in run_mediation_arm (not a config knob) so the mediation arm can never write
    # a standard-pipeline artifact path.
    output_dir: str
    initial_dir: str
    refined_dir: str

    # evaluation join (gold quarantined to here). If None, the gold column of the composed
    # fixture is used as the (partial, stub) reference.
    reference_path: Optional[str] = None

    # oracle transport: deterministic offline stub by default; False means a served model.
    stub_oracle: bool = True
    stub_accept_hash_mod: int = 2            # stub accepts iff sha256(src|tgt) % mod == 0 (gold-free)
    max_workers: int = 4

    # --- served oracle (used iff stub_oracle=False) ----------------------------------------------
    # Both the transport and part of the run identity (see compute_mediation_run_id): without them,
    # two different panel models would collide on one run-id.
    model_name: Optional[str] = None
    model_revision: str = ""
    base_url: Optional[str] = None
    interaction_style: str = "vllm"
    api_key: str = "EMPTY"                   # may be an "ENV:VARNAME" sentinel; never a secret
    temperature: float = 0.0
    top_p: float = 1.0
    max_completion_tokens: int = 2048        # must be <= the server's max-model-len
    reasoning_effort: str = "minimal"
    response_mode: str = "structured"
    answer_format: str = "true_false"
    enable_thinking: bool = False
    seed: int = 42
    engine: str = "vllm"
    engine_version: str = ""

    # --- ontologies the candidates live in (required for a served run) ---------------------------
    # Without these the prompt degrades to bare URIs and the oracle rejects everything (measured).
    # run_mediation_arm hard-fails on a served run that omits them, rather than silently reporting
    # a prompt bug as "mediation adds nothing".
    onto_source_filepath: Optional[str] = None
    onto_target_filepath: Optional[str] = None
    owlready2_cache_dir: Optional[str] = None
    prompt_template: str = "one_level_of_parents_and_synonyms"   # the domain class template
    ontology_domain: str = ""
    # Tests only: lets the offline (no-ontology) isolation tests exercise the served path with the
    # URI-only builder. Never set it in a real experiment — that is the guard it disarms. (Gold is
    # stripped in build_mediation_m_ask before any prompt builder sees the frame.)
    allow_ungrounded_prompts: bool = False

    # n_min sweep without re-paying the oracle: filter_by_support runs before the oracle and a
    # verdict does not depend on n_min (few_shot_examples=None, each prompt is built from the pair
    # alone), so higher-n_min arms may reuse the n_min=1 arm's predictions CSV. Each arm still gets
    # its own tree, refined alignment and evaluation; the reused predictions are folded into the
    # run-id.
    reuse_predictions_csv: Optional[str] = None

    # scope safety
    track_scope_guard: bool = True
    scoped_tracks: tuple[str, ...] = ()
    scoped_track_prefixes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _guard(self) -> "MediationConfig":
        if self.n_min < 1:
            raise ValueError(f"n_min must be >= 1 (got {self.n_min}).")
        if self.stub_accept_hash_mod < 1:
            raise ValueError(f"stub_accept_hash_mod must be >= 1 (got {self.stub_accept_hash_mod}).")
        if self.track_scope_guard:
            assert_mediation_scope(self.track, self.scoped_tracks, self.scoped_track_prefixes)
        return self


# ---------------------------------------------------------------------------
# Composed-candidate loading, support filter, M_ask coercion
# ---------------------------------------------------------------------------

_COMPOSED_COLS = ["source_entity_uri", "target_entity_uri", "N", "gold"]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_composed_candidates(composed_path: str | Path) -> pd.DataFrame:
    """
    Load a composed-candidate fixture: CSV, comma-separated, no header, columns
    (source_entity_uri, target_entity_uri, N, gold).

    `gold` is quarantined — downstream it is read only by the evaluation join, never by prompt
    construction or the oracle path. `N` is the mediating-ontology vote count used by
    `filter_by_support`.
    """
    df = pd.read_csv(composed_path, header=None, names=_COMPOSED_COLS, dtype=str)
    df["N"] = pd.to_numeric(df["N"], errors="coerce").astype("Int64")
    # keep gold as its raw string form (True/False/or an injected sentinel) for the eval join only
    df["gold_raw"] = df["gold"].astype(str)
    return df


def filter_by_support(df: pd.DataFrame, n_min: int) -> pd.DataFrame:
    """Keep only candidates supported by at least `n_min` mediating ontologies (the precision
    lever): higher `n_min` -> fewer, higher-agreement candidates."""
    if n_min <= 1:
        kept = df[df["N"].notna()].copy()
    else:
        kept = df[df["N"].fillna(0).astype(int) >= n_min].copy()
    return kept.reset_index(drop=True)


def build_mediation_m_ask(df: pd.DataFrame) -> pd.DataFrame:
    """
    Coerce composed candidates into the internal M_ask schema so the baseline oracle path is
    reused unmodified. Output columns (exact order the consultation + refinement code expects):

        [source_entity_uri, target_entity_uri, relation, confidence, entityType]

    `N` and `gold` are stripped here — the M_ask that reaches prompts/oracle carries no support
    count and no gold label. Composed candidates are class equivalences => entityType=CLS,
    relation="=". `confidence` is a mediation-support proxy in (0,1] (N normalised), used only
    for the alignment record; set-based scoring ignores it.
    """
    if len(df) == 0:
        return pd.DataFrame(columns=[
            COL_SOURCE_ENTITY_URI, COL_TARGET_ENTITY_URI, COL_RELATION, COL_CONFIDENCE, COL_ENTITY_TYPE
        ])
    n = df["N"].fillna(1).astype(int)
    n_max = max(int(n.max()), 1)
    m_ask = pd.DataFrame({
        COL_SOURCE_ENTITY_URI: df["source_entity_uri"].astype(str).values,
        COL_TARGET_ENTITY_URI: df["target_entity_uri"].astype(str).values,
        COL_RELATION: "=",
        COL_CONFIDENCE: (n / n_max).clip(upper=1.0).astype(float).values,
        COL_ENTITY_TYPE: "CLS",
    })
    return m_ask


def build_mediation_prompts_grounded(
    m_ask: pd.DataFrame,
    onto_source_filepath: str,
    onto_target_filepath: str,
    cache_dir: Optional[str] = None,
    template_name: str = "one_level_of_parents_and_synonyms",
    ontology_domain: str = "",
) -> dict[str, str]:
    """
    Build ontology-grounded prompts for the composed candidates.

    Calls the baseline prompt builder (`oracle.prompts.templates.build_oracle_user_prompts`) over
    the baseline ontology loader (`ontology.access.load_ontologies`) — the exact pair `stage_two`
    uses — so a mediation candidate is described to the oracle exactly like a LogMap M_ask
    candidate: same model, same template, same ontology context; only the candidate's origin
    differs. Candidates are class equivalences, so only the class template is needed; `gold` is
    already stripped from `m_ask` before it reaches here.
    """
    from logmap_llm.ontology.access import load_ontologies          # baseline loader, reused unmodified
    import logmap_llm.oracle.prompts.templates as opb               # baseline templates, reused unmodified

    from logmap_llm.oracle.prompts.context import PromptContext

    OA_source, OA_target = load_ontologies(
        onto_source_filepath, onto_target_filepath, cache_dir=cache_dir,
    )
    prompts = opb.build_oracle_user_prompts(
        template_name, onto_source_filepath, onto_target_filepath, m_ask,
        OA_source=OA_source, OA_target=OA_target,
        ctx=PromptContext(ontology_domain=ontology_domain),
    )
    if not prompts:
        raise RuntimeError(
            "grounded mediation prompt build produced no prompts — refusing to consult an oracle with "
            "an empty prompt set (that would silently reproduce the URI-only failure mode)."
        )
    return prompts


def build_mediation_prompts(m_ask: pd.DataFrame) -> dict[str, str]:
    """
    Offline URI-only prompt dict, kept for the stub-oracle isolation tests, which must remain
    runnable with no ontologies present.

    Deprecated for real runs: a prompt that names only URIs carries no signal and a real oracle
    will reject essentially everything. `run_mediation_arm` selects the grounded builder whenever
    the ontologies are supplied, and refuses a served run without them. Each prompt embeds the
    `src|tgt` key (so the deterministic stub's decision is unambiguous per candidate) and never
    any gold label — the M_ask fed here already has gold stripped.
    """
    prompts: dict[str, str] = {}
    for _, row in m_ask.iterrows():
        src = str(row[COL_SOURCE_ENTITY_URI])
        tgt = str(row[COL_TARGET_ENTITY_URI])
        key = f"{src}{PAIRS_SEPARATOR}{tgt}"
        prompts[key] = (
            f"[mediation candidate {key}] Are the entity <{src}> and the entity <{tgt}> "
            f"equivalent? Answer true or false."
        )
    return prompts


# ---------------------------------------------------------------------------
# Deterministic offline stub oracle
# ---------------------------------------------------------------------------

def _stable_accept(prompt_or_key: str, mod: int) -> bool:
    """Deterministic accept decision. Uses sha256 (not the salted builtin hash()) so the decision
    is identical across processes and runs. Depends only on the prompt text (which encodes the
    src|tgt pair, never gold)."""
    digest = hashlib.sha256(prompt_or_key.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % mod) == 0


class _StubMessage:
    def __init__(self, content: str):
        self.content = content
        self.parsed = None


class _StubChoice:
    def __init__(self, content: str):
        self.message = _StubMessage(content)
        self.logprobs = None            # -> _extract_response falls back to logprobs=[]


class _StubUsage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _StubResponse:
    def __init__(self, content: str):
        self.choices = [_StubChoice(content)]
        self.usage = _StubUsage()


class _StubCompletions:
    def __init__(self, mod: int):
        self._mod = mod

    def _decide(self, kwargs) -> str:
        # the target query is the last user message assembled by the baseline manager
        messages = kwargs.get("messages", [])
        target = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                target = msg.get("content", "")
                break
        return "true" if _stable_accept(target, self._mod) else "false"

    def create(self, **kwargs):          # baseline `_consult_via_plain` calls create()
        return _StubResponse(self._decide(kwargs))

    def parse(self, **kwargs):           # defensive: never used in plain mode
        return _StubResponse(self._decide(kwargs))


class _StubChat:
    def __init__(self, mod: int):
        self.completions = _StubCompletions(mod)


class StubOpenAIClient:
    """A deterministic, offline, OpenAI-compatible stand-in. Constructed with the accept modulus;
    ignores api_key/base_url. Only `chat.completions.create` is exercised (plain response mode)."""
    def __init__(self, mod: int = 2, *args, **kwargs):
        self.chat = _StubChat(mod)


@contextlib.contextmanager
def patched_stub_transport(mod: int):
    """Scoped monkeypatch of the network client the baseline oracle manager constructs. Reverts
    on exit, so it cannot leak into the standard arm or any concurrent run."""
    import logmap_llm.oracle.manager as mgr
    original = mgr.OpenAI

    def _factory(*args, **kwargs):
        return StubOpenAIClient(mod, *args, **kwargs)

    mgr.OpenAI = _factory
    try:
        yield
    finally:
        mgr.OpenAI = original


def _stub_oracle_config(mcfg: MediationConfig) -> OracleConfig:
    """A minimal OracleConfig for the offline stub: plain response mode routes the baseline manager
    to `_consult_via_plain` (the simplest transport to stub); base_url is inert under the patch."""
    return OracleConfig(
        model_name="stub-mediation-oracle",
        api_key="EMPTY",
        base_url="http://localhost:0/v1",       # inert: the stub client ignores it
        interaction_style="vllm",
        response_mode="plain",
        answer_format="true_false",
        temperature=0.0,
        top_p=1.0,
        max_workers=mcfg.max_workers,
        enable_thinking=False,
        max_completion_tokens=8,
    )


def _served_oracle_config(mcfg: MediationConfig) -> OracleConfig:
    """Served-model OracleConfig. Same OracleConfig type the standard pipeline builds, so
    `consult_oracle_for_mappings_to_ask` is reached completely unmodified."""
    if not mcfg.model_name or not mcfg.base_url:
        raise ValueError(
            "stub_oracle=False requires model_name and base_url on MediationConfig (the served-model "
            "identity). Refusing to run a 'real' mediation arm against the stub endpoint."
        )
    return OracleConfig(
        model_name=mcfg.model_name,
        api_key=mcfg.api_key,               # may be an ENV: sentinel; resolved at call time
        base_url=mcfg.base_url,
        interaction_style=mcfg.interaction_style,
        response_mode=mcfg.response_mode,
        answer_format=mcfg.answer_format,
        temperature=mcfg.temperature,
        top_p=mcfg.top_p,
        max_workers=mcfg.max_workers,
        enable_thinking=mcfg.enable_thinking,
        max_completion_tokens=mcfg.max_completion_tokens,
        reasoning_effort=mcfg.reasoning_effort,
    )


def _oracle_config(mcfg: MediationConfig) -> OracleConfig:
    return _stub_oracle_config(mcfg) if mcfg.stub_oracle else _served_oracle_config(mcfg)


# ---------------------------------------------------------------------------
# Run identity + a minimal ctx that exposes only what _kg_refine_in_python reads
# ---------------------------------------------------------------------------

def compute_mediation_run_id(mcfg: MediationConfig, composed_sha: str,
                             initial_sha: str, reference_sha: str) -> str:
    """Content-addressed mediation run id. Folds in arm='mediation' plus every input that can
    change the refined output or the evaluation — the composed set, the initial alignment
    (refined = initial ∪ accepted, so a different initial is a different experiment), the external
    reference override, and the support-filter/oracle-transport params — so two runs collide on a
    run-id only when they are the same experiment. The 'med-' prefix keeps the namespace disjoint
    from the baseline compute_run_id 16-hex ids."""
    ident = {
        "arm": "mediation",
        "track": mcfg.track,
        "composed_sha256": composed_sha,
        "initial_alignment_sha256": initial_sha,
        "reference_sha256": reference_sha,
        "n_min": mcfg.n_min,
        "stub_oracle": mcfg.stub_oracle,
        "stub_accept_hash_mod": mcfg.stub_accept_hash_mod,
    }
    # With a real served model the oracle's identity is the single biggest determinant of the
    # result, so it must be in the content address — otherwise panel models would hash to the same
    # run-id for a given (track, n_min) and clobber each other. (api_key is deliberately absent:
    # a secret that never changes the experiment. max_workers is absent: perf-only.)
    if not mcfg.stub_oracle:
        ident["served_model"] = {
            "model_name": mcfg.model_name,
            "model_revision": mcfg.model_revision,
            "base_url": mcfg.base_url,
            "interaction_style": mcfg.interaction_style,
            "temperature": mcfg.temperature,
            "top_p": mcfg.top_p,
            "max_completion_tokens": mcfg.max_completion_tokens,
            "reasoning_effort": mcfg.reasoning_effort,
            "response_mode": mcfg.response_mode,
            "answer_format": mcfg.answer_format,
            "enable_thinking": mcfg.enable_thinking,
            "seed": mcfg.seed,
            "engine": mcfg.engine,
            "engine_version": mcfg.engine_version,
        }
    # An arm that consumed another arm's predictions is not the same experiment as one that
    # consulted the oracle itself, even at the same n_min — record which.
    if mcfg.reuse_predictions_csv:
        ident["reuse_predictions_sha256"] = (
            sha256_file(mcfg.reuse_predictions_csv)
            if os.path.exists(mcfg.reuse_predictions_csv) else "missing"
        )
    blob = json.dumps(ident, sort_keys=True, default=str)
    return "med-" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _make_ctx(run_paths: PipelinePaths) -> PipelineContext:
    """`_kg_refine_in_python` reads only ctx.run_paths; logmap/cfg are unused there, so a minimal
    context keeps the mediation arm free of the JVM/LogMap bootstrap."""
    return PipelineContext(cfg=None, run_paths=run_paths, logmap=None, config_path=None)


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------

def run_mediation_arm(mcfg: MediationConfig) -> dict:
    """
    Execute one mediation arm end to end:

        load composed -> filter_by_support(N) -> build own M_ask (gold stripped)
        -> baseline consult_oracle_for_mappings_to_ask
        -> baseline _kg_refine_in_python set-union (refined = initial ∪ accepted)
        -> evaluation_results.json (gold read only here)

    Returns a summary dict and writes: predictions CSV, refined alignment TSV,
    evaluation_results.json, and a mediation run manifest — all under the run's own isolated tree.
    """
    # Hard stop before inspecting paths, creating directories, contacting a
    # provider, loading a model, invoking a matcher, or running evaluation.
    raise MediationQuarantinedError(MEDIATION_QUARANTINE_MESSAGE)

    if not mcfg.enable_mediation:  # pragma: no cover - retained historical implementation
        raise RuntimeError(
            "run_mediation_arm called with enable_mediation=False. The mediation arm is OFF by "
            "default; set enable_mediation=true explicitly to run it."
        )
    if mcfg.track_scope_guard:
        assert_mediation_scope(mcfg.track, mcfg.scoped_tracks, mcfg.scoped_track_prefixes)

    # The mediation arm is additive to the plain-LogMap alignment (refined = initial ∪ accepted),
    # so a non-empty initial is required. Validate it before the run-id so the run-id can
    # content-address it (an empty initial is refused rather than silently fabricated; it would
    # also crash the baseline set-union's pd.read_csv).
    if not (mcfg.initial_alignment_path and os.path.exists(mcfg.initial_alignment_path)
            and os.path.getsize(mcfg.initial_alignment_path) > 0):
        raise ValueError(
            "run_mediation_arm requires a non-empty initial_alignment_path (the plain-LogMap initial "
            "alignment the mediation candidates expand). At P8 this is the real LogMap alignment; "
            f"for the P5.5 offline dry run supply a fixture. Got: {mcfg.initial_alignment_path!r}."
        )

    composed_sha = sha256_file(mcfg.composed_candidates_path)
    initial_sha = sha256_file(mcfg.initial_alignment_path)
    reference_sha = (sha256_file(mcfg.reference_path)
                     if mcfg.reference_path and os.path.exists(mcfg.reference_path) else "none")
    run_id = compute_mediation_run_id(mcfg, composed_sha, initial_sha, reference_sha)

    # isolate_run is hard-pinned True: the mediation arm always writes to its own content-addressed
    # run-id tree, so it can never overwrite a standard-pipeline artifact path.
    run_paths = PipelinePaths(
        output_dir=mcfg.output_dir,
        initial_dir=mcfg.initial_dir,
        refined_dir=mcfg.refined_dir,
        task_name=mcfg.track,
        oupt_name="mediation",
        run_id=run_id,
        isolate_run=True,
    )
    run_paths.create_base_dirs()

    # --- initial LogMap alignment (its own copy at the exact path _kg_refine_in_python reads) ---
    initial_path = run_paths.logmap_mappings()
    initial_df = pd.read_csv(mcfg.initial_alignment_path, sep=PAIRS_SEPARATOR, header=None)
    n_initial = len(initial_df)
    import shutil
    shutil.copyfile(mcfg.initial_alignment_path, initial_path)

    # --- load composed candidates, apply the support filter, build our own M_ask ---
    composed = load_composed_candidates(mcfg.composed_candidates_path)
    n_composed_total = len(composed)
    filtered = filter_by_support(composed, mcfg.n_min)
    n_after_filter = len(filtered)
    m_ask = build_mediation_m_ask(filtered)               # gold + N stripped here

    # Prompt construction. A served run must be ontology-grounded: bare URIs are unanswerable and
    # the oracle would reject everything, which would look exactly like "mediation contributes
    # nothing" — refuse rather than produce that. The URI-only builder is for the offline stub only.
    grounded = bool(mcfg.onto_source_filepath and mcfg.onto_target_filepath)
    if not mcfg.stub_oracle and not grounded and not mcfg.allow_ungrounded_prompts:
        raise ValueError(
            "a served mediation run requires onto_source_filepath + onto_target_filepath: without them "
            "the prompt names only URIs, which carries no signal (measured: 394/394 rejected). Refusing "
            "to report a prompt bug as a mediation result."
        )
    if grounded:
        prompts = build_mediation_prompts_grounded(
            m_ask,
            onto_source_filepath=mcfg.onto_source_filepath,
            onto_target_filepath=mcfg.onto_target_filepath,
            cache_dir=mcfg.owlready2_cache_dir,
            template_name=mcfg.prompt_template,
            ontology_domain=mcfg.ontology_domain,
        )
    else:
        prompts = build_mediation_prompts(m_ask)

    # --- baseline oracle consultation ---
    # stub_oracle wraps the call in the deterministic stub transport; otherwise nullcontext, so
    # the same baseline call talks to a real served model.
    if mcfg.reuse_predictions_csv:
        # The n_min arms ask the same model the same per-pair questions over nested subsets, so
        # take the cached arm's predictions restricted to this arm's M_ask rather than
        # re-consulting. Only sound because the prompt for a pair depends on the pair alone
        # (few_shot_examples=None) — assert the coverage rather than trusting it.
        cached = pd.read_csv(mcfg.reuse_predictions_csv)
        cached = cached.loc[:, ~cached.columns.str.startswith("Unnamed")]   # drop stray index columns
        want_pairs = {
            (str(r[COL_SOURCE_ENTITY_URI]), str(r[COL_TARGET_ENTITY_URI]))
            for _, r in m_ask.iterrows()
        }
        keep = cached.apply(
            lambda r: (str(r[COL_SOURCE_ENTITY_URI]), str(r[COL_TARGET_ENTITY_URI])) in want_pairs,
            axis=1,
        )
        predictions_df = cached[keep].reset_index(drop=True) if len(cached) else cached
        got_pairs = {
            (str(r[COL_SOURCE_ENTITY_URI]), str(r[COL_TARGET_ENTITY_URI]))
            for _, r in predictions_df.iterrows()
        }
        missing = want_pairs - got_pairs
        if missing:
            raise RuntimeError(
                f"reuse_predictions_csv does not cover this arm's M_ask: {len(missing)} of "
                f"{len(want_pairs)} candidates absent (e.g. {sorted(missing)[:3]}). The cached "
                f"predictions must be a SUPERSET (run n_min=1 first). Refusing to score a partial arm."
            )
    else:
        transport = (patched_stub_transport(mcfg.stub_accept_hash_mod)
                     if mcfg.stub_oracle else contextlib.nullcontext())
        with transport:
            predictions_df = oc.consult_oracle_for_mappings_to_ask(
                m_ask_prompts=prompts,
                m_ask_init_alignment_df=m_ask,
                oracle_cfg=_oracle_config(mcfg),
                developer_prompt_text="Decide whether the two entities are equivalent.",
                developer_prompt_map=None,
                few_shot_examples=None,
            )

    if predictions_df is None:
        raise RuntimeError("mediation oracle consultation aborted (failure-abort tripped).")

    predictions_df.to_csv(run_paths.predictions_csv(), na_rep="nan", index=False)

    # --- baseline set-union refinement ---
    oracle_result = OracleResult(predictions=predictions_df)
    ctx = _make_ctx(run_paths)
    _kg_refine_in_python(ctx, oracle_result)              # writes run_paths.refined_mappings_tsv()

    # --- evaluation (gold read only here, for the join) ---
    accepted = filter_accepted_predictions(predictions_df)
    accepted_set = {
        (str(r[COL_SOURCE_ENTITY_URI]), str(r[COL_TARGET_ENTITY_URI]))
        for _, r in accepted.iterrows()
    }
    gold_pos_set = _load_gold_positive_reference(composed, mcfg.reference_path)

    preds_list = [
        {
            "source": str(r[COL_SOURCE_ENTITY_URI]),
            "target": str(r[COL_TARGET_ENTITY_URI]),
            "prediction": (None if r["Oracle_prediction"] in ("error", "skipped")
                           else bool(r["Oracle_prediction"])),
            "confidence": r.get("Oracle_confidence"),
        }
        for _, r in predictions_df.iterrows()
    ]

    global_block = compute_prf(accepted_set, gold_pos_set)
    global_block["note"] = (
        "STUB partial-reference: system = oracle-accepted composed candidates; reference = composed "
        "gold-positives only. This is the standalone mediation-diagnosis metric, NOT a full-track "
        "score. Full-reference DeepOnto/MELT scoring against the official reference is P8."
    )
    oracle_block = compute_oracle_metrics(preds_list, gold_pos_set, partial_reference=False)

    refined_df = pd.read_csv(run_paths.refined_mappings_tsv(), sep="\t", header=None) \
        if os.path.getsize(run_paths.refined_mappings_tsv()) > 0 else pd.DataFrame()
    n_refined = len(refined_df)

    eval_results = {
        "arm": "mediation",
        "track": mcfg.track,
        "is_stub": bool(mcfg.stub_oracle),
        "protocol": "stub_composed_partial_reference",
        "run_id": run_id,
        "composed_set_sha256": composed_sha,
        "n_min": mcfg.n_min,
        "counts": {
            "n_composed_total": int(n_composed_total),
            "n_after_support_filter": int(n_after_filter),
            "n_consulted": int(len(predictions_df)),
            "n_accepted": int(len(accepted_set)),
            "n_initial_alignment": int(n_initial),
            "n_refined_total": int(n_refined),
            "gold_positives_in_reference": int(len(gold_pos_set)),
        },
        "global": global_block,
        "oracle": oracle_block,
        "stub_oracle": {
            "enabled": bool(mcfg.stub_oracle),
            "decision": f"deterministic sha256(prompt)%{mcfg.stub_accept_hash_mod}==0; gold NEVER read",
        },
        "served_model": (None if mcfg.stub_oracle else {
            "model_name": mcfg.model_name, "model_revision": mcfg.model_revision,
            "base_url": mcfg.base_url, "interaction_style": mcfg.interaction_style,
            "max_completion_tokens": mcfg.max_completion_tokens,
            "reasoning_effort": mcfg.reasoning_effort, "seed": mcfg.seed,
            "engine": mcfg.engine, "engine_version": mcfg.engine_version,
        }),
        "predictions_reused_from": mcfg.reuse_predictions_csv,
    }
    with open(run_paths.eval_json(), "w") as fp:
        json.dump(eval_results, fp, indent=2, default=str)

    manifest = {
        "arm": "mediation",
        "run_id": run_id,
        "track": mcfg.track,
        "n_min": mcfg.n_min,
        "composed_candidates_path": str(mcfg.composed_candidates_path),
        "composed_set_sha256": composed_sha,
        "candidate_source_sha256": composed_sha,   # candidate source == composed set (not M_ask)
        "initial_alignment_sha256": initial_sha,   # folded into the run-id
        "reference_sha256": reference_sha,         # folded into the run-id
        "isolate_run": True,                       # hard-pinned on for the mediation arm
        "output_dir": str(run_paths.output_dir),
        "initial_dir": str(run_paths.initial_dir),
        "refined_dir": str(run_paths.refined_dir),
        "predictions_csv": str(run_paths.predictions_csv()),
        "refined_tsv": str(run_paths.refined_mappings_tsv()),
        "evaluation_results": str(run_paths.eval_json()),
        "stub_oracle": bool(mcfg.stub_oracle),
    }
    manifest_path = run_paths.output_dir / f"{mcfg.track}-mediation-run-manifest.json"
    with open(manifest_path, "w") as fp:
        json.dump(manifest, fp, indent=2, default=str)

    return {
        "run_id": run_id,
        "run_dir": str(run_paths.output_dir),
        "manifest": str(manifest_path),
        "evaluation_results": str(run_paths.eval_json()),
        "eval": eval_results,
    }


def _load_gold_positive_reference(composed: pd.DataFrame, reference_path: Optional[str]) -> set[tuple[str, str]]:
    """Build the (partial, stub) reference of gold-positive (src,tgt) pairs. This is the only
    place the `gold` column is read. If an external reference file is supplied, it wins."""
    if reference_path and os.path.exists(reference_path):
        # Track references are tab-separated; delegate to the evaluation module's loader, the
        # project's single source of truth for mapping-file parsing (handles tab/pipe/comma).
        from logmap_llm.evaluation.io import load_mapping_pairs
        return {(str(a), str(b)) for a, b in load_mapping_pairs(Path(reference_path))}
    is_pos = composed["gold_raw"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    pos = composed[is_pos]
    return {
        (str(r["source_entity_uri"]), str(r["target_entity_uri"]))
        for _, r in pos.iterrows()
    }


# ---------------------------------------------------------------------------
# CLI  (own entry point: `python -m logmap_llm.pipeline.mediation ...` — no baseline dispatch)
# ---------------------------------------------------------------------------

def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LogMapLLM ontology-mediation arm (OFF by default).")
    p.add_argument("--track", required=True)
    p.add_argument("--composed", required=True, help="copied composed-candidate fixture (src,tgt,N,gold)")
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--initial-alignment", default=None)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--initial-dir", required=True)
    p.add_argument("--refined-dir", required=True)
    p.add_argument("--reference", default=None)
    p.add_argument("--enable-mediation", action="store_true", help="required to run (the gate)")
    p.add_argument("--stub-accept-hash-mod", type=int, default=2)
    p.add_argument("--no-scope-guard", action="store_true", help="DEBUG only; disables the scope guard")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    mcfg = MediationConfig(
        enable_mediation=args.enable_mediation,
        track=args.track,
        composed_candidates_path=args.composed,
        n_min=args.n_min,
        initial_alignment_path=args.initial_alignment,
        output_dir=args.output_dir,
        initial_dir=args.initial_dir,
        refined_dir=args.refined_dir,
        reference_path=args.reference,
        stub_oracle=True,
        stub_accept_hash_mod=args.stub_accept_hash_mod,
        track_scope_guard=not args.no_scope_guard,
    )
    result = run_mediation_arm(mcfg)
    print(json.dumps(result["eval"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
