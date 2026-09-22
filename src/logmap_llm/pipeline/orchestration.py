"""
logmap_llm.pipeline.orchestration

Ported from jd-extended, see:

    https://github.com/jonathondilworth/logmap-llm/blob/jd-extended/pipeline_steps.py

With some modifications (ready for future branches/features).
"""
from __future__ import annotations

import sys
import os
import subprocess
import json
import pandas as pd
import numpy as np

from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.contracts import (
    AlignmentResult,
    PromptBuildResult,
    OracleResult,
    ModelSelectionResult,
    RefinementResult,
    EvaluationResult,
)
from logmap_llm.constants import (
    AlignMode,
    PromptBuildMode,
    ConsultMode,
    RefineMode,
    RefinementStrategy,
    PAIRS_SEPARATOR,
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    COL_RELATION,
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
    DEFAULT_CONFIDENCE_FALLBACK,
)
from logmap_llm.utils.data import (
    normalise_prediction_column,
    filter_accepted_predictions,
)
from logmap_llm.utils.io import atomic_json_write_strict, atomic_write_text_strict
from logmap_llm.utils.logging import (
    fatal,
    critical,
    warning,
    warn,
    step,
    success,
)

# NOTE: Do not import oracle_prompt_building or onto_access here.
# Those modules transitively import owlready2, which cannot coexist
# with JPype in the same process.  Prompt building runs in a subprocess.


# PRIVATE HELPERS
#################

def _subprocess_context_args(ctx: PipelineContext) -> list[str]:
    """Arguments that keep child processes in the parent's run namespace."""
    args: list[str] = []
    if ctx.run_paths.run_root is not None:
        args.extend(["--run-root", str(ctx.run_paths.run_root)])
    if ctx.no_cache:
        args.append("--no-cache")
    return args


def _detect_bidirectional(prompts: dict) -> bool:
    """
    Detect whether prompts were built in bidirectional mode; bidirectional prompt keys
    have a direction suffix, specifically: 'src_uri|tgt_uri|REVERSE', whereas standard
    keys are just: 'src_uri|tgt_uri'
    """
    if not prompts:
        return False
    return any(key.count(PAIRS_SEPARATOR) >= 2 for key in prompts)


def _configured_oracle_params(oracle_cfg, capabilities: dict | None = None) -> dict:
    """Serializable inference controls, including no-candidate consultations."""
    capabilities = capabilities or {}
    return {
        "model_name": oracle_cfg.model_name,
        "interaction_style": oracle_cfg.interaction_style,
        "temperature": oracle_cfg.temperature,
        "top_p": oracle_cfg.top_p,
        "max_completion_tokens": oracle_cfg.max_completion_tokens,
        "reasoning_effort": oracle_cfg.reasoning_effort,
        "reasoning_token_budget": oracle_cfg.reasoning_token_budget,
        "enable_thinking": oracle_cfg.enable_thinking,
        "max_workers": oracle_cfg.max_workers,
        "answer_format": oracle_cfg.answer_format,
        "response_mode": oracle_cfg.response_mode,
        "request_timeout_seconds": oracle_cfg.request_timeout_seconds,
        "connect_timeout_seconds": oracle_cfg.connect_timeout_seconds,
        "transient_retries": oracle_cfg.transient_retries,
        "seed": oracle_cfg.seed,
        "request_logprobs": oracle_cfg.request_logprobs,
        "openrouter_provider": oracle_cfg.openrouter_provider,
        "openrouter_allow_fallbacks": oracle_cfg.openrouter_allow_fallbacks,
        "openrouter_require_parameters": oracle_cfg.openrouter_require_parameters,
        "logprobs_requested": capabilities.get(
            "logprobs_requested", oracle_cfg.request_logprobs
        ),
        "logprobs_effective": capabilities.get(
            "logprobs_effective", oracle_cfg.request_logprobs
        ),
        "capability_downgrades": capabilities.get("downgrades", []),
    }


def _validate_prompt_keys(prompts: dict, bidirectional: bool, template_name: str = "") -> None:
    """
    Validates that the prompt key format matches the consultation mode.
    """
    if not prompts:
        return

    keys_have_direction = _detect_bidirectional(prompts)

    if bidirectional and not keys_have_direction:
        raise ValueError(
            f"Bidirectional template '{template_name}' selected but prompts   "
            f"lack direction keys (eg. '...|REVERSE'). The prompt JSON was    "
            f"likely built with a non-bidirectional template. Rebuild prompts "
            f"or select a matching template."
        )
    if not bidirectional and keys_have_direction:
        raise ValueError(
            f"Standard template '{template_name}' selected but prompts have  "
            f"direction keys (eg. '...|REVERSE'). The prompt JSON was likely "
            f"built with a bidirectional template. Rebuild prompts or select "
            f"a matching template."
        )


def _load_few_shot_artifact(
    path,
    *,
    few_shot_cfg,
    oracle_cfg,
    expected_query_keys,
):
    """Load few-shot JSON, revalidating strict prebuilt handoffs before inference."""
    with open(path, encoding="utf-8") as stream:
        loaded = json.load(stream)

    if few_shot_cfg.prebuilt_few_shot_bundle_path is not None:
        # Stage two is a subprocess. Revalidate its persisted handoff in this
        # consumer process so a stale or modified artifact cannot become a
        # shared prefix or silently drop one query to zero-shot.
        from logmap_llm.pipeline.rag_fewshot import (
            validate_prebuilt_few_shot_examples,
        )

        validate_prebuilt_few_shot_examples(
            loaded,
            expected_query_keys=expected_query_keys,
            k=few_shot_cfg.few_shot_k,
            answer_format=oracle_cfg.answer_format,
            response_mode=oracle_cfg.response_mode,
        )

    if isinstance(loaded, dict):
        if few_shot_cfg.prebuilt_few_shot_bundle_path is None:
            # stage_two always writes one entry per prompt key (zero-shot fallbacks
            # keep their key with an empty pair list), so exact key coverage is the
            # invariant of a fresh, matching artifact. A stale empty or partial dict
            # would silently drop queries to zero-shot under --reuse-prompts.
            expected = {str(key) for key in expected_query_keys}
            actual = {str(key) for key in loaded}
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    "Few-shot artifact keys do not cover the prompt keys — the artifact is "
                    "stale for this configuration (e.g. left over from a k=0 run or a failed "
                    "RAG generation under --reuse-prompts); rebuild prompts to regenerate it "
                    f"(missing={missing[:5]!r}, extra={extra[:5]!r})"
                )
        return {
            key: [tuple(pair) for pair in pairs]
            for key, pairs in loaded.items()
        }
    if isinstance(loaded, list):
        if not loaded:
            raise ValueError(
                "Few-shot artifact is an empty list while few_shot_k > 0; refusing to run "
                "the condition as an undeclared zero-shot baseline. Rebuild prompts to "
                "regenerate the artifact."
            )
        return [tuple(pair) for pair in loaded]
    raise ValueError("Few-shot artifact must contain a JSON object or list")


def _resolve_accepted_confidences(accepted_subset: pd.DataFrame, oracle_accepted: pd.DataFrame, retain_logprobs_conf: bool) -> np.ndarray | None:
    """
    Return logprobs-derived confidences to overwrite LogMap,
    or None to keep LogMap confidence.
    """
    if not retain_logprobs_conf:
        return None

    if COL_CONFIDENCE not in accepted_subset.columns:
        warn(f"retain_logprobs_conf=True but '{COL_CONFIDENCE}' absent from initial alignment; keeping LogMap confidences.")
        return None

    if "Oracle_confidence" not in oracle_accepted.columns:
        warn("retain_logprobs_conf=True but 'Oracle_confidence' absent from oracle predictions; keeping LogMap confidences.")
        return None

    raw = oracle_accepted["Oracle_confidence"].to_numpy()
    return np.where(np.isfinite(raw), raw, DEFAULT_CONFIDENCE_FALLBACK)


def load_local_oracle_verdicts(directory: str | os.PathLike[str]) -> dict:
    """
    Count the verdicts LogMap's ``LocalOracle.loadLocalOraculoLLM`` will load from
    ``directory``, with the Java loader's own rule: every file whose name contains
    ``.csv`` (non-recursive), comment lines (``#``) and lines without a comma skipped, the
    line split on ``,`` and the third field read as a boolean (``true`` case-insensitively
    is an accepted mapping, anything else — including the ``Prediction`` header — a
    rejected one). Returns ``files``, ``true``, ``false`` (a literal ``false``) and
    ``non_boolean`` (header and malformed lines, which the Java loader counts as false).

    The pipeline passes the directory to LogMap with a trailing separator: the Java
    loader concatenates ``base_path + filename``, so a bare directory path silently
    opens nothing, every candidate is rejected and the run still reports success (the
    defect found on 17 Sep 2026); the count here is also the guard against that.
    """
    path = str(directory)
    if not os.path.isdir(path):
        fatal(f"local oracle predictions directory does not exist: {path}", FileNotFoundError)
    counts = {"files": 0, "true": 0, "false": 0, "non_boolean": 0}
    for name in sorted(os.listdir(path)):
        if ".csv" not in name or not os.path.isfile(os.path.join(path, name)):
            continue
        counts["files"] += 1
        with open(os.path.join(path, name), encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                line = raw.rstrip("\r\n")
                if line.startswith("#") or "," not in line:
                    continue
                fields = line.split(",")
                verdict = fields[2].strip().lower() if len(fields) > 2 else ""
                if verdict == "true":
                    counts["true"] += 1
                elif verdict == "false":
                    counts["false"] += 1
                else:
                    counts["non_boolean"] += 1
    return counts


def _kg_refine_in_python(ctx: PipelineContext, oracle_result: OracleResult, retain_logprobs_conf: bool = False) -> None:
    r"""
    Produce refined alignment for KG track in Python.

    Computes: refined = { initial - M_ask } U { m \in M_ask : oracle(m) = True }

    This bypasses LogMap's Java refinement: a quick approximation of the refined
    alignment that lacks final conflict resolution.

    Setting retain_logprobs_conf = True overwrites the LogMap confidence with the
    LogProbs confidence.

    TODO: handle both pipe and tab separators, or use the same format detection
    logic as evaluate.py.
    """
    # load initial alignment
    initial_path = ctx.run_paths.logmap_mappings()
    initial_df = pd.read_csv(initial_path, sep=PAIRS_SEPARATOR, header=None)
    initial_df.columns = [
        COL_SOURCE_ENTITY_URI,
        COL_TARGET_ENTITY_URI,
        COL_RELATION,
        COL_CONFIDENCE,
        COL_ENTITY_TYPE
    ][:len(initial_df.columns)]

    # build set of m_ask pairs
    preds = oracle_result.predictions
    m_ask_pairs = set()

    # reference the URI columns by name when available (robust to any stray index
    # column); fall back to positional order otherwise.
    _src_col = COL_SOURCE_ENTITY_URI if COL_SOURCE_ENTITY_URI in preds.columns else preds.columns[0]
    _tgt_col = COL_TARGET_ENTITY_URI if COL_TARGET_ENTITY_URI in preds.columns else preds.columns[1]

    for _, row in preds.iterrows():
        m_ask_pairs.add((str(row[_src_col]), str(row[_tgt_col])))

    # retain rows disjoint from m_ask (ie. the initial mapping)
    keep_mask = initial_df.apply(
        lambda row: (str(row.iloc[0]), str(row.iloc[1])) not in m_ask_pairs,
        axis=1
    )
    retained = initial_df[keep_mask]

    # m_ask entries where oracle predicted true
    oracle_accepted = filter_accepted_predictions(preds)

    # take the first N columns matching the initial alignment format (combine)
    ncols = len(initial_df.columns)
    accepted_subset = oracle_accepted.iloc[:, :ncols].copy()
    accepted_subset.columns = initial_df.columns

    # logprob confidence override:
    resolved = _resolve_accepted_confidences(
        accepted_subset,
        oracle_accepted, retain_logprobs_conf
    )
    if resolved is not None:
        accepted_subset[COL_CONFIDENCE] = resolved

    refined = pd.concat([retained, accepted_subset], ignore_index=True)

    # write to the refined output directory (atomically: a crash mid-write must not leave
    # a truncated TSV that validate_run_artifacts and the evaluation harness would accept)
    output_path = ctx.run_paths.refined_mappings_tsv()
    os.makedirs(ctx.run_paths.refined_dir, exist_ok=True)
    atomic_write_text_strict(
        output_path, lambda fp: refined.to_csv(fp, sep='\t', header=False, index=False)
    )
    conf_source = "logprobs" if resolved is not None else "logmap"

    step(f"[Step 4] KG refined alignment: {len(refined)} mappings "
        f"({len(retained)} retained + {len(accepted_subset)} oracle-accepted, "
        f"accepted-subset confidence source: {conf_source})")


###
# END: PRIVATE HELPERS; START: PIPELINE FNS
# align -> prompt_build -> consult_oracle -> refine_alignment -> evaluate
###


def align(ctx: PipelineContext) -> AlignmentResult:
    """
    Step 1: Pass the neccesary files to LogMap to perform an initial alignment and get M_ask.
    `import bridging as br` allows for mapping LogMap IO between java and python (and vice versa)
    """
    import logmap_llm.bridging as br

    step("[Step 1] Align ontologies and obtain mappings to ask an Oracle")

    match ctx.cfg.pipeline.align_ontologies:

        case AlignMode.ALIGN:

            step("[Step 1] Performing fresh initial LogMap alignment")
            if ctx.logmap is None:
                fatal("Fresh alignment requires an initialized LogMap JVM")
            logmap = ctx.logmap
            logmap.perform_alignment()
            return AlignmentResult(
                m_ask_df=br.java_mappings_2_python(
                    logmap.get_mappings_for_llm(),
                    dedupe_uri_pairs=True,
                ),
                mappings=br.java_mappings_2_python(
                    logmap.get_mappings()
                ),
            )

        case AlignMode.REUSE:

            step(f"[Step 1] Loading mappings from file: {ctx.run_paths.logmap_m_ask()}")
            try:
                mappings = pd.read_csv(
                    ctx.run_paths.logmap_mappings(),
                    sep=PAIRS_SEPARATOR,
                    header=None,
                )
            except pd.errors.EmptyDataError:
                mappings = pd.DataFrame(columns=range(5))
            return AlignmentResult(
                m_ask_df=br.load_m_ask_from_file(
                    ctx.run_paths.logmap_m_ask()
                ),
                mappings=mappings,
            )

        case AlignMode.EXTERNAL:

            # Annotate mode: the external file is the M_ask (and stands in for the initial
            # alignment); no LogMap step. See pipeline/annotate.py.
            from logmap_llm.pipeline.annotate import load_external_mappings, write_external_m_ask

            external_path = ctx.cfg.alignmentTask.external_mappings_filepath
            step(f"[Step 1] Loading external mappings as M_ask from: {external_path}")
            if not external_path or not os.path.isfile(external_path):
                fatal(f"external mappings file not found: {external_path}", FileNotFoundError)
            rows = load_external_mappings(external_path)
            if not rows:
                fatal(f"external mappings file holds no mapping: {external_path}")
            m_ask_df = write_external_m_ask(ctx.run_paths, rows)
            step(f"[Step 1] External M_ask published to: {ctx.run_paths.logmap_m_ask()} "
                 f"({len(m_ask_df)} unique of {len(rows)} rows)")
            return AlignmentResult(m_ask_df=m_ask_df, mappings=m_ask_df.copy())

        case AlignMode.BYPASS:
            warn("Bypassing initial LogMap alignment")
            return AlignmentResult()

        case _:
            fatal(f"config: align_ontologies param not recognised: {ctx.cfg.pipeline.align_ontologies}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.align_ontologies}")


def prompt_build(ctx: PipelineContext, initial_alignment: AlignmentResult) -> PromptBuildResult:
    """
    Step 2: Build user prompts for oracle consultation.
    In BUILD mode, dispatches to pipeline/stage_two.py as a subprocess (isolates JVMs & owlready2).
    """
    step("\n[Step 2] Build user prompts for oracle consultation")

    match ctx.cfg.pipeline.build_oracle_prompts:

        case PromptBuildMode.BUILD:

            step("[Step 2] Building fresh oracle user prompts via subprocess")

            if not ctx.config_path:
                fatal("Stage 2 in BUILD mode without a valid config path.") # @raises

            prompts_path = ctx.run_paths.prompts_json()
            prompts_path.unlink(missing_ok=True)

            cmd = [
                sys.executable, "-m", "logmap_llm.pipeline.stage_two",
                "--config", ctx.config_path,
                *_subprocess_context_args(ctx),
            ]

            proc = subprocess.run(cmd, capture_output=False)
            if proc.returncode != 0:
                fatal(f"Stage 2 subprocess failed with return code {proc.returncode}") # @raises

            # read the prompts back from the JSON file
            if not prompts_path.exists():
                if initial_alignment.n_m_ask == 0:
                    atomic_json_write_strict(prompts_path, {})
                    if ctx.cfg.few_shot.few_shot_k > 0:
                        atomic_json_write_strict(ctx.run_paths.few_shot_json(), {})
                else:
                    fatal(
                        f"Stage 2 succeeded but did not publish required prompts: {prompts_path}",
                        FileNotFoundError,
                    )

            with open(prompts_path) as fp:
                prompts = json.load(fp)
            if not isinstance(prompts, dict):
                fatal(f"Prompts artifact must contain a JSON object: {prompts_path}")
            if initial_alignment.n_m_ask > 0 and not prompts:
                fatal(
                    "Stage 2 produced zero prompts for a non-empty M_ask; refusing to "
                    "continue with an undeclared baseline fallback."
                )

            bidirectional = _detect_bidirectional(prompts)
            return PromptBuildResult(
                prompts=prompts,
                bidirectional=bidirectional,
            )

        case PromptBuildMode.REUSE:

            step(f"[Step 2] Loading LLM oracle user prompts from: {ctx.run_paths.prompts_json()}")

            with open(ctx.run_paths.prompts_json()) as fp:
                prompts = json.load(fp)
            if not isinstance(prompts, dict):
                fatal(
                    f"Prompts artifact must contain a JSON object: {ctx.run_paths.prompts_json()}"
                )
            if initial_alignment.n_m_ask > 0 and not prompts:
                fatal("Cannot reuse an empty prompts artifact for a non-empty M_ask")

            bidirectional = _detect_bidirectional(prompts)

            _validate_prompt_keys(
                prompts, bidirectional,
                template_name=ctx.cfg.prompts.cls_usr_prompt_template_name,
            )
            return PromptBuildResult(
                prompts=prompts,
                bidirectional=bidirectional,
            )

        case PromptBuildMode.BYPASS:

            warn("Bypassing use of LLM oracle user prompts")
            return PromptBuildResult()

        case _:
            fatal(f"config: build_oracle_prompts param not recognised: {ctx.cfg.pipeline.build_oracle_prompts}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.build_oracle_prompts}")


def _developer_prompts(cfg) -> tuple[str, dict | None]:
    """The class developer prompt and, when property/instance user templates are configured,
    the per-entity-type map that routes OPROP/DPROP/INST consultations to their own."""
    import logmap_llm.oracle.prompts.developer as dp

    ###
    # CLS PROMPT
    ###

    developer_prompt_map = {}

    cls_dev_prompt_text = dp.get_developer_prompt(
        name=cfg.prompts.cls_dev_prompt_template_name,
        answer_format=cfg.oracle.answer_format,
        response_mode=cfg.oracle.response_mode,
    )

    ###
    # PROPERTY PROMPT: templates will morph to accomodate both data and object properties
    ###

    prop_dev_prompt_text = None
    if (
        cfg.prompts.prop_usr_prompt_template_name
        or cfg.prompts.dprop_usr_prompt_template_name
    ):
        prop_dev_prompt_text = dp.get_developer_prompt(
            cfg.prompts.prop_dev_prompt_template_name,
            answer_format=cfg.oracle.answer_format,
            response_mode=cfg.oracle.response_mode,
        )
        if cfg.prompts.prop_usr_prompt_template_name:
            developer_prompt_map["OPROP"] = prop_dev_prompt_text
        developer_prompt_map["DPROP"] = prop_dev_prompt_text

    ###
    # INSTANCE PROMPT
    ###

    inst_dev_prompt_text = None
    if cfg.prompts.inst_usr_prompt_template_name:
        inst_dev_prompt_text = dp.get_developer_prompt(
            cfg.prompts.inst_dev_prompt_template_name,
            answer_format=cfg.oracle.answer_format,
            response_mode=cfg.oracle.response_mode,
        )
        developer_prompt_map["INST"] = inst_dev_prompt_text

    if len(developer_prompt_map.keys()) == 0:
        developer_prompt_map = None

    return cls_dev_prompt_text, developer_prompt_map


def _consult(oracle_cfg, prompts: dict, candidates_df: pd.DataFrame, bidirectional: bool,
             developer_prompt_text: str, developer_prompt_map: dict | None,
             few_shot_examples=None) -> pd.DataFrame | None:
    """Dispatch one consultation campaign; None when it aborted under the failure tolerance."""
    import logmap_llm.oracle.consultation as oc

    kwargs = dict(
        m_ask_prompts=prompts,
        m_ask_init_alignment_df=candidates_df,
        oracle_cfg=oracle_cfg,
        developer_prompt_text=developer_prompt_text,
        developer_prompt_map=developer_prompt_map,
        few_shot_examples=few_shot_examples,
        **oracle_cfg.consult_kwargs,
    )
    if bidirectional:
        return oc.consult_oracle_bidirectional(**kwargs)
    return oc.consult_oracle_for_mappings_to_ask(**kwargs)


def select_model(ctx: PipelineContext, prompt_build_result: PromptBuildResult) -> ModelSelectionResult:
    """
    Step 2b (only when [model_selection].automatic = true): ask every permitted model
    configuration the anchor-derived questions stage two rendered, rank the candidates by the
    number answered correctly, and make the best one the run's oracle from here on.
    Self-supervised: LogMap's anchors are the labels; no reference alignment is read.
    """
    from logmap_llm.pipeline.model_selection import (
        load_ranking_artifact, rank_candidates, score_candidate,
    )
    from logmap_llm.pipeline.reporting import classify_endpoint

    step("\n[Step 2b] Automatic model selection")

    selection_path = ctx.run_paths.model_selection_json()
    selection_path.unlink(missing_ok=True)

    if prompt_build_result.n_prompts == 0:
        warning("[Step 2b] M_ask is empty, so there is nothing to consult; selection skipped")
        atomic_json_write_strict(
            selection_path, {"schema": 1, "status": "skipped", "reason": "empty M_ask"}, indent=2,
        )
        return ModelSelectionResult()

    ranking_path = ctx.run_paths.model_ranking_json()
    if not ranking_path.is_file():
        fatal(f"automatic model selection requires the ranking prompts artifact: {ranking_path}",
              FileNotFoundError)
    ranking_df, prompts = load_ranking_artifact(ranking_path)
    if ranking_df.empty:
        fatal("automatic model selection: stage two rendered no anchor question", RuntimeError)

    candidates = ctx.cfg.candidate_oracle_configs()
    cls_dev_prompt_text, developer_prompt_map = _developer_prompts(ctx.cfg)
    bidirectional = _detect_bidirectional(prompts)
    records = []

    for index, candidate in enumerate(candidates):
        step(f"[Step 2b] Candidate {index + 1}/{len(candidates)}: {candidate.model_name} "
             f"({classify_endpoint(candidate.base_url)}), {len(ranking_df)} questions")
        predictions = _consult(
            candidate, prompts, ranking_df, bidirectional, cls_dev_prompt_text, developer_prompt_map,
        )
        record = score_candidate(index, candidate, ranking_df, predictions)
        if predictions is None:
            warning(f"[Step 2b] {candidate.model_name}: consultation aborted; ranked last")
        step(f"[Step 2b] {candidate.model_name}: {record['correct']}/{record['asked']} correct, "
             f"{record['errors']} unanswered")
        records.append(record)

    ranking = rank_candidates(records)
    selected = ranking[0]
    # every later phase (consultation, reporting) reads the winner from the shared config
    ctx.cfg = ctx.cfg.model_copy(update={"oracle": candidates[selected["index"]]})
    atomic_json_write_strict(
        selection_path,
        {"schema": 1, "status": "selected", "questions": len(ranking_df),
         "ranking": ranking, "selected": selected},
        indent=2,
    )
    success(f"[Step 2b] Selected {selected['model_name']} "
            f"({selected['correct']}/{selected['asked']} correct)")
    return ModelSelectionResult(questions=len(ranking_df), ranking=ranking, selected=selected)


def consult_oracle(ctx: PipelineContext, initial_alignment: AlignmentResult, prompt_build_result: PromptBuildResult) -> OracleResult:
    """
    Step 3: Consult Oracle for mappings to ask.
    """
    step("\n[Step 3] Consult Oracle for mappings to ask")

    cls_dev_prompt_text, developer_prompt_map = _developer_prompts(ctx.cfg)

    ###
    # SWITCH ON CONSULT MODE (SPECIFIED IN CONFIG)
    ###

    match ctx.cfg.pipeline.consult_oracle:

        case ConsultMode.CONSULT:

            if prompt_build_result.prompts is None or len(prompt_build_result.prompts) == 0:
                if initial_alignment.n_m_ask > 0:
                    fatal(
                        "No prompts are available for a non-empty M_ask; refusing to "
                        "silently treat the condition as a zero-consultation baseline."
                    )
                warn("[Step 3] M_ask is empty — publishing an explicit empty predictions artifact")
                empty_predictions = initial_alignment.m_ask_df.copy()
                empty_predictions["Oracle_prediction"] = pd.Series(dtype="object")
                empty_predictions["Oracle_confidence"] = pd.Series(dtype="float64")
                empty_predictions["Oracle_input_tokens"] = pd.Series(dtype="Int64")
                empty_predictions["Oracle_output_tokens"] = pd.Series(dtype="Int64")
                atomic_write_text_strict(
                    ctx.run_paths.predictions_csv(),
                    lambda fp: empty_predictions.to_csv(fp, na_rep="nan", index=False),
                )
                return OracleResult(
                    predictions=empty_predictions,
                    oracle_params=_configured_oracle_params(ctx.cfg.oracle),
                )

            # else: check few-shot configuration
            few_shot_examples = None
            if ctx.cfg.few_shot.few_shot_k > 0:
                few_shot_fp = ctx.run_paths.few_shot_json()

                if os.path.isfile(few_shot_fp):
                    few_shot_examples = _load_few_shot_artifact(
                        few_shot_fp,
                        few_shot_cfg=ctx.cfg.few_shot,
                        oracle_cfg=ctx.cfg.oracle,
                        expected_query_keys=prompt_build_result.prompts,
                    )
                    # stage_two writes a dict {m_ask_key -> [[user, assistant], ...]} for
                    # query-specific few-shot; legacy static few-shot is a flat list of pairs.
                    # A dict is dispatched per-consultation; a list is the shared frozen prefix.
                    if isinstance(few_shot_examples, dict):
                        success(f"Loaded query-specific few-shot for {len(few_shot_examples)} M_ask keys from {few_shot_fp}")
                    else:
                        success(f"Loaded {len(few_shot_examples)} (shared) few-shot examples from {few_shot_fp}")

                else: # failed to find few-shot file
                    fatal(
                        f"few_shot_k={ctx.cfg.few_shot.few_shot_k} but the required "
                        f"few-shot artifact is missing: {few_shot_fp}",
                        FileNotFoundError,
                    )

            # END: FEW-SHOT-HANDLER

            ###
            # EXECUTE CONSULTATION
            ###

            ctx.run_paths.predictions_csv().unlink(missing_ok=True)

            mode = " (bidirectional)" if prompt_build_result.bidirectional else ""
            step(f"[Step 3] Consulting LLM oracle{mode} with model: {ctx.cfg.oracle.model_name}")
            oracle_predictions_df = _consult(
                ctx.cfg.oracle, prompt_build_result.prompts, initial_alignment.m_ask_df,
                prompt_build_result.bidirectional, cls_dev_prompt_text, developer_prompt_map,
                few_shot_examples,
            )

            if oracle_predictions_df is None:
                fatal(
                    "Oracle consultation aborted before producing predictions; no partial "
                    "condition will be reported as successful.",
                    RuntimeError,
                )

            capabilities = oracle_predictions_df.attrs.get("oracle_capabilities", {})

            oracle_result = OracleResult(
                predictions=oracle_predictions_df,
                oracle_params=_configured_oracle_params(ctx.cfg.oracle, capabilities),
                bidirectional=prompt_build_result.bidirectional,
            )
            atomic_write_text_strict(
                ctx.run_paths.predictions_csv(),
                lambda fp: oracle_result.predictions.to_csv(fp, na_rep="nan", index=False),
            ) # index=False so a REUSE re-read does not gain a phantom 'Unnamed: 0' column
              # that shifts every positional column right by one. Atomic so a crash
              # mid-write cannot leave a truncated CSV a later --reuse step silently
              # accepts as complete.
            return oracle_result

        case ConsultMode.REUSE:

            step(f"[Step 3] Loading LLM oracle predictions from: {ctx.run_paths.predictions_csv()}")
            predictions_df = pd.read_csv(ctx.run_paths.predictions_csv())
            # belt-and-braces: drop any legacy 'Unnamed:*' index column so positional
            # access stays aligned.
            predictions_df = predictions_df.loc[:, ~predictions_df.columns.str.startswith("Unnamed:")]
            predictions_df = normalise_prediction_column(predictions_df) # pred.str_T/F/Y/N -> bool
            return OracleResult(
                predictions=predictions_df,
                oracle_params=_configured_oracle_params(ctx.cfg.oracle),
                # decomposition columns are written only by bidirectional runs,
                # so their presence identifies one.
                bidirectional="Oracle_fwd_prediction" in predictions_df.columns,
            )

        case ConsultMode.LOCAL:

            # LogMap's LocalOracle opens base_path + filename, so the directory must end
            # with the separator; and a directory from which it would load no verdict is
            # a configuration error, not a run in which the oracle rejected everything.
            local_dir = os.path.join(str(ctx.cfg.oracle.local_oracle_predictions_dirpath), "")
            step(f"[Step 3] Local oracle predictions from: {local_dir}")
            counts = load_local_oracle_verdicts(local_dir)
            step(
                f"[Step 3] local oracle: {counts['true']} true / {counts['false']} false verdicts "
                f"from {counts['files']} CSV file(s) ({counts['non_boolean']} header/other line(s))"
            )
            if counts["true"] + counts["false"] == 0:
                fatal(
                    f"local oracle loaded no verdict from {local_dir}: it needs CSV files with "
                    "Source,Target,Prediction[,Confidence] rows (LogMap's LocalOracle format)",
                    ValueError,
                )
            if counts["true"] == 0:
                warning("[Step 3] local oracle holds no accepted (true) mapping; every M_ask candidate will be rejected")
            return OracleResult(
                local_dir=local_dir,
                predictions=None,
                local_oracle_verdicts=counts,
            )

        case ConsultMode.BYPASS:

            warn("Bypassing oracle consultations")
            return OracleResult()

        case _:
            fatal(f"config: consult_oracle param not recognised: {ctx.cfg.pipeline.consult_oracle}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.consult_oracle}")


def refine_alignment(ctx: PipelineContext, oracle_result: OracleResult) -> RefinementResult:
    """
    Step 4: Refine alignment using oracle mapping predictions.
    """
    import logmap_llm.bridging as br

    step("\n[Step 4] Refine alignment using oracle mapping predictions")

    match ctx.cfg.pipeline.refine_alignment:

        case RefineMode.REFINE:

            # unlink any refined TSV left by a previous run of the same task, so a
            # stale artifact cannot be evaluated and sealed as this run's output
            ctx.run_paths.refined_mappings_tsv().unlink(missing_ok=True)

            if not oracle_result.has_predictions and not oracle_result.is_local:

                warn("[Step 4] No oracle predictions — skipping refinement (initial alignment is final)")

                import shutil

                # moves the initial mappings to the refined dir
                src = ctx.run_paths.logmap_mappings_tsv()
                dst = ctx.run_paths.refined_mappings_tsv()

                if not src.exists():
                    fatal(
                        f"[Step 4] Initial alignment TSV is missing: {src}; cannot "
                        "publish a refined alignment."
                    )
                os.makedirs(ctx.run_paths.refined_dir, exist_ok=True)
                shutil.copy2(str(src), str(dst))
                step(f"[Step 4] Initial alignment copied to refined dir: {dst.name}")

                return RefinementResult()

            # KG track: unpatched LogMap builds crash with NPEs on instance mappings
            # during Java refinement. The LogMap shipped with this repository refines
            # the KG track fine (and resolves logical conflicts, which is recommended),
            # but users on a custom LogMap build may lack those fixes, so we also
            # include the PYTHON_SET_UNION bypass (see below):

            if (ctx.cfg.pipeline.refinement_strategy == RefinementStrategy.PYTHON_SETUNION and oracle_result.has_predictions):

                step("[Step 4] Python-based refinement (set-union bypass)")
                _kg_refine_in_python(ctx, oracle_result)
                return RefinementResult()

            if oracle_result.has_predictions:
                if ctx.logmap is None:
                    fatal("LogMap refinement requires an initialized LogMap JVM")
                logmap = ctx.logmap
                logmap.set_output_dir(ctx.run_paths.refined_dir)

                step("[Step 4] Refining initial LogMap alignment with LLM Oracle predictions")
                preds_java = br.python_oracle_mapping_predictions_2_java(
                    oracle_result.predictions
                )
                step(f"[Step 4] Number of mappings predicted True by Oracle given to LogMap: {len(preds_java)}")
                logmap.refine_alignment(preds_java)
                mappings_java = logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            elif oracle_result.is_local:
                if ctx.logmap is None:
                    fatal("Local-oracle refinement requires an initialized LogMap JVM")
                logmap = ctx.logmap
                logmap.set_output_dir(ctx.run_paths.refined_dir)

                step("[Step 4] Refining initial LogMap alignment with local Oracle predictions")
                # trailing separator: LocalOracle.loadLocalOraculoLLM concatenates base_path + filename
                logmap.refine_alignment(os.path.join(str(oracle_result.local_dir), ""))
                mappings_java = logmap.get_mappings()
                step(f"[Step 4] Number of mappings in LogMap refined alignment: {len(mappings_java)}")
                return RefinementResult(
                    refined_mappings=br.java_mappings_2_python(mappings_java)
                )

            else:
                critical("Check your config to ensure 'refine_alignment' is set appropriately!")
                critical("The oracle response is empty and no local predictions file is specified.")
                fatal(
                    "Oracle payload passed to `refine_alignment` is either "
                    "malformed or is empty.",
                    IOError,
                )

        case RefineMode.BYPASS:

            warn("Bypassing alignment refinement")
            return RefinementResult()

        case _:
            fatal(f"config: refine_alignment param not recognised: {ctx.cfg.pipeline.refine_alignment}")

    fatal(f"unable to match on: {ctx.cfg.pipeline.refine_alignment}")


def evaluate(ctx) -> EvaluationResult:
    """
    Step 5: Evaluate the refined alignment.
    Dispatches to logmap_llm.evaluation.harness as a subprocess for JVM isolation:
    the default DeepOnto evaluator needs its own JVM, as do most alternative engines
    (eg. MELT, HOBBIT, SEALS), though CustomEvaluationEngine
    (logmap_llm.evaluation.engines.custom) does not.
    Evaluation can be skipped by setting 'evaluate = false' under '[evaluation]' in config.toml
    """
    step("\n[Step 5] Evaluation")

    if not ctx.cfg.evaluation.evaluate:
        warn("Evaluation disabled in config")
        return EvaluationResult()

    if not ctx.config_path:
        warning("No config path available for evaluation subprocess")
        return EvaluationResult(subprocess_failed=True)

    cmd = [
        sys.executable, "-m", "logmap_llm.evaluation.harness",
        "--config", ctx.config_path,
        *_subprocess_context_args(ctx),
    ]

    step("[Step 5] Running evaluation subprocess")
    ctx.run_paths.eval_json().unlink(missing_ok=True)

    proc = subprocess.run(cmd, capture_output=False)

    subprocess_failed = proc.returncode != 0

    if subprocess_failed:
        warning(f"Evaluation subprocess exited with code {proc.returncode}")

    eval_results_path = ctx.run_paths.eval_json()

    # only trust the results file if the subprocess succeeded
    if not subprocess_failed and eval_results_path.exists():
        with open(eval_results_path) as fp:
            results_dict = json.load(fp)

        success(f"Evaluation results loaded from: {eval_results_path}")
        return EvaluationResult(results=results_dict)

    if not subprocess_failed:
        warning(
            f"Evaluation subprocess succeeded but required result is missing: {eval_results_path}"
        )
        subprocess_failed = True

    # A stale result is never trusted after a failed or incomplete child.
    return EvaluationResult(subprocess_failed=subprocess_failed)
