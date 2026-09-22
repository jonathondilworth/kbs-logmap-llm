'''
logmap_llm.pipeline.stage_two
Designated subprocess responsible for prompt construction — essentially a
stand-alone 'build prompts' script. Config loading and filepath resolution are
shared with runner.py via subprocess_bootstrap (utils/subprocess.py) and
PipelinePaths, so both processes resolve artifacts identically.
'''
import os
import sys
import pandas as pd

from functools import partial

import logmap_llm.oracle.prompts.templates as opb

from logmap_llm.utils.subprocess import subprocess_bootstrap
from logmap_llm.ontology.access import load_ontologies
from logmap_llm.oracle.prompts.context import PromptContext
from logmap_llm.constants import (
    EntityType,
    PromptBuildMode,
    M_ASK_COLUMNS,
    PAIRS_SEPARATOR,
    DEFAULT_MAX_SIBLING_CANDIDATES,
    DEFAULT_OWLREADY2_CACHE_DIR,
)
from logmap_llm.ontology.sibling_retrieval import SiblingSelector
from logmap_llm.ontology.sibling_strategy import (
    SiblingSelectionStrategy,
    resolve_sibling_strategy,
)
from logmap_llm.utils.logging import (
    warning,
    warn,
    success,
    step,
)
from logmap_llm.utils.io import atomic_json_write_strict as _atomic_write_json
from logmap_llm.config.schema import (
    PromptTemplateConfig,
)


###
# START: PRIVATE HELPERS
###

def _copy_and_coerce_tuple_to_list(xs: tuple) -> list:
    return list(xs)


def _get_m_ask_column_names() -> list[str]:
    return _copy_and_coerce_tuple_to_list(M_ASK_COLUMNS)


def _rag_artifact_path(run_paths, filename: str) -> str:
    return os.path.join(os.path.dirname(os.fspath(run_paths.few_shot_json())), filename)


def _remove_artifact(run_paths, filename: str) -> None:
    try:
        os.unlink(_rag_artifact_path(run_paths, filename))
    except FileNotFoundError:
        pass


def _remove_stale_rag_fallback(run_paths) -> None:
    """Clear both fallback records.

    A zero-shot retry in an existing attempt directory must not inherit either one
    from an earlier few-shot attempt; used by the empty-stage and k=0 paths.
    """
    _remove_artifact(run_paths, "rag_fallback.json")
    _remove_artifact(run_paths, "rag_negative_fallback.json")


def _publish_empty_prompt_artifacts(run_paths) -> None:
    """Make an empty stage explicit so a retry cannot consume old prompts/examples."""
    _atomic_write_json(run_paths.prompts_json(), {})
    _atomic_write_json(run_paths.few_shot_json(), {})
    _atomic_write_json(_rag_artifact_path(run_paths, "rag_traces.json"), {})
    _remove_stale_rag_fallback(run_paths)


def _handle_rag_generation_failure(run_paths, few_shot_cfg, exc: Exception) -> None:
    """Clear stale examples, then either fail strictly or record zero-shot degradation."""
    _atomic_write_json(run_paths.few_shot_json(), {})
    _atomic_write_json(_rag_artifact_path(run_paths, "rag_traces.json"), {})

    policy = getattr(few_shot_cfg, "rag_failure_policy", "error")
    fallback_path = _rag_artifact_path(run_paths, "rag_fallback.json")
    if policy == "error":
        try:
            os.unlink(fallback_path)
        except FileNotFoundError:
            pass
        raise RuntimeError(
            "Query-specific RAG generation failed under strict rag_failure_policy='error'"
        ) from exc
    if policy != "record_zero_shot":
        raise ValueError(
            "few_shot.rag_failure_policy must be 'error' or 'record_zero_shot'"
        ) from exc

    _atomic_write_json(
        fallback_path,
        {
            "requested_few_shot_k": getattr(few_shot_cfg, "few_shot_k", None),
            "requested_negative_strategy": getattr(
                few_shot_cfg, "few_shot_negative_strategy", None
            ),
            "requested_encoder_kind": getattr(few_shot_cfg, "rag_encoder_kind", None),
            "effective_mode": "zero-shot",
            "reason_type": type(exc).__name__,
            "reason": str(exc),
        },
        indent=2,
    )


def _publish_negative_construction_fallbacks(run_paths, few_shot_cfg, traces: dict) -> bool:
    """Publish per-lane negative-construction fallbacks, separately from rag_fallback.json.

    The two must never share an artefact: `rag_fallback.json` means the retriever could
    not run the requested mode or k, and `run._is_degraded` raises BatchRunError the
    moment it exists under any policy other than record_zero_shot. A negative-construction
    fallback is not that — the block is complete and correctly paired, only the branch's
    primary target rule was unavailable for some negative (e.g. undeclared properties,
    instances with no typed peers, classes with no siblings). It is a reportable rate,
    written per lane and per reason, not a degradation.
    """
    affected = {
        key: trace.get("negative_fallback_reason")
        for key, trace in traces.items()
        if trace.get("negative_fallback_reason")
    }
    path = _rag_artifact_path(run_paths, "rag_negative_fallback.json")
    if not affected:
        _remove_artifact(run_paths, "rag_negative_fallback.json")
        return False

    by_lane: dict = {}
    by_reason: dict = {}
    for key, reason in affected.items():
        lane = str(traces[key].get("entity_type", "UNKNOWN"))
        lane_record = by_lane.setdefault(lane, {"queries": 0, "reasons": {}})
        lane_record["queries"] += 1
        for part in str(reason).split("; "):
            lane_record["reasons"][part] = lane_record["reasons"].get(part, 0) + 1
            by_reason[part] = by_reason.get(part, 0) + 1

    lane_totals: dict = {}
    for trace in traces.values():
        lane = str(trace.get("entity_type", "UNKNOWN"))
        lane_totals[lane] = lane_totals.get(lane, 0) + 1

    _atomic_write_json(
        path,
        {
            "schema": 1,
            "kind": "rag-negative-construction-fallback",
            "note": (
                "Per-query fallback from a negative-construction layout's primary target "
                "rule. The demonstration block is complete and correctly paired; this is a "
                "reportable rate, not a degradation."
            ),
            "negative_layout": next(
                (t.get("negative_layout") for t in traces.values() if t.get("negative_layout")),
                None,
            ),
            "queries_with_fallback": len(affected),
            "queries_total": len(traces),
            "by_reason": dict(sorted(by_reason.items())),
            "by_lane": {
                lane: {
                    **record,
                    "queries_total": lane_totals.get(lane, 0),
                    "reasons": dict(sorted(record["reasons"].items())),
                }
                for lane, record in sorted(by_lane.items())
            },
        },
        indent=2,
    )
    return True


def _publish_recorded_trace_fallbacks(run_paths, few_shot_cfg, traces: dict) -> bool:
    """Publish query-level degradations already recorded by the retriever."""
    affected = {
        key: {
            "effective_mode": trace.get("effective_mode"),
            "effective_k": trace.get("effective_k"),
            "fallback_reason": trace.get("fallback_reason"),
        }
        for key, trace in traces.items()
        if trace.get("fallback_reason")
    }
    if not affected:
        _remove_artifact(run_paths, "rag_fallback.json")
        return False

    _atomic_write_json(
        _rag_artifact_path(run_paths, "rag_fallback.json"),
        {
            "requested_few_shot_k": getattr(few_shot_cfg, "few_shot_k", None),
            "requested_negative_strategy": getattr(
                few_shot_cfg, "few_shot_negative_strategy", None
            ),
            "requested_encoder_kind": getattr(few_shot_cfg, "rag_encoder_kind", None),
            "effective_mode": "query-level-degradation",
            "affected_queries": affected,
        },
        indent=2,
    )
    return True


#: Module alias for the precedence rule shared with config validation (which cannot
#: import owlready2).
_resolve_sibling_strategy = resolve_sibling_strategy


def _build_sibling_selector(
    prompts_cfg: PromptTemplateConfig, ontology_domain: str | None = None,
) -> SiblingSelector | None:
    try:
        strategy = _resolve_sibling_strategy(
            prompts_cfg.sibling_strategy,
            ontology_domain,
        )
        model_override = prompts_cfg.sibling_model
        max_cands = prompts_cfg.sibling_max_candidates

        if max_cands is None:
            max_cands = DEFAULT_MAX_SIBLING_CANDIDATES

        step(f'[STEP TWO] Initialising SiblingSelector (strategy={strategy.value}, max_candidates={max_cands}).', important=True)

        sel = SiblingSelector(
            strategy=strategy,
            device=prompts_cfg.sibling_encoder_device,
            model_name_or_path=model_override,
            max_candidates=max_cands,
            model_revision=prompts_cfg.sibling_model_revision,
        )
        success(f'SiblingSelector ready (device: {sel.device})')
        return sel

    except Exception as outer_e:
        # A missing encoder checkpoint must fail loudly: silently falling back to
        # alphanumeric sibling selection would run a different experiment under the
        # name of the intended one.
        requested = prompts_cfg.sibling_strategy
        raise RuntimeError(
            f'SiblingSelector initialisation FAILED: {outer_e}\n'
            f'  requested strategy : {requested or "(auto by ontology_domain)"}\n'
            f'  resolved strategy  : {_resolve_sibling_strategy(requested, ontology_domain).value}\n'
            f'  ontology_domain    : {ontology_domain!r}\n'
            'Refusing to silently fall back to ALPHANUMERIC sibling selection: that would run a '
            'DIFFERENT experiment under the name of the intended one. Install the embedding model '
            'for the configured strategy (the CLS-pooled checkpoint, or all-MiniLM-L12-v2 for '
            'sbert), or pass --sibling-strategy '
            'alphanumeric explicitly if that is genuinely what you want.'
        ) from outer_e

###
# END: PRIVATE HELPERS
###


def main():

    ###
    # BOOTSTRAP (THIS SUBPROCESS)
    #############################

    # This must run before the JVM starts and before any ontology is loaded. Under the
    # "fork" start method owlready2 parses any ontology >= 8 MB in a fork()ed child, and
    # this subprocess also starts a JVM via jpype (for bridging.py) — fork() from that
    # multi-threaded process can inherit a locked malloc/JVM mutex and deadlock, inside
    # ontology/cache.py's exclusive flock, stalling every other run behind it. Forcing
    # "spawn" makes owlready2 parse in-process: the large parse is no longer overlapped
    # with a child process, but the quadstore cache means we pay it once per ontology
    # per node anyway.
    import multiprocessing
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
        multiprocessing.set_start_method("spawn", force=True)

    # CONFIG, PATH & FILESYSTEM MANAGEMENT & LOGGING

    step("[STEP 2] Running PROMPT BUILD as an isolated subprocess ... ")

    from logmap_llm.pipeline.cli import parse_args
    args = parse_args()

    config, run_paths, tee = subprocess_bootstrap("PROMPT_BUILD_STAGE_TWO", args=args)

    config_path = args.config

    step(f"[STEP 2] Configuration loaded from: {config_path}")

    initial_mappings_fp = run_paths.logmap_mappings()
    step(f"[STEP 2] Reading initial alignment mappings from file: {initial_mappings_fp}")
    try:
        mappings = pd.read_csv(initial_mappings_fp, sep=PAIRS_SEPARATOR, header=None)
    except pd.errors.EmptyDataError:
        # An empty (0-byte) initial alignment is data, not a parse failure (matches
        # oracle/rag/bundle._alignment's empty_ok handling of the same format).
        mappings = pd.DataFrame(columns=range(5))
    step(f'[STEP 2] Number of mappings in initial alignment: {len(mappings)}')

    # The dedupe helper is pure Python; Java imports are lazy at the actual
    # conversion boundary, so prompt construction does not need a second JVM.
    from logmap_llm.bridging import dedupe_m_ask_by_uri_pair
    m_ask_fp = run_paths.logmap_m_ask()

    # If LogMap produced no uncertain mappings, exit early — this must happen before
    # pd.read_csv, which raises EmptyDataError on a 0-byte file. Common on well-aligned
    # pairs. Empty M_ask means the LogMapLLM refined alignment is identical to the
    # plain-LogMap initial alignment (there is nothing for the oracle to refine).
    if (not os.path.exists(m_ask_fp)) or os.path.getsize(m_ask_fp) == 0:
        warning("M_ask file is empty — no uncertain mappings to build prompts for.")
        _publish_empty_prompt_artifacts(run_paths)
        success("Published explicit empty prompt and few-shot artifacts.")
        warning("Exiting Stage 2 subprocess (nothing to do).")
        sys.stdout = tee.original_stdout
        tee.close()
        sys.exit(0)

    m_ask_df = pd.read_csv(m_ask_fp, sep=PAIRS_SEPARATOR, header=None)
    m_ask_df.columns = _get_m_ask_column_names()
    # dedupe by (source,target) so a mixed predicate emitted in both property lanes (KG-ABOX 'both'
    # policy) is prompted once, consistent with the consultation path (bridging.load_m_ask_from_file).
    m_ask_df = dedupe_m_ask_by_uri_pair(m_ask_df)
    step(f'[STEP 2] Loading mappings to ask an Oracle from: {m_ask_fp}')

    # The response configuration and ontology domain travel as an immutable
    # PromptContext, constructed once here and passed explicitly to every template.

    ###
    # CONFIGURE RESPONSE FORMAT FOR PROMPTS
    #######################################
    # apply response configuration to prompt templates
    # must happen before any prompt functions are called

    step(f"[STEP 2] Response config set to: answer_format={config.oracle.answer_format}.")
    step(f"[STEP 2] Response config set to: response_mode={config.oracle.response_mode}")
    prompt_context = PromptContext.from_config(config)
    step(f"[STEP 2] Prompt context: {prompt_context.answer_format}/"
         f"{prompt_context.response_mode}, domain={prompt_context.ontology_domain!r}",
         important=True)


    ###
    # CONFIGURE ONTOLOGY DOMAIN QUALIFIER
    #####################################
    # apply ontology domain qualifier to prompt templates
    # must happen before any prompt functions are called

    step(f"[STEP 2] Specifying 'ontology_domain' as '{config.alignmentTask.ontology_domain}' ... ", important=True)


    ###
    # CONFIGURE PROMPT TEMPLATE AND DIRECTION
    #########################################
    # obtain the prompt template name, check whether it is
    # a 'forward only' or a forward+reverse (ie. bidirectional) template

    oupt_name = config.prompts.cls_usr_prompt_template_name
    bidirectional_mode = opb.registry.is_bidirectional(oupt_name)
    step(f"[STEP 2] Using prompt template: {oupt_name} (bidirectional={str(bidirectional_mode)}).", important=True)


    ###
    # CONFIGURE SIBLING CONTEXT (IF REQUIRED)
    #########################################
    # if the selected template requires sibling-based
    # context then initialise the SiblingSelector

    # A selector is needed for two independent reasons: the class template may render sibling
    # context, and paired-sibling-v2 needs siblings to build negatives. Only the first is
    # conditioned on the template, so the second must be asked for separately - the campaign
    # templates (one_level_of_parents_and_synonyms, prop_domain_range,
    # inst_full_context_entropy) declare no sibling requirement at all.
    template_needs_siblings = opb.registry.requires_siblings(oupt_name)
    negatives_need_siblings = (
        config.few_shot.few_shot_k > 0
        and config.few_shot.rag_negative_layout == "paired-sibling-v2"
    )
    sibling_selector = None
    if template_needs_siblings or negatives_need_siblings:
        # _build_sibling_selector refuses to degrade to alphanumeric when a semantic
        # checkpoint cannot be loaded, which is exactly the guarantee the sibling arm needs.
        sibling_selector = _build_sibling_selector(
            config.prompts, config.alignmentTask.ontology_domain,
        )
    if sibling_selector is not None:
        step(f"[STEP 2] SiblingSelector has been initialised (using {sibling_selector.__class__.__name__}; "
             f"template={template_needs_siblings}, negatives={negatives_need_siblings}).")


    ###
    # RESOLVE PROPERTY PROMPT (IF REQUIRED)
    #######################################
    # if the user has specified a property prompt template within the config
    # ie. when `prop_usr_prompt_template_name` is set under [prompts] in config.toml
    # then we should construct property prompts for use within this experimental run

    property_prompt_function = None
    property_prompt_template_name = config.prompts.prop_usr_prompt_template_name

    if property_prompt_template_name is not None:
        resolved_prop_template_entry = opb.registry.get(property_prompt_template_name)

        # TODO: handle DPROP and OPROPs differently in future
        # Bind to the prompt context, never the raw registry callable: every template
        # takes `ctx` as a required keyword-only argument, and an unbound callable raises
        # TypeError inside build_oracle_user_prompts' per-candidate try, which turns it
        # into a warning and drops the candidate — silently emptying the lane.
        if resolved_prop_template_entry.entity_type == EntityType.OBJECTPROPERTY:
            property_prompt_function = opb.get_oracle_user_prompt_template_function(
                property_prompt_template_name, prompt_context)

        if property_prompt_function:
            step(f"[STEP 2] Using PROPERTY PROMPT TEMPLATE: {property_prompt_template_name}", important=True)

        else:
            warning(f"THE PROPERTY PROMPT TEMPLATE '{property_prompt_template_name}' CANNOT BE RESOLVED FROM THE REGISTRY!")
            warning("ARE YOU SURE YOU USED THE CORRECT PROPERTY PROMPT TEMPLATE NAME?!")
            warning("Property mappings will be skipped!")


    ###
    # RESOLVE DATA-PROPERTY PROMPT (OPTIONAL)
    #########################################
    # An optional dedicated DPROP template. When unset, DPROP candidates fall back to the
    # object-property template resolved above. Accepts a template registered as either
    # DATAPROPERTY or OBJECTPROPERTY (both are property-shaped).

    data_property_prompt_function = None
    data_property_prompt_template_name = config.prompts.dprop_usr_prompt_template_name

    if data_property_prompt_template_name is not None:
        resolved_dprop_template_entry = opb.registry.get(data_property_prompt_template_name)

        if resolved_dprop_template_entry.entity_type in (EntityType.DATAPROPERTY, EntityType.OBJECTPROPERTY):
            data_property_prompt_function = opb.get_oracle_user_prompt_template_function(
                data_property_prompt_template_name, prompt_context)

        if data_property_prompt_function:
            step(f"[STEP 2] Using DATA-PROPERTY PROMPT TEMPLATE: {data_property_prompt_template_name}", important=True)
        else:
            warning(f"THE DATA-PROPERTY PROMPT TEMPLATE '{data_property_prompt_template_name}' CANNOT BE RESOLVED FROM THE REGISTRY!")
            warning("DPROP mappings will fall back to the object-property template.")


    ###
    # RESOLVE INSTANCE PROMPT (IF REQUIRED)
    #######################################
    # if the user has specified an instance prompt template within the config
    # ie. when `inst_usr_prompt_template_name` is set under [prompts] in config.toml
    # then we should construct instance prompts for use within this experimental run

    instance_prompt_function = None
    instance_prompt_template_name = config.prompts.inst_usr_prompt_template_name

    if instance_prompt_template_name is not None:
        resolved_inst_template_entry = opb.registry.get(instance_prompt_template_name)

        if resolved_inst_template_entry.entity_type == EntityType.INSTANCE:
            instance_prompt_function = opb.get_oracle_user_prompt_template_function(
                instance_prompt_template_name, prompt_context)

        if instance_prompt_function:
            step(f"[STEP 2] Using INSTANCE PROMPT TEMPLATE: {instance_prompt_template_name}", important=True)

        else:
            warning(f"THE INSTANCE PROMPT TEMPLATE '{instance_prompt_template_name}' CANNOT BE RESOLVED FROM THE REGISTRY!")
            warning("ARE YOU SURE YOU USED THE CORRECT INSTANCE PROMPT TEMPLATE NAME?!")
            warning("Instance mappings will be skipped!")


    ###
    # END: BOOTSTRAP
    ###


    ###
    # START: PROMPT BUILD
    #####################

    print()

    step("[STEP 2] STARTING: Building oracle user prompts ... ")
    if bidirectional_mode:
        step("[STEP 2] (bidirectional mode is set) Constructing two prompts per candidate.")


    ###
    # LOAD ONTOLOGIES
    #################

    step("[STEP 2] Loading ontologies ... (CHECKING IF OWLREADY2 CACHE EXISTS) ")
    if args.no_cache:
        warn("The '--no-cache' flag has been specified, skipping cache check.")

    cache_dir = None if args.no_cache else DEFAULT_OWLREADY2_CACHE_DIR

    OA_source, OA_target = load_ontologies(
        config.alignmentTask.onto_source_filepath,
        config.alignmentTask.onto_target_filepath,
        cache_dir=cache_dir,
        stub_import_iris=config.alignmentTask.stub_import_iris,
        vocabulary=config.alignmentTask.resolved_vocabulary,
    )

    success("LOADED ONTOLOGIES!\n")


    ###
    # PROMPT CONSTRUCTION
    #####################

    # Bidirectional mode applies to CLASS candidates; typed OPROP/DPROP/INST rows are routed
    # through the forward property/instance templates (hybrid lanes).

    if bidirectional_mode:
        step("[STEP 2] CONSTRUCTING BIDIRECTIONAL TEMPLATES.")
        m_ask_oracle_user_prompts, n_equiv_cands, n_non_equiv_skipped = (
            opb.build_oracle_user_prompts_bidirectional(
                oupt_name, config.alignmentTask.onto_source_filepath, config.alignmentTask.onto_target_filepath, m_ask_df,
                OA_source=OA_source, OA_target=OA_target,
                sibling_selector=sibling_selector,
                # hybrid lanes: typed property/instance rows keep their forward templates
                property_prompt_name=property_prompt_template_name,
                instance_prompt_name=instance_prompt_template_name,
                property_prompt_function=property_prompt_function,
                instance_prompt_function=instance_prompt_function,
                data_property_prompt_name=data_property_prompt_template_name,
                data_property_prompt_function=data_property_prompt_function,
                ctx=prompt_context,
            )
        )
        step(f"[STEP 2] (n_prompts={str(len(m_ask_oracle_user_prompts))})")
        step(f"[STEP 2] (n_equiv_cands={str(n_equiv_cands)})")
        step(f"[STEP 2] (n_non_equiv_skipped={str(n_non_equiv_skipped)})")

    else:
        step("[STEP 2] CONSTRUCTING TEMPLATES.")
        m_ask_oracle_user_prompts = opb.build_oracle_user_prompts(
            oupt_name, config.alignmentTask.onto_source_filepath, config.alignmentTask.onto_target_filepath, m_ask_df,
            OA_source=OA_source, OA_target=OA_target,
            sibling_selector=sibling_selector,
            property_prompt_name=property_prompt_template_name,
            instance_prompt_name=instance_prompt_template_name,
            property_prompt_function=property_prompt_function,
            instance_prompt_function=instance_prompt_function,
            data_property_prompt_name=data_property_prompt_template_name,
            data_property_prompt_function=data_property_prompt_function,
            ctx=prompt_context,
        )

    if m_ask_oracle_user_prompts is not None:
        step(f"[STEP 2] Number of LLM Oracle user prompts obtained: {len(m_ask_oracle_user_prompts)}", important=True)


    # NOTE: probably a redundant guard — this script isn't even called unless
    # PromptBuildMode is set to BUILD.


    ###
    # WRITE PROMPTS TO DISK
    #######################

    step(f"[STEP 2] PromptBuildMode is set to: {config.pipeline.build_oracle_prompts}.")

    if config.pipeline.build_oracle_prompts == PromptBuildMode.BUILD:

        step("[STEP 2] Saving prompts to disk ... ")

        prompts_json_fp = run_paths.prompts_json()
        _atomic_write_json(prompts_json_fp, m_ask_oracle_user_prompts)

        step(f'[STEP 2] LLM Oracle user prompts saved to file: {prompts_json_fp}')


    ###
    # END: PROMPT CONSTRUCTION
    ###


    ###
    # START: QUERY-SPECIFIC RAG FEW-SHOT
    ###
    # Each M_ask candidate gets its own typed examples (CLS queries get CLS examples,
    # OPROP/DPROP get property examples, INST get instance examples), written to
    # few_shot_json() as a dict keyed by the M_ask key, plus per-query retrieval traces
    # for provenance. STATIC_* strategies use query-agnostic examples.

    few_shot_k = config.few_shot.few_shot_k

    if few_shot_k > 0:

        # plan.py refuses to generate a batch without a layout; this refuses to run one,
        # which also covers the single-run `logmap-llm` entry point that never passes
        # through the planner. Deliberately not a schema validator: that would make every
        # frozen historical config unloadable (see config/schema.py).
        negative_layout = config.few_shot.rag_negative_layout
        if negative_layout is None:
            raise ValueError(
                "few_shot.rag_negative_layout is required when few_shot_k > 0; choose "
                "'paired-sibling-v2', 'paired-donor-v2', or the frozen reproduction layout "
                "'donor-cross-v1'. Refusing to guess how negatives should be constructed."
            )

        step(f"[STEP 2] Building query-specific RAG few-shot "
             f"(strategy={config.few_shot.few_shot_negative_strategy}, k={few_shot_k}, "
             f"negative_layout={negative_layout})", important=True)

        try:
            from logmap_llm.pipeline.rag_fewshot import (
                build_query_specific_few_shot,
                load_prebuilt_few_shot_bundle,
                make_sibling_fn,
                rag_dataset_fingerprint,
            )

            prebuilt_path = config.few_shot.prebuilt_few_shot_bundle_path
            if prebuilt_path:
                step(
                    f"[STEP 2] Loading strict prebuilt few-shot bundle: {prebuilt_path}",
                    important=True,
                )
                per_query, traces = load_prebuilt_few_shot_bundle(
                    prebuilt_path,
                    mappings=mappings,
                    m_ask_df=m_ask_df,
                    m_ask_path=os.fspath(m_ask_fp),
                    expected_query_keys=m_ask_oracle_user_prompts,
                    receiver_task=config.alignmentTask.task_name,
                    train_tsv_path=config.evaluation.train_alignment_path,
                    k=few_shot_k,
                    strategy=config.few_shot.few_shot_negative_strategy,
                    encoder_kind=config.few_shot.rag_encoder_kind,
                    encoder_model=config.few_shot.rag_encoder_model,
                    encoder_revision=config.few_shot.rag_encoder_revision,
                    answer_format=config.oracle.answer_format,
                    response_mode=config.oracle.response_mode,
                    prompt_family=oupt_name,
                    property_prompt_family=property_prompt_template_name,
                    data_property_prompt_family=data_property_prompt_template_name,
                    instance_prompt_family=instance_prompt_template_name,
                    bidirectional=bidirectional_mode,
                )
            else:
                # the class user-prompt template (bound with the sibling selector when required)
                cls_fn = opb.get_oracle_user_prompt_template_function(oupt_name, prompt_context)
                if sibling_selector is not None and opb.registry.requires_siblings(oupt_name):
                    cls_fn = partial(cls_fn, sibling_selector=sibling_selector)

                sibling_fn = None
                sibling_strategy = ""
                sibling_encoder_revision = ""
                if negatives_need_siblings:
                    if sibling_selector is None:
                        raise RuntimeError(
                            "rag_negative_layout='paired-sibling-v2' requires a SiblingSelector"
                        )
                    sibling_fn = make_sibling_fn(
                        OA_source, OA_target, sibling_selector,
                        max_candidates=config.prompts.sibling_max_candidates,
                    )
                    sibling_strategy = sibling_selector.strategy.value
                    sibling_encoder_revision = sibling_selector.model_revision
                    step(f"[STEP 2] Sibling negatives enabled "
                         f"(strategy={sibling_strategy}, revision={sibling_encoder_revision or 'n/a'})",
                         important=True)

                per_query, traces = build_query_specific_few_shot(
                    mappings=mappings,
                    m_ask_df=m_ask_df,
                    OA_source=OA_source, OA_target=OA_target,
                    cls_fn=cls_fn,
                    property_fn=property_prompt_function,
                    data_property_fn=data_property_prompt_function,
                    instance_fn=instance_prompt_function,
                    strategy=config.few_shot.few_shot_negative_strategy,
                    k=few_shot_k,
                    seed=config.few_shot.few_shot_seed,
                    bidirectional=bidirectional_mode,
                    answer_format=config.oracle.answer_format,
                    response_mode=config.oracle.response_mode,
                    prompt_family=oupt_name,
                    negative_layout=negative_layout,
                    sibling_fn=sibling_fn,
                    sibling_strategy=sibling_strategy,
                    sibling_encoder_revision=sibling_encoder_revision,
                    train_tsv_path=config.evaluation.train_alignment_path,
                    dataset_sha=rag_dataset_fingerprint(
                        mappings,
                        m_ask_df,
                        config.evaluation.train_alignment_path,
                    ),
                    cache_dir=config.few_shot.rag_cache_dir,
                    encoder_kind=config.few_shot.rag_encoder_kind,
                    encoder_model=config.few_shot.rag_encoder_model,
                    encoder_revision=config.few_shot.rag_encoder_revision,
                    encoder_device=config.few_shot.rag_encoder_device,
                    encoder_max_length=config.few_shot.rag_encoder_max_length,
                    failure_policy=config.few_shot.rag_failure_policy,
                )

            few_shot_examples_json_fp = run_paths.few_shot_json()
            _atomic_write_json(few_shot_examples_json_fp, per_query)

            traces_fp = os.path.join(os.path.dirname(str(few_shot_examples_json_fp)), "rag_traces.json")
            _atomic_write_json(traces_fp, traces, indent=2)
            recorded_fallback = _publish_recorded_trace_fallbacks(
                run_paths, config.few_shot, traces
            )
            recorded_negative_fallback = _publish_negative_construction_fallbacks(
                run_paths, config.few_shot, traces
            )

            step(f"[STEP 2] SAVED query-specific few-shot for {len(per_query)} M_ask keys to "
                 f"{str(few_shot_examples_json_fp)} (traces: {traces_fp})", important=True)
            if recorded_fallback:
                warn(
                    "One or more RAG queries used a configured fallback; details are "
                    f"recorded in {_rag_artifact_path(run_paths, 'rag_fallback.json')}"
                )
            if recorded_negative_fallback:
                warn(
                    "One or more negatives fell back from the layout's primary target rule; "
                    "per-lane rates are recorded in "
                    f"{_rag_artifact_path(run_paths, 'rag_negative_fallback.json')}"
                )

        except Exception as e:
            warning(f'Query-specific RAG few-shot generation failed: {e}')
            _handle_rag_generation_failure(run_paths, config.few_shot, e)
            fallback_fp = _rag_artifact_path(run_paths, "rag_fallback.json")
            warn(f'Consultation will proceed zero-shot; fallback recorded to {fallback_fp}')

    else:
        # A zero-shot retry in an existing attempt directory must not inherit
        # examples or fallback metadata from an earlier few-shot attempt.
        _atomic_write_json(run_paths.few_shot_json(), {})
        _atomic_write_json(_rag_artifact_path(run_paths, "rag_traces.json"), {})
        _remove_stale_rag_fallback(run_paths)


if __name__ == "__main__":
    main()
