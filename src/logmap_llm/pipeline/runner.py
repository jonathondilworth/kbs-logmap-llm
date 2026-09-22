"""
logmap_llm.pipeline.runner

Main entry point for the LogMap-LLM pipeline. Bootstraps the run (config
validation, path resolution via PipelinePaths, JVM/LogMap startup, tee
logging) into a PipelineContext, then executes the phases in order:
align -> prompt_build -> consult_oracle -> refine_alignment -> evaluate
(optional) -> reporting. Prompt building runs in a subprocess to avoid
conflicts between concurrent JVM and owlready2 instances; evaluation
also supports partial gold standards (OAEI 2025 KG track semantics).
"""
from __future__ import annotations

import sys
import os
import os.path
import time
from datetime import datetime, timezone
from argparse import Namespace

from logmap_llm.interface import start_jvm, LogMapInterface
from logmap_llm.config.schema import LogMapLLMConfig
from logmap_llm.config.loader import (
    load_and_validate_config,
    print_config_summary,
)
from logmap_llm.pipeline.contracts import TimingRecord
from logmap_llm.pipeline.paths import PipelinePaths
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.orchestration import (
    align,
    prompt_build,
    consult_oracle,
    refine_alignment,
    evaluate,
)
from logmap_llm.pipeline.reporting import (
    print_timing_summary,
    print_experimental_parameters,
    write_results_file,
)
from logmap_llm.utils.logging import (
    TeeWriter,
    info,
    success,
    critical,
)


# at this level of abstraction the code should be self-documenting; see
# config/schema.py and pipeline/context.py for what is passed between
# phases via PipelineContext.


def main(args: Namespace | None = None) -> int:
    """Begin a full pipeline run."""

    ###
    # BOOTSTRAPPING
    ###############

    if args is None:
        from logmap_llm.pipeline.cli import parse_args
        args:Namespace = parse_args()

    expr_run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    cfg:LogMapLLMConfig = load_and_validate_config(
        args.config,
        reuse_align=args.reuse_align,
        reuse_prompts=args.reuse_prompts,
    )

    # RESOLVING FILE PATHS
    #######################
    # PipelinePaths (paths.py) manages artifact file paths (mappings, results, few-shot
    # examples, etc) - esp. useful when running multiple LogMap-LLM processes concurrently.
    # Paths concatenate the task name, prompt name and an optional suffix (e.g. fs8_hard =
    # few-shot, 8 examples, hard contrastive strategy); the suffix can be set in the config
    # or by an orchestrator, keeping filesystem artifacts isolated between processes.

    run_paths:PipelinePaths = PipelinePaths.from_config(
        cfg,
        run_root=getattr(args, "run_root", None),
    )

    # Logging and the JVM both write beneath these paths. A brand-new run root
    # must therefore be created before either is initialised.
    if not run_paths.create_base_dirs():
        raise OSError("Experimental directories could not be created")
    # A failed retry must not leave a prior success marker in place.
    run_paths.run_result().unlink(missing_ok=True)

    # logging (see utils.logging)
    tee_branch_out:TeeWriter = TeeWriter(
        str(run_paths.run_log(expr_run_timestamp)),
        sys.stdout,
    )
    sys.stdout = tee_branch_out

    try:

        print_config_summary(cfg)

        info(f"Summary of File Paths:\n\n{run_paths.summary()}\n\n")

        ###
        # START LOGMAP
        ##############

        # if logmap_dirpath is unset, assume the official logmap directory lives within the
        # project root (or wherever you run this pipeline from) - required for LogMap params
        # + java-deps

        logmap_dirpath = os.path.join(os.getcwd(), 'logmap')

        if cfg.alignmentTask.logmap_parameters_dirpath:
            logmap_dirpath = cfg.alignmentTask.logmap_parameters_dirpath

        needs_logmap = (
            cfg.pipeline.align_ontologies.value == "align"
            or (
                cfg.pipeline.refine_alignment.value == "refine"
                and not cfg.pipeline.stop_after_consultation
                and (
                    cfg.pipeline.refinement_strategy.value == "logmap"
                    or cfg.pipeline.consult_oracle.value == "local"
                )
            )
        )
        logmap = None
        if needs_logmap:
            start_jvm(
                logmap_dir=logmap_dirpath,
                max_heap=cfg.alignmentTask.logmap_jvm_memory,
            )
            logmap = LogMapInterface.create_from_cfg(cfg, logmap_dirpath)
            logmap.set_output_dir(run_paths.initial_dir)


        # ONTOLOGY DOMAIN
        #################
        # lets the prompts reference the type of ontologies being aligned
        # (eg. 'conference', 'circular economy', 'knowledge graphs')

        if cfg.alignmentTask.ontology_domain:
            info(f"Ontology domain: {cfg.alignmentTask.ontology_domain}\n")

        # PIPELINE CONTEXT (CTX)
        ########################
        # bundles the shared 'context' (config + convenient accessors) passed between
        # pipeline phases, so we don't pass a long parameter list down the call stack

        pipeline_ctx = PipelineContext(
            cfg,
            run_paths,
            logmap,
            config_path=args.config,
            no_cache=getattr(args, "no_cache", False),
        )


        # PIPELINE PHASES
        #################
        # each phase follows the same pattern: start timer, execute the phase
        # (imported from orchestration.py), stop timer, print completion message

        info("LogMap-LLM pipeline starting.")

        timing = TimingRecord()


        # ALIGN
        #######

        align_start_time = time.time()

        align_result = align(pipeline_ctx)

        timing.align_seconds = time.time() - align_start_time

        if align_result.n_mappings > 0:
            success(f"Number of mappings within the initial alignment: {align_result.n_mappings}")
            success(f"Number of mappings within M_ask: {align_result.n_m_ask}")


        # PROMPT BUILD
        ##############

        prompt_build_start_time = time.time()

        prompt_build_result = prompt_build(pipeline_ctx, align_result)

        timing.prompt_build_seconds = time.time() - prompt_build_start_time

        if prompt_build_result.n_prompts > 0 and not prompt_build_result.bidirectional:
            success(f"Number of LLM oracle user prompts: {prompt_build_result.n_prompts}")

        if prompt_build_result.n_prompts > 0 and prompt_build_result.bidirectional:
            success_suffix = f" ({prompt_build_result.n_prompts // 2} candidates x 2 directions)"
            success(f"Number of LLM oracle user prompts: {prompt_build_result.n_prompts}{success_suffix}")


        # CONSULT ORACLE
        ################

        consult_oracle_start_time = time.time()

        oracle_result = consult_oracle(
            pipeline_ctx, align_result, prompt_build_result
        )

        timing.consult_seconds = time.time() - consult_oracle_start_time

        if oracle_result.prediction_summary() is not None:
            for summary_message in oracle_result.prediction_summary(return_list=True):
                success(summary_message)
            success(f"Oracle predictions saved to: {run_paths.predictions_csv()}")


        # ANNOTATED M_ASK
        #################
        # every consultation (consult or reuse) also leaves the verdicts next to the
        # candidates (pipeline/annotate.py); local and bypass modes have no verdicts here

        if oracle_result.has_predictions and cfg.pipeline.consult_oracle.value in ("consult", "reuse"):
            from logmap_llm.pipeline.annotate import write_annotated_files

            annotated_txt, annotated_tsv = write_annotated_files(
                run_paths, oracle_result.predictions, oracle_result.bidirectional
            )
            success(f"Annotated M_ask saved to: {annotated_txt} and {annotated_tsv}")

        stopped_after = None

        if cfg.pipeline.stop_after_consultation:

            # STOP AFTER CONSULTATION
            #########################

            from logmap_llm.pipeline.contracts import RefinementResult, EvaluationResult

            stopped_after = "consultation"
            info("stop_after_consultation = true: refinement and evaluation are skipped")
            refinement_result = RefinementResult()
            eval_result = EvaluationResult()

        else:

            # REFINE ALIGNMENT
            ##################

            refine_alignment_start_time = time.time()

            refinement_result = refine_alignment(pipeline_ctx, oracle_result)

            timing.refine_seconds = time.time() - refine_alignment_start_time

            success("Refinement stage ending.") # success: no exceptions raised


            # EVALUATE
            ##########

            evaluate_start_time = time.time()

            eval_result = evaluate(pipeline_ctx)

            timing.evaluate_seconds = time.time() - evaluate_start_time

            if eval_result.subprocess_failed:
                critical("Evaluation subprocess failed — run completed with errors")
            else:
                success("Evaluation stage ending.")


        # REPORTING
        ###########

        all_task_times = [
            timing.align_seconds,
            timing.prompt_build_seconds,
            timing.consult_seconds,
            timing.refine_seconds,
            timing.evaluate_seconds,
        ]

        timing.total_seconds = sum(
            task_time
            for task_time in all_task_times if task_time is not None
        )

        n_consultations = prompt_build_result.n_prompts

        print_timing_summary(timing, n_consultations)

        print_experimental_parameters(
            cfg, oracle_result, prompt_build_result, timing
        )


        # WRITE FINAL RESULTS TO DISK
        #############################

        if refinement_result.n_refined_mappings > 0:
            success(f"Total refined mappings: {refinement_result.n_refined_mappings}")

        if eval_result.subprocess_failed:
            critical("LogMap-LLM session ending (with errors)")
            exit_code = 1
        else:
            results_path = run_paths.run_result()
            write_results_file(
                results_path,
                cfg,
                timing,
                eval_result,
                oracle_result,
                prompt_build_result,
                run_paths,
                stopped_after=stopped_after,
            )
            success("LogMap-LLM session ending")
            exit_code = 0


    finally:
        sys.stdout = tee_branch_out.original_stdout
        tee_branch_out.close()

    return exit_code



if __name__ == "__main__":
    sys.exit(main())
