"""
logmap_llm.pipeline.paths
"""
from __future__ import annotations

import json
import hashlib

from pathlib import Path
from logmap_llm.config.schema import LogMapLLMConfig


def _file_sha256(path: str | Path | None) -> str | None:
    if not path:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_run_id(cfg: "LogMapLLMConfig", extra: dict | None = None) -> str:
    """
    Content-addressed run id: a short sha256 over the semantically-identifying config fields,
    so two runs collide on a path only when they are the same experiment (distinct configs get
    distinct output subdirs, making parallel grids under a shared output_dir safe).

    `extra` folds in identity that lives outside the config: matcher JAR sha, parameters sha +
    resolved launch dir, framework commit, serving engine+version, seed/repeat, eval protocol.
    """
    def get_field(obj, name, default=None):
        return getattr(obj, name, default)

    ident = {
        "task": get_field(cfg.alignmentTask, "task_name"),
        "onto_src": get_field(cfg.alignmentTask, "onto_source_filepath"),
        "onto_tgt": get_field(cfg.alignmentTask, "onto_target_filepath"),
        "model": get_field(cfg.oracle, "model_name"),
        "temperature": get_field(cfg.oracle, "temperature"),
        "top_p": get_field(cfg.oracle, "top_p"),
        "answer_format": get_field(cfg.oracle, "answer_format"),
        "response_mode": get_field(cfg.oracle, "response_mode"),
        "enable_thinking": get_field(cfg.oracle, "enable_thinking"),
        "max_completion_tokens": get_field(cfg.oracle, "max_completion_tokens"),
        "reasoning_effort": get_field(cfg.oracle, "reasoning_effort"),
        "reasoning_token_budget": get_field(cfg.oracle, "reasoning_token_budget"),
        "seed": get_field(cfg.oracle, "seed"),
        "request_logprobs": get_field(cfg.oracle, "request_logprobs"),
        "openrouter_provider": get_field(cfg.oracle, "openrouter_provider"),
        "openrouter_allow_fallbacks": get_field(cfg.oracle, "openrouter_allow_fallbacks"),
        "openrouter_require_parameters": get_field(cfg.oracle, "openrouter_require_parameters"),
        "request_timeout_seconds": get_field(cfg.oracle, "request_timeout_seconds"),
        "connect_timeout_seconds": get_field(cfg.oracle, "connect_timeout_seconds"),
        "transient_retries": get_field(cfg.oracle, "transient_retries"),
        "cls_prompt": get_field(cfg.prompts, "cls_usr_prompt_template_name"),
        "prop_prompt": get_field(cfg.prompts, "prop_usr_prompt_template_name"),
        "dprop_prompt": get_field(cfg.prompts, "dprop_usr_prompt_template_name"),
        "inst_prompt": get_field(cfg.prompts, "inst_usr_prompt_template_name"),
        "cls_dev_prompt": get_field(cfg.prompts, "cls_dev_prompt_template_name"),
        "prop_dev_prompt": get_field(cfg.prompts, "prop_dev_prompt_template_name"),
        "inst_dev_prompt": get_field(cfg.prompts, "inst_dev_prompt_template_name"),
        "sibling_strategy": get_field(cfg.prompts, "sibling_strategy"),
        "few_shot_k": get_field(cfg.few_shot, "few_shot_k"),
        "few_shot_strategy": get_field(cfg.few_shot, "few_shot_negative_strategy"),
        "few_shot_seed": get_field(cfg.few_shot, "few_shot_seed"),
        "rag_encoder_kind": get_field(cfg.few_shot, "rag_encoder_kind"),
        "rag_encoder_model": get_field(cfg.few_shot, "rag_encoder_model"),
        "rag_encoder_revision": get_field(cfg.few_shot, "rag_encoder_revision"),
        "rag_encoder_max_length": get_field(cfg.few_shot, "rag_encoder_max_length"),
        "rag_failure_policy": get_field(cfg.few_shot, "rag_failure_policy"),
        "prebuilt_few_shot_bundle_sha256": _file_sha256(
            get_field(cfg.few_shot, "prebuilt_few_shot_bundle_path")
        ),
    }
    if extra:
        ident.update(extra)
    blob = json.dumps(ident, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class PipelinePaths:
    """
    Manages pipeline artefact paths.

    Naming convention:
    - artefacts:      output_dir  / f"{task_name}-{oupt_name}-{suffix}"
    - initial aligns: initial_dir / f"{task_name}-logmap_mappings.txt"
    - m_ask:          initial_dir / f"{task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"
    - refined aligns: refined_dir / f"{task_name}-logmap_mappings.tsv"
    """

    def __init__(
        self,
        output_dir: str | Path,
        initial_dir: str | Path,
        refined_dir: str | Path,
        task_name: str,
        oupt_name: str,
        run_id: str | None = None,
        isolate_run: bool = False,
        run_root: str | Path | None = None,
    ):
        # Absolutise so a relative output_dir does not resolve against a per-process CWD (parallel-safety).
        self.output_dir = Path(output_dir).resolve()
        self.initial_dir = Path(initial_dir).resolve()
        self.refined_dir = Path(refined_dir).resolve()
        self.run_root = Path(run_root).resolve() if run_root is not None else None
        self.run_id = run_id
        # isolate_run roots every artifact under a content-addressed {run_id} subdir, so a parallel
        # grid sharing one output_dir cannot silently overwrite/misattribute. Opt-in: default False
        # preserves single-run behaviour; the parallel scheduler sets it True.
        if isolate_run and run_id:
            self.output_dir = self.output_dir / run_id
            self.initial_dir = self.initial_dir / run_id
            self.refined_dir = self.refined_dir / run_id
        self.task_name = task_name
        self.oupt_name = oupt_name

    @classmethod
    def from_config(
        cls,
        cfg: LogMapLLMConfig,
        isolate_run: bool = False,
        run_id_extra: dict | None = None,
        run_root: str | Path | None = None,
    ) -> PipelinePaths:
        """
        Construct from a validated LogMapLLMConfig. Pass isolate_run=True (used by the parallel
        scheduler) to root all artifacts under a content-addressed run-id subdir so concurrent
        grid runs are collision-free; run_id_extra folds in matcher/params/engine/seed identity.
        """
        resolved_root = Path(run_root).resolve() if run_root is not None else None
        output_dir = (
            resolved_root / "logmapllm-outputs"
            if resolved_root is not None
            else cfg.outputs.logmapllm_output_dirpath
        )
        initial_dir = (
            resolved_root / "logmap-initial-alignment"
            if resolved_root is not None
            else cfg.outputs.logmap_initial_alignment_output_dirpath
        )
        refined_dir = (
            resolved_root / "logmap-refined-alignment"
            if resolved_root is not None
            else cfg.outputs.logmap_refined_alignment_output_dirpath
        )
        return cls(
            output_dir=output_dir,
            initial_dir=initial_dir,
            refined_dir=refined_dir,
            task_name=cfg.alignmentTask.task_name,
            oupt_name=cfg.prompts.cls_usr_prompt_template_name,
            run_id=compute_run_id(cfg, extra=run_id_extra),
            isolate_run=isolate_run,
            run_root=resolved_root,
        )

    def _artifact(self, suffix: str) -> Path:
        return self.output_dir / f"{self.task_name}-{self.oupt_name}-{suffix}"

    def prompts_json(self) -> Path:
        return self._artifact("mappings_to_ask_oracle_user_prompts.json")

    def predictions_csv(self) -> Path:
        return self._artifact("mappings_to_ask_with_oracle_predictions.csv")

    def few_shot_json(self) -> Path:
        return self._artifact("few_shot_examples.json")

    def annotated_txt(self) -> Path:
        """M_ask with the oracle verdicts appended, LogMap pipe format (pipeline/annotate.py)."""
        return self._artifact("annotated.txt")

    def annotated_tsv(self) -> Path:
        """The same with a header and the logprob-derived LLM_confidence column."""
        return self._artifact("annotated.tsv")

    def model_ranking_json(self) -> Path:
        """Anchor-derived questions for automatic model selection (pipeline/model_selection.py)."""
        return self._artifact("model_ranking_prompts.json")

    def model_selection_json(self) -> Path:
        """Per-candidate scores and the selected model configuration."""
        return self._artifact("model_selection.json")

    def eval_json(self) -> Path:
        return self.output_dir / "evaluation_results.json"

    def run_log(self, timestamp: str) -> Path:
        return self.output_dir / f"pipeline_log_{timestamp}.txt"

    def subprocess_log(self, timestamp: str, subprocess_name: str = "UNSET") -> Path:
        return self.output_dir / f"subprocess_{subprocess_name}_{timestamp}.txt"

    def run_result(self) -> Path:
        """Canonical machine-readable result, atomically published last."""
        return self.output_dir / "run_result.json"

    def run_results(self, timestamp: str | None = None) -> Path:
        """Backward-compatible alias for :meth:`run_result`."""
        return self.run_result()

    def logmap_mappings(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings.txt"

    def logmap_mappings_tsv(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings.tsv"

    def logmap_m_ask(self) -> Path:
        return self.initial_dir / f"{self.task_name}-logmap_mappings_to_ask_oracle_user_llm.txt"

    def refined_mappings_tsv(self) -> Path:
        return self.refined_dir / f"{self.task_name}-logmap_mappings.tsv"

    def create_base_dirs(self) -> bool:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_dir.mkdir(parents=True, exist_ok=True)
        self.refined_dir.mkdir(parents=True, exist_ok=True)
        return (
            self.output_dir.is_dir()
            and self.initial_dir.is_dir()
            and self.refined_dir.is_dir()
        )

    def summary(self) -> str:
        """Human-readable summary of resolved artifact paths"""
        lines = [
            f"  Task                : {self.task_name}",
            f"  Prompt template     : {self.oupt_name}",
            f"  Run root            : {self.run_root or '<config paths>'}",
            f"  Output directory    : {self.output_dir}",
            f"  Initial align dir   : {self.initial_dir}",
            f"  Refined align dir   : {self.refined_dir}",
            f"  Prompts JSON        : {self.prompts_json().name}",
            f"  Predictions CSV     : {self.predictions_csv().name}",
            f"  Few-shot JSON       : {self.few_shot_json().name}",
            f"  Annotated M_ask     : {self.annotated_txt().name}, {self.annotated_tsv().name}",
        ]
        return "\n".join(lines)
