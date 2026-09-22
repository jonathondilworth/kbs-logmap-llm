"""
WP4 bug fixes (22 Sep 2026, LOCAL_CHANGES.md §6):

1. `consult_oracle = "local"`: the predictions directory is handed to LogMap with a trailing
   separator (`LocalOracle.loadLocalOraculoLLM` concatenates `base_path + filename`);
2. the verdicts LogMap will load are counted with the Java loader's rule and a directory
   without any verdict is fatal (an empty local oracle used to pass as a successful run in
   which every candidate was rejected);
3. an OpenRouter 200 response whose `choices` is `None` (provider error object in the body)
   is a transient failure: retried under `transient_retries`, then recorded as an error.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from logmap_llm.config.schema import validate_config
from logmap_llm.constants import BinaryOutputFormat, InteractionStyle
from logmap_llm.pipeline.context import PipelineContext
from logmap_llm.pipeline.contracts import AlignmentResult, OracleResult, PromptBuildResult
from logmap_llm.pipeline.orchestration import consult_oracle, load_local_oracle_verdicts, refine_alignment
from logmap_llm.pipeline.paths import PipelinePaths

CSV = """Source,Target,Prediction,Confidence
# a comment line
http://a#1,http://b#1,True,0.99
http://a#2,http://b#2,true,0.80

http://a#3,http://b#3,False,0.70
this line has no comma
"""


def _predictions_dir(tmp_path: Path, name: str = "preds", content: str | None = CSV) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    if content is not None:
        (directory / "gemini_results.csv").write_text(content)
        (directory / "notes.txt").write_text("Source,Target,Prediction\nx,y,True\n")   # not a .csv: ignored
    return directory


def _config(tmp_path: Path, predictions_dir: Path) -> tuple:
    root = tmp_path / "run"
    cfg = validate_config({
        "alignmentTask": {"task_name": "t", "onto_source_filepath": "s.owl", "onto_target_filepath": "t.owl"},
        "oracle": {"model_name": "replayed", "interaction_style": "local", "response_mode": "plain",
                   "local_oracle_predictions_dirpath": str(predictions_dir)},
        "outputs": {"logmapllm_output_dirpath": str(root / "out"),
                    "logmap_initial_alignment_output_dirpath": str(root / "init"),
                    "logmap_refined_alignment_output_dirpath": str(root / "ref")},
        "pipeline": {"align_ontologies": "reuse", "build_oracle_prompts": "bypass", "consult_oracle": "local",
                     "refine_alignment": "refine"},
    })
    paths = PipelinePaths.from_config(cfg)
    paths.create_base_dirs()
    return cfg, paths


# ---------------------------------------------------------------- 1 + 2: local oracle loader

def test_local_oracle_verdicts_follow_the_java_loader_rule(tmp_path):
    counts = load_local_oracle_verdicts(_predictions_dir(tmp_path))
    assert counts == {"files": 1, "true": 2, "false": 1, "non_boolean": 1}


def test_local_oracle_empty_directory_is_fatal(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_local_oracle_verdicts(tmp_path / "missing")
    empty = _predictions_dir(tmp_path, "empty", content=None)
    assert load_local_oracle_verdicts(empty) == {"files": 0, "true": 0, "false": 0, "non_boolean": 0}
    cfg, paths = _config(tmp_path, empty)
    ctx = PipelineContext(cfg, paths, logmap=None)
    with pytest.raises(ValueError):
        consult_oracle(ctx, AlignmentResult(), PromptBuildResult())


def test_consult_local_records_counts_and_trailing_separator(tmp_path):
    predictions = _predictions_dir(tmp_path)
    cfg, paths = _config(tmp_path, predictions)
    ctx = PipelineContext(cfg, paths, logmap=None)
    result = consult_oracle(ctx, AlignmentResult(), PromptBuildResult())
    assert result.is_local and not result.has_predictions
    assert str(result.local_dir).endswith(os.sep)
    assert result.local_oracle_verdicts == {"files": 1, "true": 2, "false": 1, "non_boolean": 1}


class _FakeJavaSet:
    def toArray(self):
        return []

    def __len__(self):
        return 0


class _FakeLogMap:
    def __init__(self):
        self.refine_paths: list[str] = []
        self.output_dirs: list = []

    def set_output_dir(self, path):
        self.output_dirs.append(path)

    def refine_alignment(self, argument):
        self.refine_paths.append(argument)

    def get_mappings(self):
        return _FakeJavaSet()


def test_refine_passes_directory_with_separator_to_logmap(tmp_path):
    predictions = _predictions_dir(tmp_path)
    cfg, paths = _config(tmp_path, predictions)
    logmap = _FakeLogMap()
    ctx = PipelineContext(cfg, paths, logmap=logmap)
    # even a bare directory (no separator) reaches LogMap with one
    refine_alignment(ctx, OracleResult(local_dir=str(predictions), predictions=None))
    assert logmap.refine_paths == [str(predictions) + os.sep]
    assert logmap.output_dirs == [paths.refined_dir]


def test_run_result_counts_carry_the_local_oracle_verdicts(tmp_path):
    from logmap_llm.pipeline.reporting import write_results_file
    from logmap_llm.pipeline.contracts import EvaluationResult, TimingRecord
    import json

    predictions = _predictions_dir(tmp_path)
    cfg, paths = _config(tmp_path, predictions)
    # publish the artifacts the reporting validator requires for this config
    paths.logmap_mappings().write_text("http://a#1|http://b#1|=|0.9|CLS\n")
    paths.logmap_m_ask().write_text("http://a#1|http://b#1|=|0.9|CLS\n")
    paths.refined_mappings_tsv().write_text("http://a#1\thttp://b#1\t=\t0.9\tCLS\n")
    oracle_result = OracleResult(local_dir=str(predictions) + os.sep, predictions=None,
                                 local_oracle_verdicts={"files": 1, "true": 2, "false": 1, "non_boolean": 1})
    write_results_file(paths.run_result(), cfg, TimingRecord(), EvaluationResult(), oracle_result,
                       PromptBuildResult(), paths)
    payload = json.loads(paths.run_result().read_text())
    assert payload["counts"]["local_oracle"] == {"files": 1, "true": 2, "false": 1, "non_boolean": 1}
    assert payload["counts"]["candidates"] == 0 and payload["status"] == "succeeded"


# ---------------------------------------------------------------- 3: OpenRouter choices: None

class _FlakyCompletions:
    """`create` answers `choices=None` + an error object `failures` times, then a real choice."""

    def __init__(self, failures: int, content: str = "True"):
        self.failures = failures
        self.calls = 0
        self._content = content

    def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.failures:
            return SimpleNamespace(choices=None, error={"code": 502, "message": "Provider returned error"},
                                   provider="Google AI Studio", usage=None)
        choice = SimpleNamespace(message=SimpleNamespace(content=self._content), finish_reason="stop", logprobs=None)
        return SimpleNamespace(choices=[choice], usage=None)


def _manager(completions, retries: int = 2):
    from logmap_llm.oracle.manager import OracleConsultationManager

    mgr = OracleConsultationManager.__new__(OracleConsultationManager)
    mgr.response_format = BinaryOutputFormat
    mgr.model_name = "test-model"
    mgr.temperature = 0.0
    mgr.top_p = 1.0
    mgr.seed = 0
    mgr.max_completion_tokens = 16
    mgr.interaction_style = InteractionStyle.OPEN_ROUTER
    mgr.enable_thinking = None
    mgr.reasoning_effort = None
    mgr.reasoning_token_budget = None
    mgr.openrouter_provider = None
    mgr.openrouter_allow_fallbacks = False
    mgr.openrouter_require_parameters = True
    mgr.supports_chat_template_kwargs = False
    mgr.logprobs = False
    mgr.logprobs_requested = False
    mgr.top_logprobs = None
    mgr.transient_retries = retries
    mgr._frozen = True
    mgr._frozen_messages = ()
    mgr.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return mgr


def test_choices_none_raises_transient_error_with_provider_error():
    from logmap_llm.oracle.manager import OracleTransientResponseError

    mgr = _manager(_FlakyCompletions(failures=5))
    with pytest.raises(OracleTransientResponseError) as info:
        mgr._consult_via_create("prompt")
    assert info.value.error == {"code": 502, "message": "Provider returned error"}
    assert info.value.provider == "Google AI Studio"


def test_choices_none_is_retried_then_recorded_as_error(monkeypatch):
    import logmap_llm.oracle.consultation as oc

    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)
    completions = _FlakyCompletions(failures=10)
    key, prediction, confidence, usage = oc.consult_oracle_for_mapping("a|b", "prompt", _manager(completions, retries=2))
    assert (key, prediction) == ("a|b", "error") and math.isnan(confidence)
    assert completions.calls == 3           # first attempt + two transient retries
    assert usage.input_tokens is None
    assert oc._is_transient_oracle_error(oc.OracleTransientResponseError({"code": 502}))


def test_choices_none_recovers_on_retry(monkeypatch):
    import logmap_llm.oracle.consultation as oc

    monkeypatch.setattr(oc.time, "sleep", lambda seconds: None)
    completions = _FlakyCompletions(failures=1, content='{"answer": true}')
    key, prediction, confidence, usage = oc.consult_oracle_for_mapping("a|b", "prompt", _manager(completions, retries=2))
    assert prediction is True and completions.calls == 2
