"""
Isolation harness for the mediation arm; used only by the mediation isolation tests, never
imported by the baseline pipeline.

`run_standard_stub_arm` runs a representative standard LogMapLLM arm built only from baseline
code (baseline consultation + set-union refinement) with a local stub oracle; its artifacts
must be byte-identical whether or not the mediation module exists/is imported. The CLI
(`python -m logmap_llm.pipeline.mediation_isolation`) lets tests spawn arms as separate OS
processes. The stub oracle here is deliberately independent of mediation's StubOpenAIClient
so the "without mediation" configuration is genuine.
"""
from __future__ import annotations

import os
import sys
import json
import hashlib
import argparse
import contextlib
from pathlib import Path

import pandas as pd

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
from logmap_llm.pipeline.orchestration import _kg_refine_in_python
import logmap_llm.oracle.consultation as oc


# --- a deterministic stub oracle local to this harness (independent of mediation.py) ---

def _stub_accept(prompt: str) -> bool:
    d = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return (int(d[:8], 16) % 2) == 0


class _Msg:
    def __init__(self, c):
        self.content = c
        self.parsed = None


class _Choice:
    def __init__(self, c):
        self.message = _Msg(c)
        self.logprobs = None


class _Usage:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0


class _Resp:
    def __init__(self, c):
        self.choices = [_Choice(c)]
        self.usage = _Usage()


class _Completions:
    def create(self, **kw):
        target = ""
        for m in reversed(kw.get("messages", [])):
            if m.get("role") == "user":
                target = m.get("content", "")
                break
        return _Resp("true" if _stub_accept(target) else "false")

    def parse(self, **kw):
        return self.create(**kw)


class _Chat:
    def __init__(self):
        self.completions = _Completions()


class _StdStubClient:
    def __init__(self, *a, **k):
        self.chat = _Chat()


@contextlib.contextmanager
def _patched_std_transport():
    import logmap_llm.oracle.manager as mgr
    original = mgr.OpenAI
    mgr.OpenAI = lambda *a, **k: _StdStubClient()
    try:
        yield
    finally:
        mgr.OpenAI = original


def _std_oracle_config(max_workers: int = 4) -> OracleConfig:
    return OracleConfig(
        model_name="stub-standard-oracle", api_key="EMPTY",
        base_url="http://localhost:0/v1", interaction_style="vllm",
        response_mode="plain", answer_format="true_false",
        temperature=0.0, top_p=1.0, max_workers=max_workers,
        enable_thinking=False, max_completion_tokens=8,
    )


# A fixed standard M_ask fixture (a handful of class candidates) + a fixed initial alignment.
# Held constant so the standard arm's artifacts are deterministic and comparable across processes.
_STD_INITIAL = [
    ("http://ex.org/o1#A1", "http://ex.org/o2#B1", "=", 0.99, "CLS"),
    ("http://ex.org/o1#A2", "http://ex.org/o2#B2", "=", 0.97, "CLS"),
]
_STD_M_ASK = [
    ("http://ex.org/o1#C1", "http://ex.org/o2#D1", "=", 0.80, "CLS"),
    ("http://ex.org/o1#C2", "http://ex.org/o2#D2", "=", 0.81, "CLS"),
    ("http://ex.org/o1#C3", "http://ex.org/o2#D3", "=", 0.82, "CLS"),
    ("http://ex.org/o1#C4", "http://ex.org/o2#D4", "=", 0.83, "CLS"),
]


def run_standard_stub_arm(out_root: str) -> dict:
    """Run a representative standard LogMapLLM arm from baseline code + the local stub oracle.
    Returns artifact paths + their sha256. Deterministic."""
    cols = [COL_SOURCE_ENTITY_URI, COL_TARGET_ENTITY_URI, COL_RELATION, COL_CONFIDENCE, COL_ENTITY_TYPE]
    run_paths = PipelinePaths(
        output_dir=os.path.join(out_root, "out"),
        initial_dir=os.path.join(out_root, "initial"),
        refined_dir=os.path.join(out_root, "refined"),
        task_name="standard-stub", oupt_name="standard",
        run_id="std-fixed", isolate_run=True,
    )
    run_paths.create_base_dirs()

    # write the fixed initial alignment where _kg_refine_in_python reads it (pipe-sep, no header)
    init_df = pd.DataFrame(_STD_INITIAL, columns=cols)
    init_df.to_csv(run_paths.logmap_mappings(), sep=PAIRS_SEPARATOR, header=False, index=False)

    m_ask = pd.DataFrame(_STD_M_ASK, columns=cols)
    prompts = {
        f"{s}{PAIRS_SEPARATOR}{t}": f"standard candidate {s} vs {t}: equivalent? true/false"
        for (s, t, *_rest) in _STD_M_ASK
    }

    with _patched_std_transport():
        preds = oc.consult_oracle_for_mappings_to_ask(
            m_ask_prompts=prompts, m_ask_init_alignment_df=m_ask,
            oracle_cfg=_std_oracle_config(), developer_prompt_text="standard dev prompt",
            developer_prompt_map=None, few_shot_examples=None,
        )
    preds.to_csv(run_paths.predictions_csv(), na_rep="nan", index=False)

    _kg_refine_in_python(PipelineContext(cfg=None, run_paths=run_paths, logmap=None), OracleResult(predictions=preds))

    def _sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    artifacts = {
        "predictions_csv": str(run_paths.predictions_csv()),
        "refined_tsv": str(run_paths.refined_mappings_tsv()),
        "initial_txt": str(run_paths.logmap_mappings()),
    }
    hashes = {k: _sha(v) for k, v in artifacts.items()}
    return {"run_id": "std-fixed", "run_dir": str(run_paths.output_dir),
            "artifacts": artifacts, "hashes": hashes}


def _cli(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["standard", "mediation", "import-check"], required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--import-mediation", action="store_true",
                   help="import pipeline.mediation before running (T2: prove presence is inert)")
    # mediation-mode passthrough
    p.add_argument("--track", default=None)
    p.add_argument("--composed", default=None)
    p.add_argument("--n-min", type=int, default=1)
    p.add_argument("--initial-alignment", default=None)
    args = p.parse_args(argv)

    if args.mode == "mediation":
        print(
            "MEDIATION_QUARANTINED: the historical isolation harness cannot execute "
            "mediation; a future provenance-disjoint appendix requires a separate entry point.",
            file=sys.stderr,
        )
        return 78

    if args.import_mediation:
        import logmap_llm.pipeline.mediation as _med  # noqa: F401  (presence must be inert)

    if args.mode == "import-check":
        # import-inertness: importing mediation must not mutate the oracle transport symbol
        import logmap_llm.oracle.manager as mgr
        before = mgr.OpenAI
        import logmap_llm.pipeline.mediation  # noqa: F401
        after = mgr.OpenAI
        print(json.dumps({"openai_symbol_unchanged": before is after}))
        return 0

    # The baseline logger prints DEBUG to stdout (VERBOSE=True in constants.py). Redirect stdout
    # -> stderr while the arm runs, so this process emits only a single clean JSON line to its
    # real stdout for the parent harness to parse.
    real_stdout = sys.stdout

    if args.mode == "standard":
        with contextlib.redirect_stdout(sys.stderr):
            res = run_standard_stub_arm(args.out)
        print(json.dumps(res), file=real_stdout)
        return 0

    if args.mode == "mediation":  # pragma: no cover - blocked above
        import logmap_llm.pipeline.mediation as med
        with contextlib.redirect_stdout(sys.stderr):
            mcfg = med.MediationConfig(
                enable_mediation=True, track=args.track, composed_candidates_path=args.composed,
                n_min=args.n_min, initial_alignment_path=args.initial_alignment,
                output_dir=os.path.join(args.out, "out"),
                initial_dir=os.path.join(args.out, "initial"),
                refined_dir=os.path.join(args.out, "refined"),
                stub_oracle=True,
            )
            res = med.run_mediation_arm(mcfg)
        import glob
        refined = glob.glob(os.path.join(args.out, "out", "**", "*.json"), recursive=True)
        print(json.dumps({
            "run_id": res["run_id"], "run_dir": res["run_dir"],
            "manifest": res["manifest"], "evaluation_results": res["evaluation_results"],
            "n_json_artifacts": len(refined),
        }), file=real_stdout)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
