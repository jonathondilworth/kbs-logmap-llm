# Architecture

LogMapLLM extends the classical LogMap matcher with an LLM oracle. LogMap still does the
matching: it produces an initial alignment and flags the subset of candidate mappings it is
unsure about (M_ask). The LLM's only job is to adjudicate that subset — every candidate becomes
a natural-language question, the model answers True or False, and the accepted candidates are
merged back into the alignment. A run is a fixed sequence of phases driven by
`logmap_llm/pipeline/runner.py`:

    align -> prompt_build -> consult_oracle -> refine_alignment -> evaluate -> reporting

`python -m logmap_llm --config <path>` is the entry point (see [running the
pipeline](pipeline.md)); the whole run is parameterised by a single TOML file validated by the
Pydantic schema in `logmap_llm/config/schema.py` (see the [configuration
reference](configuration.md)). Each phase in `pipeline/orchestration.py` matches a mode enum
from that config (`align`/`reuse`/`bypass` and friends), so any prefix of a run can be replayed
from artifacts on disk instead of recomputed.

## One process, two subprocesses

LogMap is Java. The parent process starts a JVM through JPype (`logmap_llm/interface.py`, a thin
wrapper around LogMap's `LogMapLLM_Interface`) and keeps it for alignment and refinement. Two
phases cannot share that process: prompt building uses owlready2, which cannot coexist with
JPype in the same process, and the DeepOnto evaluation backend insists on starting its own JVM.
So Step 2 runs as `python -m logmap_llm.pipeline.stage_two` and Step 5 as
`python -m logmap_llm.evaluation.harness`, each re-reading the same frozen config via
`logmap_llm/utils/subprocess.py::subprocess_bootstrap` and writing its artifact for the parent
to pick up. The parent forwards `--run-root` and `--no-cache` so the children stay in the same
run namespace.

## Phase walkthrough

**Step 1 — align** (`pipeline/orchestration.py::align`). LogMap performs the classical
alignment and exposes two Java mapping sets: the full initial alignment and M_ask.
`logmap_llm/bridging.py` converts them into pandas DataFrames with the columns
`source_entity_uri, target_entity_uri, relation, confidence, entityType` (LogMap's own symbols:
relations `<`, `>`, `=`; entity types `CLS`, `DPROP`, `OPROP`, `INST`, `UNKNO`), canonically
sorted so JVM hash-set iteration order never leaks into persisted artifacts.

**Step 2 — prompt build** (`pipeline/stage_two.py`, subprocess). Loads both ontologies through
the [ontology access layer](ontology.md) (`logmap_llm/ontology/`), then renders one user prompt
per M_ask candidate using the per-lane templates in `logmap_llm/oracle/prompts/` — separate
templates for classes, object properties, data properties and instances, routed by LogMap's
entityType tag. When `few_shot.few_shot_k > 0` it also builds per-query few-shot demonstrations
via `pipeline/rag_fewshot.py` and the retriever in `logmap_llm/oracle/rag/` (see [RAG few-shot
retrieval](rag.md)): high-confidence LogMap anchors become pseudo-labelled positives, with
constructed negatives.

**Step 3 — consult** (`logmap_llm/oracle/consultation.py`, in-process; see [the LLM
oracle](oracle.md)). Fans the prompts out
over a thread pool to any OpenAI-compatible chat-completions endpoint (OpenRouter, vLLM,
SGLang, OpenAI), with transient-error retry and a failure-abort circuit breaker. Each verdict
lands as an `Oracle_prediction` value (`True`, `False`, `'error'` or `'skipped'`) alongside a
logprobs-derived `Oracle_confidence` and token counts, appended to the M_ask DataFrame. In
bidirectional mode each candidate is asked in both directions and equivalence requires both
answers to be True.

**Step 4 — refine** (`pipeline/orchestration.py::refine_alignment`). Two strategies, selected
by `pipeline.refinement_strategy`. The default, `logmap`, hands the accepted predictions back
to LogMap's Java refinement, which resolves logical conflicts. The alternative, `python`, is a
set-union approximation computed by `_kg_refine_in_python`:

    refined = { initial - M_ask } ∪ { m ∈ M_ask : oracle(m) = True }

It exists as a bypass for LogMap builds that crash on instance mappings during Java refinement
(the bundled LogMap is patched; unpatched builds are not).

**Step 5 — evaluate** (`logmap_llm/evaluation/harness.py`, subprocess). Runs only when
`evaluation.evaluate = true`. Scores the refined alignment against a reference (complete or
OAEI KG-track partial gold standard) and the oracle as a binary classifier, through a pluggable
engine (pure-Python custom, partial-reference, or DeepOnto). See [evaluation](evaluation.md).

**Reporting.** The runner prints a timing summary and the experimental parameters, then
`pipeline/reporting.py::write_results_file` validates that every non-bypassed stage published
its required artifact and atomically writes `run_result.json` — the canonical machine-readable
record (schema_version 1, status, redacted oracle parameters, prediction counts, timing,
evaluation results, and a sha256-checksummed artifact list). If the evaluation subprocess
failed, the run exits 1 and no `run_result.json` is written; the bootstrap also unlinks any
previous one, so a stale success marker can never survive a failed retry.

## Artifacts on disk

`pipeline/paths.py::PipelinePaths` resolves every path from three configured directories
(`[outputs]` in the TOML), or from fixed subdirectories `logmapllm-outputs`,
`logmap-initial-alignment` and `logmap-refined-alignment` under `--run-root DIR` when given.
With `{task}` = `alignmentTask.task_name` and `{tpl}` = `prompts.cls_usr_prompt_template_name`:

- `initial_dir/{task}-logmap_mappings.txt` — full initial alignment; headerless,
  pipe-separated, five columns (`src|tgt|relation|confidence|entityType`).
- `initial_dir/{task}-logmap_mappings_to_ask_oracle_user_llm.txt` — M_ask, same format.
- `output_dir/{task}-{tpl}-mappings_to_ask_oracle_user_prompts.json` — prompts keyed
  `src_uri|tgt_uri` (plus `src_uri|tgt_uri|REVERSE` in bidirectional runs).
- `output_dir/{task}-{tpl}-few_shot_examples.json` — per-query `[user, assistant]` example
  pairs; `rag_traces.json` carries retrieval provenance, with `rag_fallback.json` and
  `rag_negative_fallback.json` recording any degradation.
- `output_dir/{task}-{tpl}-mappings_to_ask_with_oracle_predictions.csv` — M_ask plus the
  `Oracle_*` columns.
- `refined_dir/{task}-logmap_mappings.tsv` — refined alignment; headerless, tab-separated.
- `output_dir/evaluation_results.json` and `output_dir/run_result.json`.
- `output_dir/pipeline_log_{timestamp}.txt` and `subprocess_*_{timestamp}.txt` — tee'd console
  logs (timestamped, so retries accumulate; the JSON artifacts are overwritten per run).

Note that the initial and refined TSVs share the filename `{task}-logmap_mappings.tsv` and are
distinguished only by directory. The prompt-build, consult, refine and evaluate stages each
delete their output before regenerating it, and
the strict artifacts are published with tmp-fsync-rename writes (`logmap_llm/utils/io.py`), so
a crash mid-write cannot leave a truncated file that a later `reuse` mode would accept.

## Content-addressed run ids

`pipeline/paths.py::compute_run_id` derives a 16-hex-character sha256 from a sorted-key JSON
dump of the semantically identifying config fields: task and ontology paths, oracle model,
sampling and request parameters, every prompt template name, the sibling strategy, and the
few-shot / RAG encoder settings. A prebuilt few-shot bundle contributes the sha256 of its file contents,
not its path. The point is that two runs collide on a path only when they are the same
experiment: identical configs map to the same directory, distinct configs never do, which makes
parallel grids under a shared output directory safe. Performance-only settings (worker counts)
and secrets are excluded. In a plain `python -m logmap_llm` run the id is computed but the
`{run_id}` subdirectory is not applied — `PipelinePaths` nests artifacts under it only when
constructed with `isolate_run=True`. The [experiments harness](experiments.md) instead isolates
each job with a per-attempt `--run-root`.

## The mediation arm

`pipeline/mediation.py` holds an experimental channel that composed extra candidate mappings
via third "mediating" ontologies, filtered them by vote-count support, had the oracle accept or
reject them, and unioned the accepted ones into the initial alignment as a pure
recall-expansion channel. It is quarantined: `run_mediation_arm` raises
`MediationQuarantinedError` as its first statement, before any I/O, because the historical
mediation evidence is reference-contaminated and not a valid experimental condition. Everything
after the raise is retained historical implementation that never executes, and `MediationConfig`
is deliberately not part of the baseline config schema. `pipeline/mediation_isolation.py` is a
harness proving the module's mere presence is inert — the standard pipeline's artifacts are
byte-identical whether or not mediation is imported, and its `--mode mediation` refuses with
exit code 78 without importing the module. Any future mediation appendix would need fresh,
provenance-disjoint inputs behind a separate entry point.
