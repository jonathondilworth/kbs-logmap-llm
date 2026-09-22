# Running the pipeline

A pipeline run performs one matching task end to end: LogMap aligns the two ontologies and hands back the uncertain candidate subset M_ask, the pipeline builds an LLM prompt for each candidate, an LLM oracle answers True or False per candidate, the verdicts are merged back into the alignment, and the result is optionally scored against a reference. The driver is `logmap_llm/pipeline/runner.py`; the per-step logic lives in `logmap_llm/pipeline/orchestration.py`.

```bash
python -m logmap_llm --config configs/my_task.toml
```

The config file is a TOML document covering the alignment task, oracle, prompts, few-shot settings, outputs, pipeline modes, and evaluation — see the [configuration reference](configuration.md). The CLI flags are deliberately few; almost everything is configured in the file:

- `--config PATH` (or `-c`) — path to the TOML config. Defaults to `configs/default_config.toml`.
- `--reuse-align` — override `pipeline.align_ontologies` to `"reuse"`, skipping the LogMap alignment and reading the previous run's alignment files instead.
- `--reuse-prompts` — override `pipeline.build_oracle_prompts` to `"reuse"`; implies `--reuse-align`, since prompts only make sense against the alignment they were built from.
- `--no-cache` — disable owlready2 quadstore caching, so ontologies are parsed from scratch. Forwarded to the subprocess stages.
- `--run-root DIR` — root all outputs under `DIR`, creating `logmapllm-outputs`, `logmap-initial-alignment`, and `logmap-refined-alignment` subdirectories there instead of using the three `[outputs]` directories from the config. The [experiments harness](experiments.md) uses this to give each job its own directory.

`runner.main()` loads and validates the config, resolves all artifact paths through `PipelinePaths` (`pipeline/paths.py`), starts the JVM and LogMap only if a step actually needs them, and tees everything printed to the console into a timestamped log file. It then runs the five steps in a fixed order, prints timing and parameter summaries, and publishes `run_result.json`.

## The five steps

**Step 1 — align** (`pipeline.align_ontologies`: `align` | `reuse` | `external` | `bypass`, default `align`). Runs LogMap in-process over JPype (`logmap_llm/interface.py`). LogMap writes the full initial alignment and the M_ask subset into the initial-alignment directory. The LogMap bundle (jar, `java-dependencies/`, `parameters.txt`) is expected at `./logmap` under the working directory unless `alignmentTask.logmap_parameters_dirpath` points elsewhere; the JVM heap comes from `alignmentTask.logmap_jvm_memory`. In `reuse` mode the step reads the mappings and M_ask files a previous run left behind — the JVM is not started at all if nothing else needs it. In `external` mode (annotate mode, 22 Sep 2026) the file named by `alignmentTask.external_mappings_filepath` — LogMap pipe `.txt`, TSV, or OAEI Alignment RDF — becomes the M_ask and stands in for the initial alignment: `pipeline/annotate.py` publishes its rows (relation, confidence and entity type kept when present, else `=`, `1.0`, `CLS`; duplicates by pair collapsed) as `{task}-logmap_mappings.{txt,tsv}` and `{task}-logmap_mappings_to_ask_oracle_user_llm.txt`, so every later step finds what a LogMap alignment would have left. No LogMap runs; refinement stays available (`python` strategy = external ∪ accepted; `logmap` strategy = a full LogMap run with the verdicts as local oracle).

**Step 2 — prompt build** (`pipeline.build_oracle_prompts`: `build` | `reuse` | `bypass`, default `build`). Spawns `python -m logmap_llm.pipeline.stage_two` as a subprocess and waits for it to publish the prompts JSON (and few-shot artifacts, when enabled). It runs out of process because prompt building needs owlready2, which cannot coexist with JPype's JVM in one process. In `reuse` mode the step loads the existing prompts JSON and validates its keys against the configured class template — a prompts file built for a different direction mode is rejected rather than silently reinterpreted. An empty prompts artifact for a non-empty M_ask is fatal in both modes: the pipeline refuses to degrade into an undeclared zero-prompt baseline.

**Step 3 — consult oracle** (`pipeline.consult_oracle`: `consult` | `reuse` | `local` | `bypass`, default `consult`). Sends each prompt to the LLM over an OpenAI-compatible API with threaded fan-out and retry — see [the LLM oracle](oracle.md) — and appends `Oracle_prediction`, `Oracle_confidence`, `Oracle_input_tokens`, and `Oracle_output_tokens` columns to M_ask, written as a CSV. If `few_shot.few_shot_k > 0`, the few-shot examples artifact from stage two must exist and cover every prompt key exactly, otherwise the run aborts. `reuse` reads a previous predictions CSV back; `local` defers to LogMap's file-based local oracle in `oracle.local_oracle_predictions_dirpath` (the directory is handed over with a trailing separator and its verdicts are counted first — a directory without any verdict is fatal; see [evaluation](evaluation.md)).

**Annotated M_ask (after Step 3).** Every run whose Step 3 yields a predictions DataFrame (`consult` or `reuse`) also writes `{task}-{template}-annotated.txt` and `.tsv` next to the predictions CSV: the M_ask rows in file order with the verdict appended (`LLM_annotation` ∈ `True` / `False` / `ERROR` / `SKIPPED`; bidirectional runs add `LLM_forward_subsumption` and `LLM_reverse_subsumption`; the TSV has a header and a last `LLM_confidence` column). `LLM_confidence` is not something the model is asked for: it is the logprob-derived `Oracle_confidence` (`oracle/consultation.py::calculate_logprobs_confidence`, the probability the model assigned to the answer token it produced; `nan` when the endpoint returned no logprobs — see the docstring of `pipeline/annotate.py`). With `pipeline.stop_after_consultation = true` the run ends here: Steps 4 and 5 are skipped, `run_result.json` records `stopped_after = "consultation"`, and the batch harness accepts the run without a refined alignment; the schema requires `evaluation.evaluate = false` in that case. Worked example, Ernesto's use case (validate composed LogMapBio mappings with the committed oracle):

```toml
[alignmentTask]
task_name = "mouse-human"
onto_source_filepath = "OAEI/anatomy/mouse-human/source.owl"   # the prompt builder reads the ontologies
onto_target_filepath = "OAEI/anatomy/mouse-human/target.owl"
ontology_domain = "biomedical anatomy"
external_mappings_filepath = "composed/all-composed-minus-llm-default.txt"

[oracle]      # the committed OAEI 2026 configuration (configs/oracle_configuration.toml)
[prompts]     # class_equivalence / one_level_of_parents_and_synonyms

[pipeline]
align_ontologies = "external"
build_oracle_prompts = "build"
consult_oracle = "consult"
stop_after_consultation = true

[evaluation]
evaluate = false
```

The output `mouse-human-one_level_of_parents_and_synonyms-annotated.tsv` matches the delivered `all-composed-minus-llm-default.annotated.tsv` except for verdicts the oracle answers differently on another day (temperature 0 gave 259/259 identical verdicts across runs) and the added confidence column.

**Step 4 — refine alignment** (`pipeline.refine_alignment`: `refine` | `bypass`, default `refine`). Merges verdicts into the final alignment. With `pipeline.refinement_strategy = "logmap"` (the default), accepted mappings are converted back to Java objects (`logmap_llm/bridging.py`) and LogMap's refinement resolves logical conflicts. With `"python"`, the pipeline computes the set union directly: refined = (initial − M_ask) ∪ {m ∈ M_ask : oracle said True}. The Python strategy exists as a bypass for LogMap builds that crash on instance mappings during Java refinement (the bundled LogMap is patched). If there are no predictions at all, the initial alignment TSV is copied to the refined directory unchanged.

**Step 5 — evaluate**. Gated by `evaluation.evaluate` (default `false`; when `true` the schema requires `evaluation.reference_alignment_path`). Runs `python -m logmap_llm.evaluation.harness` as a subprocess, again for JVM isolation — the DeepOnto backend starts its own JVM (it runs only when `evaluation.force_custom_eval = false`; the default `true` selects the JVM-free custom engine). The resulting `evaluation_results.json` is trusted only if the subprocess exits 0 and the file exists. See [evaluation](evaluation.md).

Each regenerating step first deletes its own output artifact, and the predictions CSV, the Python-strategy refined TSV, and `run_result.json` are written atomically, so a crash mid-write cannot leave a truncated file that a later `--reuse` run accepts. Cross-step consistency is enforced at config-validation time: `build_oracle_prompts = "reuse"` requires `align_ontologies = "reuse"`, and consulting requires prompts to have been built or reused.

## Output layout

Three directories hold everything; with `--run-root DIR` they are `DIR/logmapllm-outputs`, `DIR/logmap-initial-alignment`, and `DIR/logmap-refined-alignment`, otherwise the paths named in the `[outputs]` config section. Artifact filenames are prefixed with the task name and (for LLM-side artifacts) the class prompt template name, so several tasks can share a directory:

| File | Directory | Contents |
| --- | --- | --- |
| `{task}-logmap_mappings.txt` | initial | Full initial alignment, pipe-separated: `source_uri\|target_uri\|relation\|confidence\|entityType` |
| `{task}-logmap_mappings_to_ask_oracle_user_llm.txt` | initial | M_ask, same format |
| `{task}-{template}-mappings_to_ask_oracle_user_prompts.json` | outputs | Per-candidate prompts keyed `src_uri\|tgt_uri` (plus `...\|REVERSE` for bidirectional templates) |
| `{task}-{template}-few_shot_examples.json` | outputs | Per-query `[user, assistant]` demonstration pairs (when `few_shot_k > 0`) |
| `{task}-{template}-mappings_to_ask_with_oracle_predictions.csv` | outputs | M_ask plus the `Oracle_*` verdict columns |
| `{task}-{template}-annotated.txt` / `.tsv` | outputs | M_ask rows with the verdicts appended (annotate mode; every consult/reuse run) |
| `{task}-logmap_mappings.tsv` | refined | Final refined alignment, tab-separated, headerless |
| `evaluation_results.json` | outputs | Metrics from the evaluation harness |
| `pipeline_log_{timestamp}.txt` | outputs | Teed console log, ANSI-stripped |
| `run_result.json` | outputs | Canonical machine-readable run summary |

Note the refined alignment reuses the filename `{task}-logmap_mappings.tsv` — only the directory distinguishes it from the initial TSV.

Every run also computes a content-addressed run id: a 16-hex-character sha256 over the semantically identifying config fields (task, ontologies, model and sampling parameters, prompt template names, few-shot and RAG encoder settings — see `compute_run_id` in `pipeline/paths.py`). Identical experiments therefore collide on the same id and distinct configs never do. A plain `python -m logmap_llm` run does not use it for isolation; only a parallel scheduler passing `isolate_run=True` to `PipelinePaths` roots the three directories under a per-run-id subdirectory.

## Stage two: prompts and few-shot examples

`pipeline/stage_two.py` reads the initial alignment and M_ask, dedupes M_ask on the (source, target) URI pair, loads both ontologies through the [ontology access layer](ontology.md), and renders one prompt per candidate. LogMap's `entityType` tag routes each candidate to a template lane: class, object-property, data-property, or instance, named by `prompts.cls_usr_prompt_template_name` and its `prop`/`dprop`/`inst` counterparts (a data-property candidate falls back to the object-property template when no dedicated one is configured). If the class template is registered as bidirectional, each candidate produces a forward and a `|REVERSE` prompt, and the oracle's answers are later combined with logical AND.

Few-shot behaviour is controlled by the `[few_shot]` section. `few_shot_k = 0` (the default) disables demonstrations entirely. Otherwise `few_shot_negative_strategy` picks how examples are chosen:

- `hard` / `static-hard` — query-agnostic example selection with near-miss (recombination) negatives (the default, kept for the ablation);
- `random` / `static-random` — query-agnostic selection with random (column-swap) negatives;
- `hard-similar` / `query-rag` — query-specific RAG retrieval: per candidate, the most similar high-confidence LogMap anchor mappings become positive demonstrations and constructed near-misses become negatives (see [RAG few-shot retrieval](rag.md));
- `zero-shot` — no examples.

The default `hard` is query-agnostic; query-specific retrieval must be requested explicitly. When `few_shot_k > 0` you must also set `few_shot.rag_negative_layout` (`paired-sibling-v2`, `paired-donor-v2`, or `donor-cross-v1`) — this is checked at runtime in stage two rather than by the schema so that frozen historical configs remain loadable. `few_shot.rag_failure_policy` decides what happens if retrieval fails: `error` (default) aborts the run, `record_zero_shot` records the degradation in `rag_fallback.json` and continues zero-shot. Setting `few_shot.prebuilt_few_shot_bundle_path` instead loads a sealed prebuilt bundle, revalidated strictly against the current task before use.

Alongside the examples, stage two writes `rag_traces.json` (per-query retrieval provenance) and, when applicable, `rag_fallback.json` and `rag_negative_fallback.json` next to the other output artifacts.

## When a run finishes

The console log ends with a timing summary (Step 6), an experimental-parameters report (Step 7), and the path of the results file (Step 8). Look at:

- `run_result.json` — the canonical summary: `schema_version` 1, a `status` of `succeeded` or `degraded`, the requested parameters, redacted oracle settings, verdict counts (true/false/error/skipped), timing, evaluation metrics, and a list of artifacts with paths, sizes, and sha256 digests. It is written only on a fully successful run; a failed evaluation subprocess exits 1 without writing it, and any stale copy is deleted at bootstrap, so its presence alone certifies success. `status` is `degraded` when a RAG fallback was recorded or any oracle verdict is `error`/`skipped` — details land under the `degradation` key.
- `evaluation_results.json` — precision/recall/F1 and oracle-classifier metrics when evaluation ran.
- `{task}-{template}-mappings_to_ask_with_oracle_predictions.csv` — the per-candidate verdicts, useful for error analysis.
- `pipeline_log_{timestamp}.txt` — the full console log; retries in the same directory accumulate one log per attempt. Stage two and the evaluation harness write their own `subprocess_*` logs beside it.
- `rag_fallback.json` / `rag_negative_fallback.json` — present only when RAG retrieval degraded or negative construction fell back; their absence is the healthy case.
