# Experiments

A single `python -m logmap_llm` invocation is one run. The `logmap_llm.experiments` package is the campaign layer above that: it expands one `batch.toml` spec into a grid of runs (tasks × models × conditions × repeats), freezes the grid as an immutable batch directory of ordinary standalone [pipeline](pipeline.md) configs, executes each cell as a child `python -m logmap_llm` process, and aggregates exactly one batch into CSV/Markdown summaries. Everything is local and deterministic; there is no scheduler or database. The code lives in `logmap_llm/experiments/` — `plan.py` (spec validation and freezing), `run.py` (execution and completion validation), `aggregate.py` (aggregation), `strata.py` and `aggregate_strata.py` (post-hoc stratification), `import_alignment.py` and `environment.py`.

The CLI is `python -m logmap_llm.experiments` (argparse names itself `logmap-llm-batch`, but no console script is installed in this checkout). Exit codes: 0 on success, 2 on any error, 130 on interrupt.

## Writing a batch spec

A spec is a TOML file with the top-level keys `schema` (must be 1), `[batch]`, `[defaults]`, `[tasks.<id>]`, `[models.<id>]`, `[conditions.<id>]`, `[matrix]`, and `[aggregate]`. Each axis entry carries a `config` table holding a partial overlay of the native [configuration](configuration.md) schema (`LogMapLLMConfig`); per matrix cell the planner deep-merges `defaults`, then the task, model, and condition overlays, and rejects axes that set the same key to different values so no overlay silently wins. The `[outputs]` section is harness-managed and may never appear in a spec. Model entries additionally accept `max_parallel_runs` and `verify_endpoint`.

| Key | Meaning |
| --- | --- |
| `batch.name` | Base name; the batch directory becomes `{output_root}/{name}-{plan_hash[:12]}` |
| `batch.output_root` | Parent directory; relative paths resolve against the spec's directory |
| `batch.jobs` | Default worker-thread count for `run` (default 1) |
| `batch.timeout_seconds` | Required per-job child-process timeout |
| `batch.reuse_alignments` | Share one initial-alignment run across a task's jobs with identical `[alignmentTask]` inputs (default true) |
| `matrix.tasks` / `models` / `conditions` | Non-empty arrays of defined axis ids |
| `matrix.repeats` | Repeats per cell (default 1) |
| `matrix.exclude` | Inline tables naming task + model + condition (optional 0-based `repeat`) to drop |
| `models.<id>.max_parallel_runs` | Per-model concurrency cap during `run` |
| `models.<id>.verify_endpoint` | Probe `{oracle.base_url}/models` for the served model before running |
| `aggregate.group_by` | Grouping columns, unique names from task/model/condition/repeat (default the first three) |
| `aggregate.require_complete` | Fail `aggregate` on incomplete jobs unless `--allow-incomplete` (default true) |

Two constraints matter in practice. Secrets are never frozen: `oracle.api_key` must be the literal `EMPTY` or an `ENV:VARIABLE` reference, resolved at run time. And every input path (ontologies, LogMap parameters directory, reference alignments, prebuilt few-shot bundle, …) is resolved relative to the spec's directory and fingerprinted (sha256 plus byte size; directories also file count) at generation, so a batch refuses to run if an input changed underneath it.

## Generating a batch

```
python -m logmap_llm.experiments generate path/to/batch.toml [--dry-run]
```

`--dry-run` validates and prints the full plan without writing anything. Otherwise `generate_batch` atomically freezes the batch directory:

```
{output_root}/{name}-{plan_hash[:12]}/
  batch.toml  manifest.json  plan.tsv
  configs/{job_id}.toml
  alignments/{alignment_id}/config.toml  alignments/{alignment_id}/attempts/
  jobs/{job_id}/attempts/
```

Each `configs/{job_id}.toml` is a complete, standalone pipeline config — you can run any one by hand with `python -m logmap_llm --config …`. Job ids are content-addressed (`{task}-{model}-{condition}-r{repeat}-{execution_hash[:8]}`) so that identity follows from what the job *is*, not when it was planned: `condition_id` hashes only the scientific configuration (it deliberately drops operational knobs such as `oracle.api_key`, `oracle.base_url`, `oracle.max_workers`, `few_shot.rag_cache_dir`, and the JVM heap, so the same science on different infrastructure compares equal), while `execution_hash` adds the repeat index, timeout, and concurrency settings. The manifest is integrity-sealed: any hand edit to `manifest.json` after generation fails with a seal mismatch, and regenerating from the same spec into the same directory is a harmless no-op.

## Running and inspecting a batch

```
python -m logmap_llm.experiments run BATCH_DIR [--jobs N] [--resume] \
    [--select KEY=A,B]... [--limit N]
python -m logmap_llm.experiments status BATCH_DIR [--select KEY=A,B]...
```

Selector keys are `id`, `task`, `model`, `condition`, and `repeat`; each key may appear at most once and takes a comma-separated value list, e.g. `--select task=cmt-conf,edas-ekaw --select repeat=0`. `--limit` caps the selection in plan order.

`run_batch` takes an exclusive lock on `{batch}/.run.lock` (one runner per batch) and refuses to start if the installed `logmap_llm` source no longer hashes to the manifest's `core_sha256` — results must come from the code the batch was planned against. When `batch.reuse_alignments` is on, one classical-LogMap alignment run per task's distinct `[alignmentTask]` executes serially first; dependent jobs then receive a copy of its published artifacts in their run root and are spawned with `--reuse-align`. Jobs run on a thread pool (`batch.jobs` workers unless `--jobs`), interleaved across models and throttled by each model's `max_parallel_runs` semaphore.

Every execution is a numbered attempt under `jobs/{id}/attempts/0001/…` containing the config snapshot, `run.log`, a live `status.json`, and the pipeline's `run-root/`. The child command is exactly

```
python -m logmap_llm --config <attempt>/config.toml --run-root <attempt>/run-root [--reuse-align]
```

run in its own process group; on timeout the whole group is terminated. On success the runner verifies that every artifact the config implies actually exists — `run_result.json`, the LogMap mapping files, prompt/prediction JSON and CSV, the refined-alignment TSV, `evaluation_results.json` when [evaluation](evaluation.md) ran — and only then writes `complete.json` recording each artifact's sha256 and size. Completions are never trusted from status alone: every later read re-hashes the artifacts. Failed or timed-out jobs need `--resume`, which skips verified successes and opens fresh attempts for the rest. `status` prints one TSV row per job with states such as `success`, `degraded`, `failed`, `timed_out`, `not_run`, and `invalid_artifacts`.

## Aggregating results

```
python -m logmap_llm.experiments aggregate BATCH_DIR [--allow-incomplete]
```

`aggregate_batch` re-validates every job's latest attempt and writes three files into `{batch}/aggregate/`: `jobs.csv` (one row per job: identity columns, flattened `run.*` from `run_result.json`, `metric.*` from `evaluation_results.json`, and `strata.*` provenance), `summary.csv` (one row per `aggregate.group_by` group), and a human-readable `summary.md`. Per group it reports coverage (`expected`, `completed`, `success`, `degraded`, `failed`, `not_run`), per-metric means with contributor counts, and pooled precision/recall/F1 recomputed from summed raw confusion counts rather than averaged ratios — alignment TP/FP/FN are additive across tasks, whereas averaging per-task F1 is not the pooled F1. Undefined ratios stay empty with an explanatory `metric_notes` entry instead of being coerced to 0. Failures are never hidden: with `require_complete` (the default) an incomplete batch still gets all three files written, then the command exits 2 unless `--allow-incomplete`.

## Stratified aggregation

When a job's config sets `evaluation.stratified_class_property` (conference-style lanes `m1_class` / `m2_property` / `unknown`) or `evaluation.stratified_by_entity_type` (KG-style lanes `class` / `property` / `instance` / `unknown` — the two flags are mutually exclusive), per-lane metrics are not taken from the evaluator. Instead `aggregate_strata.py` discards the evaluator's per-lane blocks and recomputes strata post hoc from the sealed artifacts, recording the hash of every file it reads as `strata.*` provenance so each lane figure stays auditable.

Lane membership of the reference comes from sidecar TSVs: conference mode expects `<reference stem>_class.tsv` and `<reference stem>_property.tsv` next to the reference alignment; KG mode expects `reference_class.tsv`, `reference_property.tsv`, and `reference_instance.tsv` in the reference's directory. Sidecars must be mutually disjoint subsets of the full reference; leftover pairs fall into the `unknown` lane. System-side and oracle-side lanes come from LogMap's own entity-type tags in the sealed mapping files. The pure computation in `strata.py` then requires exact reconciliation — per-lane counts must sum to the sealed global and oracle confusion counts for every field — and a job whose strata do not reconcile is marked `invalid_artifacts` rather than silently included.

## Importing an external alignment

Sometimes the initial classical alignment is computed elsewhere (a different host, an earlier campaign). `import-alignment` installs it into a fresh batch as if the harness had produced it:

```
python -m logmap_llm.experiments import-alignment BATCH_DIR \
    (--task TASK | --alignment-id ID) \
    --source-dir DIR --receipt receipt.json [--source-task-prefix PREFIX]
```

The receipt is a schema-1 JSON document of kind `logmap-llm-alignment-transfer` that binds the transfer to this exact batch (batch id and the sha256 of its `manifest.json`) and declares each source file's hash and size. `import_alignment` verifies all of that, checks the alignment slot is untouched and no dependent job has run, copies the files (renaming the three core LogMap mapping files from the producer's task prefix to the destination task name), and publishes a synthetic completed attempt with full import provenance. Any failure rolls the whole import back. Afterwards `run` consumes the imported alignment exactly like a locally produced one.

## Environment record

`environment.py` records the resolved versions of the packages that can plausibly change a result (JPype1, numpy, openai, owlready2, pandas, pydantic, rdflib, torch, transformers, matplotlib) as a sealed `environment` key in the manifest. It is deliberately excluded from `condition_id`, `execution_hash`, job ids, and the plan hash, so different virtualenvs plan byte-identical job identities; regenerating an existing batch from another environment keeps the original record and prints which tracked packages now differ. It is a provenance note, not a gate — nothing blocks a run on a version mismatch.
