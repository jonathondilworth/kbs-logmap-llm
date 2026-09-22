# Evaluation

The `logmap_llm.evaluation` package answers two separate questions at the end of a run: how good is the refined alignment against a reference alignment (global P/R/F1), and how good is the LLM oracle as a binary classifier over the uncertain M_ask candidates it was asked about (oracle discrimination). Both are computed by `logmap_llm/evaluation/harness.py` and written to a single versioned JSON artifact.

## Harness lifecycle

The [pipeline](pipeline.md)'s Step 5 (`pipeline/orchestration.py::evaluate`) does not evaluate in-process. It spawns

```
python -m logmap_llm.evaluation.harness --config <config.toml> [--run-root DIR]
```

as a subprocess, because the DeepOnto engine starts its own JVM and must not share a process with the pipeline's LogMap JVM. Before launching, the parent deletes any stale `evaluation_results.json`; afterwards it only trusts the result file if the child exited 0, so a failed or incomplete evaluation can never leak a stale result into the run. The whole step is skipped when `evaluation.evaluate = false` (the default) in the [configuration](configuration.md).

In config mode the harness runs `subprocess_bootstrap("EVALUATE", ...)` (validated config, `PipelinePaths`, tee'd subprocess log) and derives every path itself: the system alignment from `run_paths.refined_mappings_tsv()` (`<refined_dir>/{task}-logmap_mappings.tsv`), the oracle predictions from `run_paths.predictions_csv()`, and the output from `run_paths.eval_json()` (`<output_dir>/evaluation_results.json`).

There is also an explicit-paths mode for ad-hoc scoring, with no config bootstrap and no subprocess log:

```
python -m logmap_llm.evaluation.harness \
  --system SYSTEM.tsv --reference REFERENCE.tsv \
  --oracle-predictions PREDICTIONS.csv --metrics global,oracle --output OUT.json
```

Stratification flags and `jvm_memory` are config-mode only. `--test-cands` and `--no-cache` are parsed but currently unused (ranking metrics are an unimplemented stub — `compute_ranking` returns `None`). The programmatic entry point is `evaluation.harness.evaluate_alignment(...)`, which validates inputs up front: a missing system, reference, or train-reference file, or missing predictions when `oracle` metrics are requested, raises `FileNotFoundError` rather than silently degrading.

Which metric blocks run is controlled by `evaluation.metrics` (default `["global", "oracle"]`; a comma-separated string is accepted). Only `global` and `oracle` are valid names.

## Global metrics

`evaluation/metrics.py::compute_prf` scores the system alignment as a set of `(source URI, target URI)` pairs against the reference set, reporting `precision`, `recall`, `f1`, `true_positives`, `false_positives`, `false_negatives`, `system_size`, and `reference_size`. Undefined ratios (for example precision over an empty system alignment) are reported as `None` with a reason under `metric_notes` — never coerced to a measured `0.0`. When `evaluation.train_alignment_path` is set, the custom engine subtracts the train pairs from both the system and the reference before scoring (the DeepOnto engine passes them as `null_reference_mappings` instead); the partial-reference engine rejects a train reference outright, since a partial gold standard does not split into train and test.

## Oracle discrimination metrics

`compute_oracle_metrics` treats the oracle as a diagnostic test over the M_ask candidates: a prediction of True for a pair in the reference is a TP, True for a pair outside it an FP, and so on. It reports `sensitivity`, `specificity`, and `youdens_j` (sensitivity + specificity − 1) alongside `oracle_precision`/`oracle_recall`/`oracle_f1` and the raw counts. Predictions that are not exactly True/False after normalisation (`'error'`, `'skipped'`, NaN, a failed consultation) land in an `errors` bucket; `total_candidates` always reconciles as tp + fp + tn + fn + `oracle_excluded`, where `oracle_excluded = errors + partial_scope_excluded`.

One deliberate asymmetry: sensitivity, specificity, and Youden's J become `None` with a `metric_notes` reason when undefined, but `oracle_precision`/`oracle_recall`/`oracle_f1` report a measured `0.0` on zero denominators. The artifact contract (below) enforces exactly this split.

## Stratified evaluation

Two mutually exclusive stratification modes exist (the config schema rejects enabling both):

- `evaluation.stratified_by_entity_type = true` splits the global metrics into class/property/instance strata. If per-type reference files `reference_class.tsv`, `reference_property.tsv`, `reference_instance.tsv` exist next to the reference alignment they are used directly; otherwise the reference is partitioned by URI convention (`classify_uri_entity_type` matches `/class/`, `/property/`, `/resource/` — DBkWik conventions). The system side is always partitioned by URI convention, since a system mapping carries no gold typing. Strata empty on both sides are omitted rather than reported as zeros. Requesting this on an engine without `supports("stratified_global")` — in practice only the DeepOnto engine — raises `RuntimeError` instead of quietly running an unstratified evaluation.
- `evaluation.stratified_class_property = true` computes the OAEI conference track's M1 (class-only) and M2 (property-only) breakdown via `utils/misc.py::compute_conference_m1_m2_stratified`. Per-type references follow the conference naming convention (`{ref_stem}_class.tsv`, `{ref_stem}_property.tsv` beside the full reference). Pairs are typed using the `entityType` column of LogMap's initial alignment (`<initial_dir>/{task}-logmap_mappings.txt`, pipe format); because a missing initial alignment would silently degrade both buckets to the full pair set, the harness raises `FileNotFoundError` when it is absent.

Stratified blocks appear in the results JSON as `global_class`/`global_property`/`global_instance` or `global_m1_class`/`global_m2_property`.

## Complete vs partial gold standards

The default semantics assume a complete gold standard: any system pair outside the reference is a false positive. The OAEI knowledge-graph track publishes partial gold standards, so `compute_kg_partial_prf` implements the track's rule instead: a system pair not in the reference only counts as an FP if its source is one of the reference's source entities or its target one of the reference's target entities; pairs whose entities both fall outside the reference's scope are ignored, and FN = |reference| − TP. The result carries the extra keys `ignored` and `evaluated_size` and `source: "kg_partial"`. The same scope rule applies to the oracle side: with a partial reference, predictions whose source and target are both outside scope are counted as `partial_scope_excluded` rather than scored.

## Engines

`evaluation/engines/base.py` defines the `EvaluationEngine` ABC (`compute_global`, `compute_oracle`, optional `compute_stratified_global`). `harness.select_engine` picks one from the config flags:

```python
if partial_reference: return PartialReferenceEvaluationEngine()
if force_custom: return CustomEvaluationEngine()
_ensure_jvm_memory(jvm_memory, overwrite=True)
if DeepOntoEvaluationEngine.is_available(): return DeepOntoEvaluationEngine()
return CustomEvaluationEngine()
```

- `engines/custom.py` — pure Python, no JVM; set-based `compute_prf`, supports entity-type stratification. Because `evaluation.force_custom_eval` defaults to `true`, this is the engine a default config uses; DeepOnto is opt-in.
- `engines/deeponto.py` — delegates global P/R/F1 to DeepOnto's `AlignmentEvaluator.f1`, converting inputs to DeepOnto TSV first. It exports `JAVA_MEMORY`/`DEEPONTO_JVM_MEMORY`/`JVM_MEMORY` from `evaluation.jvm_memory` and pre-starts the JVM itself, because DeepOnto 0.9.3 otherwise sizes its heap via an interactive prompt that hangs non-interactive harnesses. It recomputes tp/fp/fn set-based itself, refuses partial references, and does not support stratification.
- `engines/partial_reference.py` — hardcodes the KG-track partial semantics for both global (`compute_kg_partial_prf`) and oracle metrics, supports stratification (the ignored/FP rule then applies per stratum), and rejects a train reference.
- `engines/logmap_oaei.py` and `engines/bioml.py` — the track-faithful engines described below; never primary, no stratification, constructed from their `[evaluation.<engine>]` option table by `engines.build_engine`.

Note that the harness never passes `partial_reference` into engine options — partial semantics work solely because `select_engine` returns the partial-reference engine. Calling `CustomEvaluationEngine` directly requires `options["partial_reference"]=True` yourself.

## Files read and written

Inputs:

- System alignment TSV — the refined alignment; DeepOnto format (`SrcEntity\tTgtEntity\tScore` header) or headerless OAEI TSV, auto-detected by `evaluation/io.py::load_mapping_pairs`. Extra columns are ignored; a non-empty file that parses to zero pairs raises `ValueError` (usually a wrong separator — pipe-delimited LogMap output needs `sep='|'`).
- Reference alignment TSV (`evaluation.reference_alignment_path`) — same auto-detected formats.
- Oracle predictions CSV — requires columns `source_entity_uri`, `target_entity_uri`, `Oracle_prediction` (bool or `true`/`yes`/`false`/`no`/`error`/`skipped`), optionally `Oracle_confidence`; other columns are preserved for analysis but ignored here.
- Optional: train reference TSV, LogMap initial alignment (pipe format `source|target|relation|confidence|entityType`, M1/M2 typing only), per-type reference files as described above.

Outputs:

- `evaluation_results.json` — top-level keys `schema_version` (1), `task_name`, `engine`, `metrics`, then `global`, `oracle`, any `global_<stratum>` blocks and, with `evaluation.engines`, the `engines` list and the `global_<engine>` / `oracle_<engine>` blocks. Written atomically (`atomic_json_write_strict`: tmp file + fsync + rename, sorted keys, NaN rejected).
- `false_mappings.csv` — one row per oracle FP/FN (`source_entity_uri`, `target_entity_uri`, `oracle_prediction`, `oracle_confidence`, `in_reference`, `error_type`), written next to the JSON. The `false_mappings` list is stripped from the JSON itself, and a stale CSV is deleted when a run produces no FP/FN.

`evaluation/contract.py::validate_evaluation_payload` re-validates the JSON downstream in `experiments/run.py` (see [experiments](experiments.md)): counts must reconcile with ratios (`reference_size == tp + fn`, `system_size == tp + fp`, or the partial-GS variant with `ignored`/`evaluated_size`), the engine name must match the metric `source` (`partial_reference` emits `kg_partial`), every `None` metric must carry a `metric_notes` reason, and `false_mappings` must not appear in the artifact. A payload that fails any of these raises `EvaluationContractError` — an evaluation that cannot account for its own counts is treated as no evaluation at all.

## Track-faithful engines (`evaluation.engines`)

The plain engine scores oriented `(source, target)` pairs against the `=` cells of `reference.tsv` and ignores every relation symbol. The published OAEI protocols do not all work that way, and the same alignment file can receive quite different figures (the EACL 2025 FMA–SNOMED alignment is F1 0.594 plain, 0.694 under LogMap's evaluator, 0.695 coherence-aware and 0.707 with the `?` cells as positives). Since 22 Sep 2026 (`LOCAL_CHANGES.md` §5) two further engines implement those conventions, as pure functions in `evaluation/conventions.py` (cell-level loaders that keep the relation: LogMap pipe `.txt`, TSV with a `= < > ?` third column, OAEI Alignment RDF/XML; LogMap's bare `<`/`>` relation spellings are repaired on the way in):

| engine | convention | reproduces |
| --- | --- | --- |
| `logmap_oaei` (`engines/logmap_oaei.py`) | LogMap's own OAEI evaluator (`HashAlignment` + `StandardMeasures`, the SEALS client code): pairs orientation-insensitive, a system cell is a true positive only with the reference's relation (`<`/`>` swap when the pair is stored the other way round), reference cells flagged `?` ignored and a system pair on such a cell discounted from the false positives; `printed_3dp` = P and R rounded to three decimals with Java `Math.round`, F from the rounded values (`rounded = true` makes those the block's own P/R/F1, flagged `rounded_3dp`) | EACL 2025 Table 3 (9/9), OAEI 2021 LargeBio published figures (15/15) |
| `bioml` (`engines/bioml.py`) | the Bio-ML track protocols by `edition` and `setting` (table below); relation-agnostic pair existence, as the organisers' package | OAEI 2025 official figures (6/6), the organisers' 2026 standard / coherence-aware / CodaBench scores (to 4 dp) |

`[evaluation] engines = [...]` lists the engines. Unset (the default) keeps the historical selection of the `global` block byte-for-byte. When set, the first entry is the primary engine (`custom`, `partial_reference` or `deeponto`; it must be `partial_reference` exactly when `partial_reference = true`, and `force_custom_eval` is then superseded) and produces the `global` and `oracle` blocks exactly as before; every further entry adds a `global_<engine>` block and, when oracle metrics are requested, an `oracle_<engine>` block scored against that engine's own reference pairs (`logmap_oaei`: every cell of its reference, `?` included, in either orientation, i.e. the EACL Table 2 labelling; `bioml`: the headline setting's positives). Each engine takes its options from `[evaluation.<engine>]`; the paths there default to `reference_alignment_path`. Every block has the `compute_prf` shape plus a `protocol` key and protocol-specific counts, and the artifact contract reconciles it like the plain block (`system_size = tp + fp` is always the *evaluated* system size, the raw counts are in `system_cells`, `system_pairs`, …). Standalone: `python -m logmap_llm.evaluation.harness --system A.tsv --reference R.tsv --engines custom,logmap_oaei --engine-option logmap_oaei.reference_path=reference.rdf`.

The `bioml` engine reports one headline block (the configured `setting`) and every other view whose inputs are configured under `views`, each a full block:

| edition | `unsupervised` | `semi_supervised` | `codabench` | class exclusion |
| --- | --- | --- | --- | --- |
| 2022 | plain vs the full reference | null-reference: `train_alignment_path` pairs removed from both sides, recall over `test_reference_path` (or full − train) | – | none |
| 2023, 2024, 2025 | predictions touching an `ignored_classes_path` IRI dropped, then plain vs the full reference (**the OAEI 2025 official protocol**) | same exclusion, then null-reference | – | `use_in_alignment = false` classes |
| 2026 | `standard` (every cell of `reference_path` positive, `?` included) and `repaired` (`reference_repaired_path`: `?` cells removed from both sides, the track's headline) | – | both views restricted to the `test` rows of `split_path`, `train`+`valid` pairs masked from the predictions (the CodaBench scorer) | `deprecated_classes_path` IRIs dropped from predictions and references |

Views are named `unsupervised`, `semi_supervised`, `standard`, `repaired`, `codabench_standard`, `codabench_repaired`; `headline_view` names the one copied to the top level. The staging scripts of the campaign write the inputs next to `reference.tsv`: `reference_full.tsv` (every reference cell with its relation), `ignored_classes.txt`, `deprecated_classes.txt`, `reference_train.tsv`, `reference_test.tsv`, `split.tsv`.

Which convention reproduces which published table (from the 2025 ↔ 2026 reconciliation):

| published figure | engine and options |
| --- | --- |
| EACL 2025 Table 3 (LogMap + oracle P/R/F), OAEI 2021 LargeBio results | `logmap_oaei` on `reference_full.tsv` / `reference.rdf`, `printed_3dp` |
| EACL 2025 Table 2 (oracle Se/Sp/YI) | `oracle_logmap_oaei` against the EACL `refs_equiv/full.tsv` (LargeBio: `?` cells positive) |
| OAEI 2025 Bio-ML results (OM 2025 Table 1) | `bioml`, edition 2025, unsupervised, `ignored_classes.txt` |
| Bio-ML 2026 CodaBench leaderboard | `bioml`, edition 2026, `codabench`, `deprecated_classes.txt`, `split.tsv` |
| Bio-ML 2026 organiser-side complete references; LargeBio appendix of the 2026 paper | `bioml`, edition 2026, unsupervised (`repaired` / `standard` views; for LargeBio the UMLS `reference.rdf` is both) |
| campaign `results/<track>/summary.csv` | the plain `global` block (unchanged) |

## The local oracle (`pipeline.consult_oracle = "local"`)

LogMap's `LocalOracle.loadLocalOraculoLLM(base_path)` reads every `*.csv` file of `oracle.local_oracle_predictions_dirpath` (`Source,Target,Prediction[,Confidence]`; `#` lines and lines without a comma skipped; the third field `true`, case-insensitively, accepts the pair in both orientations, anything else rejects it) and opens `base_path + filename`, so the pipeline hands the directory over with a trailing separator. Before LogMap starts, `pipeline/orchestration.py::load_local_oracle_verdicts` counts the verdicts with the same rule and logs them; no verdict at all is fatal (a run in which every candidate is rejected is a configuration error, not a result), no accepted verdict a warning; the counts are recorded in `run_result.json` under `counts.local_oracle`. A candidate without a verdict in any file is rejected.

## Configuration

All keys live under `[evaluation]`; see the [configuration reference](configuration.md) for the full schema.

| Key | Default | Meaning |
| --- | --- | --- |
| `evaluate` | `false` | run Step 5; requires `reference_alignment_path` when true |
| `reference_alignment_path` | `None` | reference alignment TSV |
| `train_alignment_path` | `None` | train reference excluded from scoring (custom/DeepOnto only) |
| `metrics` | `["global", "oracle"]` | blocks to compute; list or comma-separated string |
| `force_custom_eval` | `true` | force the custom engine; DeepOnto is opt-in |
| `partial_reference` | `false` | KG-track partial gold-standard semantics |
| `stratified_by_entity_type` | `false` | class/property/instance strata |
| `stratified_class_property` | `false` | conference M1/M2 strata; needs the initial alignment |
| `jvm_memory` | `"8g"` | heap exported to DeepOnto's JVM (`^[1-9][0-9]*[mMgG]$`) |
| `engines` | unset | engine list: primary engine first (`custom`, `partial_reference`, `deeponto`), then `logmap_oaei` / `bioml`; unset = historical selection |
| `logmap_oaei.reference_path` / `.rounded` / `.orientation_insensitive` | `reference_alignment_path` / `false` / `true` | `[evaluation.logmap_oaei]` options |
| `bioml.edition` / `.setting` | `2025` / `"unsupervised"` | `[evaluation.bioml]`: edition 2022–2026, setting `unsupervised` / `semi_supervised` (≤ 2025) / `codabench` (2026) |
| `bioml.reference_path`, `.reference_repaired_path`, `.test_reference_path`, `.train_alignment_path`, `.ignored_classes_path`, `.deprecated_classes_path`, `.split_path` | unset | the setting's inputs (see the edition table) |
