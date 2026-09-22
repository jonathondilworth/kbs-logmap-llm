# LogMapLLM Experimental (KBS fork)

This repository contains experiment data and on-going work related to [LogMapLLM](https://github.com/city-artificial-intelligence/logmap-llm), a project that extends the [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher) ontology matching system with an LLM oracle.

## LogMapLLM

Expert knowledge is often required for effective ontology matching. LogMap has an interactive matching mode for this purpose, where **uncertain mappings**, denoted $M_{ask}$, are escalated for review by a human oracle. It is plausible, however, to replace the human expert by a large language model (LLM); a paradigm known as _LLMs-as-Oracles_. By **assembling prompts** for each correspondence in $M_{ask}$ and **consulting** an LLM via an OpenAI-compatible API (e.g., via OpenRouter, to vLLM, etc.), we fold the response back into the proposed alignment before **refinement** (i.e., the repair step).

Experimental features _(this repository)_:

* extended prompt construction (class, property, instance prompt templates),
* few-shot ontology-driven prompting with RAG-based features,
* collective anchor-based few-shot ontology-driven prompting,
* and self-hosted, open-weight-based evaluation.

## Usage

```sh
python -m logmap_llm --config path/to/config.toml
```

with a config like:

```toml
[alignmentTask]
task_name = "mouse-human"
onto_source_filepath = "data/mouse.owl"
onto_target_filepath = "data/human.owl"

[oracle]
model_name = "qwen/qwen3-32b"
api_key = "ENV:OPENROUTER_API_KEY"    # or "EMPTY" for a local vLLM/SGLang server

[outputs]
logmapllm_output_dirpath = "output/logmapllm"
logmap_initial_alignment_output_dirpath = "output/initial-alignment"
logmap_refined_alignment_output_dirpath = "output/refined-alignment"

# optional: score the refined alignment against a reference
[evaluation]
evaluate = true
reference_alignment_path = "data/reference.rdf"

# optional: automatic model selection (off by default)
# each candidate is a partial [oracle] table 
# local and hosted models can mix (see below)

[model_selection]
automatic = true
max_anchors = 10                       # anchors sampled for ranking, one constructed negative each

[[model_selection.candidates]]
model_name = "deepseek/deepseek-v4-flash"

[[model_selection.candidates]]
model_name = "Qwen3.5-122B-A10B"
base_url = "http://127.0.0.1:8000/v1"  # a local vLLM/SGLang server
api_key = "EMPTY"
interaction_style = "vllm"
```

The config is a TOML file validated by a Pydantic schema; at minimum it needs `[alignmentTask]` (task name and the two ontology paths), `[oracle]` (model name), and `[outputs]` (three output directories). You also need a LogMap bundle on disk (`logmap-matcher-4.0.jar`, `java-dependencies/`, `parameters.txt`), found under `./logmap` by default or wherever `alignmentTask.logmap_parameters_dirpath` points. 

The CLI flags:

- `--config PATH` (or `-c`) — the TOML configuration file; defaults to `configs/default_config.toml`.
- `--reuse-align` — reuse an existing LogMap alignment instead of re-running the matcher.
- `--reuse-prompts` — reuse previously built oracle prompts; implies `--reuse-align`.
- `--no-cache` — disable owlready2 quadstore caching and parse the ontologies from scratch.
- `--run-root DIR` — root all outputs under `DIR`, creating `logmapllm-outputs`, `logmap-initial-alignment` and `logmap-refined-alignment` subdirectories.

The run executes five phases in order: (1) align, (2) prompt build, (3) oracle consultation, (4) refinement, and (5) evaluation; followed by reporting. With `[model_selection] automatic = true` an extra phase (2b) runs between prompt build and consultation; see below. Prompt building and evaluation each run in their own subprocess, since owlready2 and JPype contained to a single process has be known to cause problems; we also separate the [DeepOnto](https://github.com/KRR-Oxford/DeepOnto) evaluator into its own JVM spawned by a subprocess when used.

### New Feature: Automatic model selection

This is off by default. Note that LogMap's anchors, the initial-alignment equivalences it did not escalate to $M_{ask}$, are assumed correct. They can be turned into questions with _known_ answers without a reference (ground truth). The pipeline (stage 2/b) can then sample up to `max_anchors` as positives, construct one negative per anchor, and produce 'automatic selection prompts'. Every `[[model_selection.candidates]]` entry is tested on these prompts, and the candidates are ranked by correct answers. The LLM oracle ranked top-1 is selected for use during the remainder of the pipeline (a candidate whose consultation aborts is ranked last). 

The questions and the ranking are written next to the prompts (`...-model_ranking_prompts.json`, `...-model_selection.json`) and summarised in `run_result.json`. The batch harness (`logmap-llm-batch`) refuses the setting, since a batch fixes each job's model on its `models` axis.

### New Feature: Collective anchors

The few-shot bundle (`few_shot.prebuilt_few_shot_bundle_path`) is set, by default, to leave-one-task-out, where every demonstration comes from another task in the plan. However, now a plan carrying `"anchor_pool": "pooled"` lets the receiver's own anchors compete as well (the campaign-wide $M_{ask}$ exclusion is unchanged).

## Project Structure

```
logmap_llm/
├── __main__.py       python -m logmap_llm -> calls -> pipeline.runner.main()
├── config/           contains the TOML schema (Pydantic) and loader
├── pipeline/         the five-phase driver: runner, orchestration, paths, reporting
├── oracle/           LLM client (manager.py) and threaded consultation campaign
│   ├── prompts/      user/developer prompt templates per entity lane (class/property/instance)
│   └── rag/          query-specific few-shot retrieval from LogMap anchors
├── evaluation/       alignment and oracle scoring, run as a subprocess harness
├── experiments/      experimental batch harness: batch.toml -> sealed batch directory
├── ontology/         owlready2 access layer, annotation indices, sibling retrieval
├── bridging.py       LogMap's Java MappingObjectStr <-> pandas DataFrames
├── interface.py      JPype wrapper around LogMap's LogMapLLM_Interface
├── constants.py      shared enums, column names, separators, defaults
└── utils/            logging, atomic writes, subprocess bootstrap
```

## Roadmap

* Improved documentation.

## License

[Apache License 2.0](./LICENSE)
