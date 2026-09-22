# LogMapLLM

LogMapLLM is a research codebase that extends the [LogMap](https://github.com/ernestojimenezruiz/logmap-matcher) ontology-matching system with an LLM oracle. LogMap can ask a human oracle about the mappings it is unsure of; here that role is played by a language model reached over any OpenAI-compatible API (OpenRouter, vLLM, SGLang, OpenAI), with everything around it — prompt construction, few-shot retrieval, refinement, evaluation, batch experiments.

## The idea

A classical LogMap run (Java, driven from Python via JPype) produces two things: an initial alignment between the source and target ontologies, and a subset of candidate mappings it considers doubtful — the file LogMap calls "mappings to ask the oracle", written here as M_ask. For each candidate in M_ask the pipeline renders a natural-language question grounded in ontology context (labels, synonyms, parents, siblings, domain and range, instance attributes), optionally prepends retrieved few-shot demonstrations, and asks the LLM for a True/False verdict on whether the two entities match.

The verdicts are then folded back into the alignment: refined = (initial − M_ask) ∪ {m ∈ M_ask : oracle(m) = True}, either through LogMap's own Java refinement (which also resolves logical conflicts) or a Python set-union approximation (`pipeline.refinement_strategy = "python"`). The refined alignment can finally be scored against a reference, including OAEI Knowledge Graph track semantics for partial gold standards, and every successful run publishes a machine-readable `run_result.json` recording parameters, counts, timings and sha256 artifact hashes.

## Running it in 30 seconds

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
```

`[alignmentTask]`, `[oracle]` and `[outputs]` are the only required tables; everything else — prompt templates, few-shot retrieval, per-step run modes — has defaults. The schema is strict, so a misspelt key is a validation error rather than silently ignored, and the `ENV:` sentinel keeps the API key out of the file (it is read from the environment when consultation starts). The [configuration reference](configuration.md) covers every field. You also need a LogMap bundle on disk (`logmap-matcher-4.0.jar`, `java-dependencies/`, `parameters.txt`), found under `./logmap` by default or wherever `alignmentTask.logmap_parameters_dirpath` points.

The CLI flags:

- `--config PATH` (or `-c`) — the TOML configuration file; defaults to `configs/default_config.toml`.
- `--reuse-align` — reuse an existing LogMap alignment instead of re-running the matcher.
- `--reuse-prompts` — reuse previously built oracle prompts; implies `--reuse-align`.
- `--no-cache` — disable owlready2 quadstore caching and parse the ontologies from scratch.
- `--run-root DIR` — root all outputs under `DIR`, creating `logmapllm-outputs`, `logmap-initial-alignment` and `logmap-refined-alignment` subdirectories.

The run executes five phases in order — align, prompt build, oracle consultation, refinement, evaluation — followed by reporting. Prompt building and evaluation each run in their own subprocess: owlready2 cannot coexist with JPype in one process, and the DeepOnto evaluator needs its own JVM. [Running the pipeline](pipeline.md) walks through each phase and the artifacts it produces.

## Package layout

```
logmap_llm/
├── __main__.py       python -m logmap_llm → pipeline.runner.main()
├── config/           TOML schema (Pydantic) and loader
├── pipeline/         the five-phase driver: runner, orchestration, paths, reporting
├── oracle/           LLM client (manager.py) and threaded consultation campaign
│   ├── prompts/      user/developer prompt templates per entity lane (class/property/instance)
│   └── rag/          query-specific few-shot retrieval from LogMap anchors
├── evaluation/       alignment and oracle scoring, run as a subprocess harness
├── experiments/      deterministic batch harness: batch.toml → sealed batch directory
├── ontology/         owlready2 access layer, annotation indices, sibling retrieval
├── bridging.py       LogMap's Java MappingObjectStr ↔ pandas DataFrames
├── interface.py      JPype wrapper around LogMap's LogMapLLM_Interface
├── constants.py      shared enums, column names, separators, defaults
└── utils/            logging, atomic writes, subprocess bootstrap
```

## Documentation map

- [Architecture](architecture.md) — how the pieces fit together, and why prompt building and evaluation live in subprocesses.
- [Running the pipeline](pipeline.md) — the five phases, their artifacts, and the reuse/bypass modes.
- [Configuration reference](configuration.md) — every `config.toml` table and field, with defaults and validation rules.
- [The LLM oracle](oracle.md) — endpoints, interaction styles, answer parsing, retries, and logprobs-derived confidence.
- [RAG few-shot retrieval](rag.md) — how high-confidence anchors become per-query demonstrations, and the negative-construction layouts.
- [Evaluation](evaluation.md) — scoring the refined alignment and the oracle, evaluation engines, and partial gold standards.
- [Experiments](experiments.md) — the batch harness: generating, running, and aggregating sealed experiment batches.
- [Ontology access layer](ontology.md) — loading and caching ontologies, the entity object model, and sibling selection.
