# Architecture

LogMapLLM extends the LogMap matcher with an LLM oracle. First, LogMap produces an initial alignment and flags the subset of candidate mappings it is unsure about ($M_{ask}$). The LLM's job is to judge _(adjudicate)_ that subset. Every candidate correspondence in $M_{ask}$ becomes a natural-language question, the LLM responds (e.g., True or False), and the accepted candidates are merged back into the alignment. A run is a fixed sequence of phases driven by `logmap_llm/pipeline/runner.py`:

    align -> prompt_build -> consult_oracle -> refine_alignment -> evaluate -> reporting

`python -m logmap_llm --config <path>` is the entry point (see [pipeline.md](#)); the whole run is parameterised by a single TOML file validated by the Pydantic schema in `logmap_llm/config/schema.py` (see the [configuration guide](#)). Each phase in `pipeline/orchestration.py` matches a mode enum from that config (`align`/`reuse`/`bypass`, etc.), so any prefix of a run can be replayed from artifacts on disk instead of recomputed.

## LogMapLLM, dependencies & Java

LogMap is written in Java. Thus, the parent process starts a JVM using JPype (`logmap_llm/interface.py` - this is a thin wrapper around LogMap's `LogMapLLM_Interface`) which it uses for alignment and refinement.

Note that prompt building uses owlready2, which also initialises a JVM (since it requires access to reasoners). Additionally, the DeepOnto evaluation engine (if used) also starts its own JVM.

Since we have encountered issues with multiple JVMs running via JPype in the same process, we use subprocesses during stage two (`python -m logmap_llm.pipeline.stage_two`) and stage five (`python -m logmap_llm.evaluation.harness`). Each re-reads the provided config via `logmap_llm/utils/subprocess.py::subprocess_bootstrap`, writing artfacts to disk, which are then read by the parent process.

## LogMapLLM Stages

**Stage 1 — align:** LogMap performs the initial alignment and exposes two Java mapping sets: the full initial alignment and $M_{ask}$.

**Stage 2 — prompt build:** Loads both input ontologies through the ontology access (`logmap_llm/ontology/`), then produces one prompt per $M_{ask}$ candidate using the separate templates for classes, properties, and instances.

**Stage 3 — consult:** Sends each prompt to any OpenAI-compatible chat-completions endpoint (OpenRouter, vLLM, SGLang, OpenAI). Each response is parsed as an `Oracle_prediction` value (`True`, `False`, `error`, `skipped`) alongside a logprobs-based `Oracle_confidence` and token counts. These are appended to the $M_{ask}$ DataFrame. In bidirectional mode each candidate is asked in both directions and equivalence requires both answers to be True.

**Stage 4 — refine:** The accepted predictions are passed back to LogMap's Java refinement, which resolves logical conflicts (i.e., performs a final repair pass).

**Stage 5 — evaluate:** Runs only when `evaluation.evaluate = true`. Scores the refined alignment against a reference (complete or OAEI KG-track partial gold standard) and the oracle as a binary classifier, through a pluggable engine (pure-Python custom, partial-reference, or DeepOnto).