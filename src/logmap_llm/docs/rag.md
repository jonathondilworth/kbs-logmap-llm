# RAG few-shot retrieval

When `few_shot_k > 0`, the [pipeline](pipeline.md) does not send each M_ask candidate to the [oracle](oracle.md) cold: it prepends k demonstrations chosen for that specific candidate. The demonstrations are pseudo-labelled — positives are LogMap's own high-confidence anchors (initial-alignment equivalence rows that are *not* in M_ask, so they are already-decided rather than under question), and negatives are constructed by perturbing those positives. No evaluation reference is ever read, and M_ask pairs are excluded from the corpus as leakage, so the mechanism stays honest as a zero-supervision baseline. The whole subsystem lives in `logmap_llm/oracle/rag/` and is deliberately owlready2-free; the ontology-dependent glue sits in `logmap_llm/pipeline/rag_fewshot.py`.

## Components

The retrieval core is four small pieces, all under `logmap_llm/oracle/rag/`:

- **Corpus** (`corpus.py`, `TypedCorpus`) — per-kind pools of `Example` objects, one pool each for CLS, OPROP, DPROP and INST, kept disjoint so a class query never retrieves a property demonstration. Deduplication and exclusion use direction-agnostic frozensets of complete IRIs, so identical local names in different namespaces stay distinct. `corpus_hash()` is an order-independent digest over the pool contents.
- **Encoder** (`encoder.py`) — anything satisfying the `Encoder` protocol: `encode(list[str]) -> np.ndarray` with L2-normalised rows, plus `repo`, `revision`, `preprocessing_version` and `runtime_versions` attributes that feed the cache key. `ClsPooledEncoder` (CLS-pooled transformer) and `SbertEncoder` (masked-mean pooling; refuses an empty revision) are the semantic options; `HashingEncoder` is a deterministic char-n-gram baseline used by the CPU-only tests.
- **Index** (`index.py`, `EmbeddingIndex`) — an embedding matrix cached on disk as `ragindex-{index_hash}.npz`. The hash covers the corpus hash, dataset fingerprint, encoder repo/revision/preprocessing/runtime versions, retriever preprocessing version, language and kind, so a cache hit can never serve vectors from a different corpus or encoder; any mismatch or corruption is a recorded rebuild. `rank()` sorts by `(-similarity, example_id)`, making retrieval fully deterministic.
- **Retriever** (`retriever.py`, `RagRetriever`) — the per-query engine. `retrieve(query_mapping, entity_type, relation, prompt_template, k, corpus_id, exclude_keys=None)` returns a `RetrievalResult` (a tuple of `ExampleRef` plus a `RetrievalTrace`). `warmup()` pre-builds the typed indexes before concurrent use; all value types are frozen dataclasses (`types.py`), so retrieval is thread-safe.

Under `mode = query_rag` the retriever encodes the query's embed text (concatenated preferred labels of both entities, built by `make_embed_text_fn` in `pipeline/rag_fewshot.py`), ranks the same-kind positive pool by cosine similarity, and excludes the live pair, its reverse, and every M_ask pair. The static modes (`static_hard`, `static_random`) reuse the same machinery with a seeded query-agnostic ordering, preserving the ablation baselines; `zero_shot` returns nothing.

## Negatives and layouts

There are no labelled negatives, so the retriever constructs them. The layout is chosen with `few_shot.rag_negative_layout`, which has no default in the pipeline path — `build_retriever_from_pipeline` makes it keyword-only precisely so a config cannot silently pick one:

- `paired-sibling-v2` — for each retrieved positive `(s, t)`, build one negative `(s, t')` where `t'` is a ranked sibling of `t` (a near-miss). Sibling lookup is injected as `sibling_fn` (built by `make_sibling_fn` in `pipeline/rag_fewshot.py`, backed by the [ontology layer's](ontology.md) `SiblingSelector`); a failed lookup never aborts the query — the donor rule takes over for that negative and the reason is recorded.
- `paired-donor-v2` — the same one-negative-per-positive pairing, but `t'` comes from the next eligible ranked donor positive.
- `donor-cross-v1` — the frozen legacy layout: cross the source of one donor with the target of another, padded with seeded random column swaps if the policy allows. Preserved byte-for-byte so completed campaigns reproduce.

The paired layouts enforce strict minimal pairs: every rendered block is a complete `(P_i, N_i)` with the same source entity, `few_shot_k` must be even, and the token-budget cut-off rolls back to the last complete pair rather than orphaning a positive. Degradations are never silent — `RetrievalTrace.fallback_reason` records retrieval-mode degradation (empty pool, budget overflow), while the separate `negative_fallback_reason` records negative-construction fallbacks that did *not* degrade the query.

## Reaching the oracle without touching the shared prefix

Stage 2 (`pipeline/stage_two.py`) calls `build_query_specific_few_shot()` in `pipeline/rag_fewshot.py`, which wires ontology-backed callables into the retriever: `make_embed_text_fn` (similarity text), `make_render_fn` (renders each demonstration with the *same* per-kind prompt template the live query uses, so a DPROP example surfaces its datatype range exactly as the query does), and optionally `make_sibling_fn`. It runs retrieval for every M_ask key, and stage 2 persists two artifacts next to the prompts:

- `{task}-{template}-few_shot_examples.json` (`run_paths.few_shot_json()`) — `{m_ask_key: [[user_prompt, assistant_answer], ...]}`, keyed `src_iri|tgt_iri` (plus `...|REVERSE` keys when bidirectional; the reverse direction reuses the same selected evidence re-rendered in the opposite orientation, so the two directions differ only in orientation, not in which demonstrations were retrieved).
- `rag_traces.json` — the per-key `RetrievalTrace` dicts.

At consultation time (`oracle/consultation.py`), the artifact's type decides the mechanism: a plain *list* of examples is legacy shared few-shot and is baked into the manager's frozen message prefix; a *dict* is query-specific and is passed per consultation instead. `OracleConsultationManager._build_base_kwargs` (`oracle/manager.py`) assembles each request locally — frozen prefix, then this query's user/assistant demonstration turns, then the query itself — so concurrent worker threads never mutate or read shared conversation state, and the frozen developer prefix stays example-free.

## Configuration

The relevant `[few_shot]` keys (`config/schema.py`, `FewShotConfig`; see the [configuration reference](configuration.md)):

| Key | Default | Meaning |
| --- | --- | --- |
| `few_shot_k` | `0` | Examples per oracle request; 0 disables the subsystem |
| `few_shot_negative_strategy` | `"hard"` | Maps to a retrieval mode via `mode_from_strategy` (`"query-rag"`/`"hard-similar"` → query_rag) |
| `few_shot_seed` | `42` | Seeds static ordering and random-negative fallback only |
| `rag_negative_layout` | none | Required whenever `few_shot_k > 0`; paired layouts require even k |
| `rag_encoder_kind` | `"cls_transformer"` | Or `"sbert"`, or `"hashing"` (named baseline) |
| `rag_encoder_model` / `rag_encoder_revision` | `""` / `None` | Semantic encoders require both; the revision pin stops a model update silently changing an experiment or its cache identity |
| `rag_encoder_device` / `rag_encoder_max_length` | `None` / `64` | Runtime placement and truncation |
| `rag_failure_policy` | `"error"` | `"record_zero_shot"` degrades a failed generation to zero-shot and writes `rag_fallback.json` instead of aborting |
| `rag_cache_dir` | `None` | Directory for the `ragindex-*.npz` cache |
| `prebuilt_few_shot_bundle_path` | `None` | Switch to a prebuilt bundle (below) |

## Prebuilt bundles

For campaigns that need cross-task demonstrations, `python -m logmap_llm.oracle.rag.bundle PLAN.json --output-dir BUNDLES` builds bundles offline from sealed [experiment batches](experiments.md). The plan JSON pins an encoder (`cls_transformer` with a 40–64-character hex revision) and names at least two sealed tasks; for each receiver task the builder takes anchors *only from other tasks* (strict leave-one-task-out), excludes the union of every task's M_ask pairs, selects the top two same-kind positives per query and crosses two negatives within a single donor task, and renders each block as exactly P,N,P,N with k=4 (selection policy `strict-loo-typed-equivalence-pnpn-v2`). `write_bundle_documents` refuses to overwrite an existing bundle whose content differs.

A bundle is a scientific input, not a cache: `load_prebuilt_few_shot_bundle()` (`pipeline/rag_fewshot.py`) accepts it only when its recorded binding matches the live dataset fingerprint, the M_ask file's sha256, and the exact requested k/strategy/encoder/prompt configuration, and re-validates the P,N,P,N answer pattern, per-example prompt hashes and donor-task provenance. Accepted bundles are republished in the ordinary `few_shot_examples.json`/`rag_traces.json` wire format, so consultation needs no second code path.

## Provenance

Every retrieval emits a full trace: requested and effective mode and k, entity type, prompt family, per-example rows (id, label, source, direction, rank, similarity, IRIs, token count, `prompt_sha256`, negative-construction provenance), corpus and index hashes, encoder identity, token budget and usage, and index cache status (hit / rebuilt / absent). Stage 2 aggregates the failure signals into two artifacts that must never be conflated: `rag_fallback.json` (a query received fewer or worse examples than requested) and `rag_negative_fallback.json` (a negative was built by a fallback rule but the query itself was not degraded). Given the same corpus, encoder, config and query, retrieval is exactly reproducible.
