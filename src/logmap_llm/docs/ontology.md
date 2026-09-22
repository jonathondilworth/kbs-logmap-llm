# Ontology access layer

The `logmap_llm.ontology` package loads the two OWL ontologies of an alignment task and answers every question prompt construction asks about them: what an entity is, what it is called, where it sits in the hierarchy, and which near-miss neighbours it has. It runs inside the Stage 2 prompt-build subprocess (see [running the pipeline](pipeline.md)) because owlready2 cannot coexist with JPype in the same process. The facade is `OntologyAccess` in `logmap_llm/ontology/access.py`; the pipeline reaches it via `load_ontologies(src_path, tgt_path, cache_dir=..., vocabulary=..., stub_import_iris=...)`, which returns one `OntologyAccess` per ontology.

## Loading and the quadstore cache

Each `OntologyAccess` owns its own `owlready2.World` — an isolated SQLite quadstore — so two loaded ontologies never share state. `load_ontology()` binds the ontology object, materialises an rdflib view of the world (`self.graph = world.as_rdflib_graph()`), and builds O(1) URI-to-object dictionaries for classes, properties and individuals. A `Reasoner` enum exists (`HERMIT`, `PELLET`, `STRUCTURAL`, `NONE`); the default is `NONE`, and `STRUCTURAL` currently has no implementation branch, so it behaves like `NONE`.

Parsing a large ontology into owlready2 is the expensive step, so `logmap_llm/ontology/cache.py` persists the parsed quadstore to disk. The canonical cache lives under `$XDG_CACHE_HOME` (or `~/.cache`) at `logmap-llm/owlready2/<sanitised-name>_<sha256[:12]>.sqlite3`, where the hash covers the resolved source path plus any stubbed import IRIs — the import policy changes the triples, so it is part of the cache identity. A cache file is considered valid when it exists, is non-empty, and its mtime is at least the source file's mtime; there is no content hash, so touching the source rebuilds the cache. Builds are serialised with an `fcntl.flock` on a `.sqlite3.lock` file: one process builds into a temp file and `os.replace`s it into place while concurrent processes wait and then reuse it. On open, the canonical file is copied to a process-private directory under `/tmp/logmap-llm-owlcache-*/` and wrapped in a `_PrivateCachedWorld`, so concurrent runs never write-contend on the same SQLite file; the private copy is removed on `close()` and again by an `atexit` handler.

Reopening a cached World needs care: owlready2 does not persist the local-file alias when an ontology declares a different public IRI, so `_bind_cached_ontology` recovers the right ontology object by base IRI, then by the declared-minus-imported root, then by "the only populated ontology". If none of those is unambiguous it raises a `RuntimeError` advising `--no-cache` rather than silently returning an empty ontology.

`--no-cache` on `python -m logmap_llm` (forwarded to the Stage 2 and evaluation subprocesses) is the escape hatch: `stage_two.py` then passes `cache_dir=None`, no `OntologyCache` is constructed, and the ontology is parsed from source into a fresh in-memory World.

`alignmentTask.stub_import_iris` (a list, empty by default) pre-registers the named IRIs as already loaded in the build World, so owlready2's `owl:imports` resolution short-circuits instead of fetching them over HTTP. This exists for the Circular Economy track, whose `CEON.rdf` declares an import that was never published and 404s for everyone.

## The entity object model

`logmap_llm/ontology/object.py` wraps raw owlready2/rdflib objects in three entity classes consumed by the [prompt templates](oracle.md):

- `ClassEntity` — annotation dict with `uri`, `preferred_names`, `synonyms`, `all_names` and `parents` (children are loaded lazily); navigation helpers such as `get_direct_parents`, `get_parents_by_levels(max_level=3)`, `get_restrictions`, `get_relational_signature`. Name sets for a declared entity are never empty: they fall back to `{entity.name}`.
- `PropertyEntity` — adds `domain_names` and `range_names` (datatype names for data properties, class labels otherwise), OWL characteristics (`transitive`, `symmetric`, `functional`, `inverse_functional`), and inverse-name lookup.
- `InstanceEntity` — wraps `OntologyAccess.getInstanceContext(uri)`, a structured dict of labels, types, abstract, categories and data/object property rows swept from the rdflib graph.

`resolve_entity(uri, onto)` maps a bare URI to `(entity, kind)` by trying class, then property, then individual, then falling back to "any subject present in the graph" (which covers KG instances outside the owlready2 index). Because that class-first order misresolves OWL2-punned URIs, `resolve_entity_as(uri, onto, lane)` honours LogMap's authoritative entity-type tag: lane `"CLS"` accepts a declared class or, failing that, an implicit label-only class view for URIs used as an `rdf:type`; lanes `"OPROP"`/`"DPROP"` always return a `PropertyEntity`, building a minimal undeclared view for ABox predicates that were never declared as OWL properties. The implicit and undeclared views (`_URIBackedImplicitClass`, `_URIBackedUndeclaredProperty`) never invent axioms: they carry labels and a URI, with empty parents, children, domain and range.

## Annotation indexing and language filtering

`indexAnnotations()` builds three dictionaries keyed by subject URI string, each mapping to a set of strings: `preferredLabels`, `entityToSynonyms`, and `allEntityAnnotations` (the lexical superset — every value lands there). The predicate groups come from `logmap_llm/ontology/data/annotation_uris.json` (`preferred_label`, `synonym`, `lexical_extra`), kept as data rather than authored source because the IRIs must match the RDF byte-for-byte; a missing or malformed resource fails loudly instead of degrading into "this ontology has no labels". Each predicate is swept once with direct rdflib triple lookups. Direct annotations (literal objects) pass through the vocabulary's language filter; indirect annotations (an intermediate node) are followed to that node's `rdfs:label`.

Language filtering is a vocabulary concern: `accepts_language(lang)` accepts everything when the filter is empty, always accepts untagged literals, and otherwise requires membership. The default filter is `("en",)`; the SKOS preset uses an empty filter for multilingual tracks. Readers are `getPreferredLabels`, `getSynonymsNames` and `getAnnotationNames`, all keyed by `entity.iri` and returning an empty set for unknown entities.

Independently of the indices, `getLabelsForURI(uri)` fetches `rdfs:label` values for any URI and returns them sorted lexicographically, falling back to the URI fragment — rdflib triple iteration order is not stable across processes, so downstream `[0]` indexing would otherwise be nondeterministic.

## Sibling retrieval

`SiblingSelector` in `logmap_llm/ontology/sibling_retrieval.py` supplies near-miss neighbours for prompt context and for hard-negative construction in [RAG few-shot retrieval](rag.md). For a class, `select_siblings(entity, max_count=2)` unions the direct children of every direct parent, removes the entity itself, sorts by the stable key `(label, IRI)`, caps the candidate set at 50 (`DEFAULT_MAX_SIBLING_CANDIDATES`), and ranks the remainder, returning `(entity, score)` pairs in descending score. Four strategies exist (`logmap_llm/ontology/sibling_strategy.py`):

- `alphanumeric` — sort by label, ascending; flat 1.0 scores.
- `shortest_label` — sort by label length, then label; flat 1.0 scores.
- `cls_transformer` — CLS-pooled transformer encoder; no built-in checkpoint, so `prompts.sibling_model` must be set.
- `sbert` — `sentence-transformers/all-MiniLM-L12-v2` with mean pooling.

Pooling is implied by the strategy and not caller-overridable, because mismatched pooling silently degrades embedding quality. Embeddings are L2-normalised so the dot product is cosine similarity, and they are cached per IRI for the run. `prompts.sibling_model_revision` pins the checkpoint commit; the resolved revision is exposed via `SiblingSelector.model_revision` for the trace. The compute device can be pinned with `prompts.sibling_encoder_device` (unset resolves to CUDA when available, else CPU); pin it across hosts, because CUDA and CPU matmuls can resolve near-ties differently.

When the candidate set already fits within `max_count`, ranking is skipped and all candidates return with score 1.0. That short-circuit is a cost optimisation for prompt templates and is wrong for negative construction, which must pass `force_rank=True` or it would silently receive alphabetically ordered siblings under the embedding strategy's name.

`select_siblings` dispatches by entity type: instances go to `select_instance_siblings` (shared `rdf:type`, via the instance index below) and properties to `select_property_siblings` — same-kind properties sharing at least one declared `rdfs:domain`, with data properties additionally needing comparable ranges. A property with no declared domains yields `[]`; on the OAEI Knowledge Graph corpus, which contains no `rdfs:domain` triples, every property query therefore falls back to the caller's donor rule, with the reason recorded rather than hidden.

Strategy choice is resolved by `resolve_sibling_strategy(configured, ontology_domain)`, importable without owlready2 so config validation stays cheap: explicit `prompts.sibling_strategy` wins, then any registered domain override (the override table is empty by default), then `sbert` as the generic fallback.

## The instance index

Instances have no class hierarchy, so a sibling instance is another individual sharing a type. `build_instance_type_index(ontology)` in `logmap_llm/ontology/instance_index.py` makes two passes over the rdflib graph — labels first (the lexicographically smallest label wins), then `rdf:type` — and buckets subjects per type, each bucket sorted by `(label, uri)`. Buckets hold lightweight `IndexedInstance` records rather than `InstanceEntity` objects, because constructing the latter would trigger a graph walk per candidate per query. `InstanceTypeIndex.candidates(type_uris, exclude_uri=..., limit=...)` prefers specific types over the uninformative trio (`owl:Thing`, `rdfs:Resource`, `owl:NamedIndividual`), tags each candidate with which kind was used so the fallback rate is reportable, and applies the limit while streaming so an oversized bucket (dbkwik's largest holds around 44k members) is never materialised. The index is memoised per ontology inside `SiblingSelector`; a failed build is memoised as the exception and re-raised, never degraded to "no siblings". Each index carries a `fingerprint` — a hash of the schema version, ontology IRI and triple count — as a cheap content address for provenance.

## Vocabularies

`logmap_llm/ontology/vocabularies.py` describes how a KG family encodes annotation conventions, as a frozen, JSON-serialisable `OntologyConventionVocabulary` with an ordered `abstract_predicates` tuple (first with a value wins), unioned `category_predicates`, `extra_handled_predicates` excluded from the generic property sweep in `getInstanceContext`, URI-substring hints, and the `language_filter`. Three presets are registered:

| Preset | Used for | Abstract source | Language filter |
| --- | --- | --- | --- |
| `default` | most tracks | `rdfs:comment` | `en` + untagged |
| `dbpedia_family` | OAEI Knowledge Graph track | DBkWik/DBpedia abstract, then `rdfs:comment` | `en` + untagged |
| `multilingual_skos` | Digital Humanities track | `skos:definition`, `skos:scopeNote`, `rdfs:comment` | all languages |

The preset is selected with `ontology_vocabulary` under `[alignmentTask]` in the [configuration](configuration.md) (default `"default"`; unknown names fail validation via `get_preset`). Custom vocabularies can be added with `register_preset`. The vocabulary also seeds `compute_predicate_entropies`, which scores predicates matching `property_uri_substring` by the Shannon entropy of their value distributions, cached in memory per pattern and on disk under the same cache root at `logmap-llm/entropies/`, in a file keyed by the ontology's path, size and mtime plus the pattern.

## Determinism

Prompt text must be reproducible across runs and hosts, so ordering is pinned wherever the underlying stores do not guarantee it: labels are sorted, `getInstanceContext` sorts types, properties and categories when `deterministic=True` (the default), class restrictions sort by `(property_name, filler_name)`, single-label selection uses `min(...)` over label sets, sibling candidates and ties break on `(label, IRI)`, and instance-index buckets are sorted at build time. Together with a pinned encoder revision and device, this keeps sibling context and constructed negatives byte-stable given the same inputs.
