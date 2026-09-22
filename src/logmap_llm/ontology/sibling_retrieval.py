"""
logmap_llm.ontology.sibling_retrieval

Sibling selection with varying strategies; selects the top-k most "appropriate" siblings

Use in:

  (1) prompt context construction (hierarchical / sibling-context templates), and
  (2) few-shot hard-negative generation (contrastive near-miss examples)

TODO (aside): a hierarchy-aware ontology embedding could construct near-miss parents
for hierarchical context (rank by s(C \sqsubseteq D), diff against the actual parents);
computationally demanding, but maybe worth exploring if the near-miss case proves
helpful — mainly for subsumption-based ranking on tasks that ship a subsumption reference.

The selection algorithm, for a class:

    1. Given C, gather direct parents of C: parents(C) [multi-inheritance aware]
    2. Union direct children of every parent into candidate set C^HAT_ch
    3. Sibling set S = C^HAT_ch \ {C}
    4. If |S| > max_candidates, slice S to max_candidates (cost cap).
    5. If |S| <= k, short-circuit and return all (score = 1.0) — unless force_rank.
    6. Otherwise rank S by the configured strategy and return top-k.

Instances and properties are supported too:

  * INST — instances have no hierarchy to walk, so a sibling is another
    individual sharing a type, from a per-ontology index built once
    (ontology/instance_index.py). Specific types are preferred over owl:Thing,
    and which was used is carried on the candidate so the rate is reportable.
  * OPROP/DPROP — properties sharing a declared rdfs:domain (and, for DPROP, a
    comparable range kind). The OAEI KG ontologies contain no rdfs:domain
    triples, so this returns [] for every KG property query and the caller
    falls back with a recorded reason. See select_property_siblings.

force_rank exists because the step-5 short-circuit is a cost optimisation for
prompt-context templates and is wrong for negative construction: it would hand
back "the sibling that sorts first" under the configured semantic strategy's
name.

Four strategies are available, exposed via SiblingSelectionStrategy:

    ALPHANUMERIC      — sort by rdfs:label, ascending (naive baseline).
    SHORTEST_LABEL    — sort by len(rdfs:label), ascending (naive baseline).
    CLS_TRANSFORMER   — embed labels with a CLS-pooled encoder (domain-specialised).
    SBERT             — embed labels with all-MiniLM-L12-v2 (mean pooling, generic).

The two embedding strategies share the same nearest-by-cosine ranking; they
differ only in the underlying encoder; embedding models are loaded lazily,
therefore, naive strategies pay no transformer-load cost.
"""
from __future__ import annotations

import numpy as np

from logmap_llm.constants import (
    DEFAULT_CLS_ENCODER_MODEL,
    DEFAULT_GENERAL_MODEL,
    DEFAULT_MAX_SIBLING_CANDIDATES,
    DEFAULT_TOP_K,
    VERBOSE,
    VERY_VERBOSE,
)
from logmap_llm.utils.logging import (
    debug,
    warn,
    warning,
    info,
    success,
    critical,
)
from logmap_llm.ontology.object import ClassEntity, InstanceEntity, PropertyEntity
from logmap_llm.ontology.sibling_strategy import SiblingSelectionStrategy
from logmap_llm.ontology.instance_index import build_instance_type_index


# Re-exported: the enum and its resolution rule live in a module with no ontology
# dependency, so configuration validation can use them without importing owlready2.
SiblingSelectionStrategy = SiblingSelectionStrategy


###
# HELPERS
###


def _local_name(iri: str) -> str:
    return str(iri).rstrip("/#").rsplit("#", 1)[-1].rsplit("/", 1)[-1] or str(iri)


def _iri_of(entity) -> str:
    """Best-effort IRI for any entity kind.

    ``InstanceEntity`` carries ``uri`` directly and never sets ``annotation``, so ``.iri``
    raises AttributeError on it; ranking and stable ordering use this helper instead.
    """
    for accessor in ("iri", "uri"):
        try:
            value = getattr(entity, accessor, None)
        except Exception:
            value = None
        if value:
            return str(value)
    annotation = getattr(entity, "annotation", None)
    if isinstance(annotation, dict) and annotation.get("uri"):
        return str(annotation["uri"])
    return ""


def _get_label(entity) -> str:
    """
    Return the entity's preferred label, or a fragment of its IRI as fallback.
    NOTE: multiple preferred terms may exist (e.g. "Widget" and "Widget (SEMANTIC_TAG)"),
    so taking min over the set is _a little hacky_, but works for now
    """
    names = entity.get_preferred_names()
    if names:
        return min(names)
    thing_class = getattr(entity, "thing_class", None)
    if thing_class is not None and getattr(thing_class, "name", None):
        return str(thing_class.name)
    return _local_name(_iri_of(entity))


def _stable_entity_key(entity) -> tuple[str, str]:
    """Order equal-label entities by IRI rather than set iteration order."""
    return (_get_label(entity), _iri_of(entity))


def property_sibling_candidates(entity, pool) -> list:
    """Which properties in ``pool`` are siblings of ``entity``: the rule, isolated.

    A sibling property shares at least one declared domain, and for a data property
    additionally has a comparable range kind (overlapping datatypes, or both undeclared).
    Split out from ``select_property_siblings`` so the rule can be tested without owlready2.
    """
    domains = set(entity.get_domain_names() or ())
    if not domains:
        return []
    is_data = bool(getattr(entity, "is_data_property", False))
    ranges = set(entity.get_range_names() or ())
    own_iri = _iri_of(entity)

    out = []
    for candidate in pool:
        candidate_iri = _iri_of(candidate)
        if not candidate_iri or candidate_iri == own_iri:
            continue
        if bool(getattr(candidate, "is_data_property", False)) != is_data:
            continue
        if not (set(candidate.get_domain_names() or ()) & domains):
            continue
        if is_data:
            candidate_ranges = set(candidate.get_range_names() or ())
            if not (candidate_ranges & ranges) and (candidate_ranges or ranges):
                continue
        out.append(candidate)
    return out


def _gather_siblings(entity: ClassEntity) -> set:
    """
    Multi-inheritance-aware sibling gathering:
        S = { cup_[p \in parents(C)] children(p) } \ { C }
    """
    siblings: set = set()
    for parent in entity.get_direct_parents():
        for child in parent.get_direct_children():
            if child != entity:
                siblings.add(child)
    return siblings


###
# MAIN (MODULE/CLASS)
###


class SiblingSelector:
    """
    Top-k sibling selection with a pluggable scoring strategy.

    usage (interface):

        selector = SiblingSelector(strategy=SiblingSelectionStrategy.CLS_TRANSFORMER)
        ranked   = selector.select_siblings(entity, max_count=2)
        # -> list[(label, score)] sorted by descending score

    embedding strategies cache embeddings by entity IRI across the run;
    this helps when the same class appears in multiple M_ask candidates
    """

    def __init__(
        self,
        strategy: SiblingSelectionStrategy | str = SiblingSelectionStrategy.CLS_TRANSFORMER,
        model_name_or_path: str | None = None,
        max_candidates: int = DEFAULT_MAX_SIBLING_CANDIDATES,
        batch_size: int = 64,
        max_length: int = 64,
        model_revision: str | None = None,
        device: str | None = None,
    ):
        self.strategy = SiblingSelectionStrategy(strategy)
        # Declared, not discovered: cuda and cpu matmuls differ in the last bits, so a
        # near-tie between candidate siblings can resolve differently per host and build
        # different negatives on different machines. The resolved value is recorded in
        # the trace.
        self._configured_device = str(device).strip() if device else None
        # An immutable checkpoint commit: without it a later model update silently changes
        # the experiment, and an offline cache holding the pinned snapshot cannot be loaded
        # (an unpinned from_pretrained resolves refs/main, which the cache lacks).
        self._configured_revision = str(model_revision).strip() if model_revision else None
        self.max_candidates = max_candidates
        self.batch_size = batch_size
        self.max_length = max_length

        # embedding-only state (loaded lazily)
        self._tokenizer = None
        self._model = None
        self._device = None
        self._pooling: str | None = None
        self._cache: dict[str, np.ndarray] = {}
        self._model_name_or_path: str | None = None
        # type -> instances indexes, memoised per ontology object (never per query)
        self._instance_indexes: dict[int, object] = {}

        if self.strategy.is_embedding_based:
            self._init_embedding_backend(model_name_or_path)

        elif model_name_or_path is not None:
            warn(f"model_name_or_path={model_name_or_path!r} ignored: ")
            warn(f"  strategy {self.strategy.value} does not use an embedding model.\n")

        info(f"SiblingSelector ready ... with: ")
        info(f"  strategy={self.strategy.value}, max_candidates={self.max_candidates}.\n")


    ###
    # EMBEDDING BACKEND (lazy)
    ###

    def _init_embedding_backend(self, model_name_or_path: str | None) -> None:

        # imports kept local so that the naive strategies do not pay the
        # torch/transformers import cost when the user picks them (can be quite heavy)

        if VERBOSE:
            debug(f"(_init_embedding_backend) Importing torch; and transformers (AutoTokenizer, AutoModel).")

        import torch  # noqa: WPS433
        from transformers import AutoTokenizer, AutoModel  # noqa: WPS433

        if model_name_or_path is None:

            model_name_or_path = (
                DEFAULT_CLS_ENCODER_MODEL
                if self.strategy is SiblingSelectionStrategy.CLS_TRANSFORMER
                else DEFAULT_GENERAL_MODEL
            )

        if model_name_or_path is None:
            # Only reachable for CLS_TRANSFORMER, which has no in-source default checkpoint.
            # Refuse rather than substitute the generic encoder: that would run a different
            # experiment under the configured strategy's name.
            raise ValueError(
                f"sibling strategy '{self.strategy.value}' has no default checkpoint; set "
                "prompts.sibling_model to the encoder this run should use (and pin "
                "prompts.sibling_model_revision). Refusing to fall back to the generic encoder."
            )

        # pooling is implied by strategy:
        #   (1) the specialised encoder was trained with a contrastive [CLS] objective
        #   (2) sentence-transformers were trained with mean pooling
        #
        # mismatched pooling silently degrades embedding quality, so the caller
        # cannot override this (if anything, belongs in expert config)

        self._pooling = "cls" if self.strategy is SiblingSelectionStrategy.CLS_TRANSFORMER else "mean"
        self._device = torch.device(
            self._configured_device
            if self._configured_device
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        info(f"Loading embedding backend: model={model_name_or_path} pooling={self._pooling} device={self._device}")

        revision = self._configured_revision
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        self._model = AutoModel.from_pretrained(
            model_name_or_path, revision=revision,
        ).to(self._device)
        self._model.eval()
        self._model_name_or_path = model_name_or_path


    @property
    def device(self):
        """For backwards compatibility with previous logging in stage_two."""
        return self._device if self._device is not None else "cpu (no model loaded)"


    def _pool(self, last_hidden_state, attention_mask):
        if self._pooling == "cls":
            return last_hidden_state[:, 0, :]
        mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts and L2-normalise rows."""

        if VERBOSE and VERY_VERBOSE:
            debug(f"(_embed_batch) Importing torch.")

        import torch  # noqa: WPS433
        all_rows: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            inputs = self._tokenizer(
                chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            pooled = self._pool(outputs.last_hidden_state, inputs["attention_mask"])
            arr = pooled.cpu().numpy()
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            all_rows.append(arr / norms)

        return np.vstack(all_rows) if all_rows else np.zeros((0, 0), dtype=np.float32)


    def _embed_with_cache(self, label: str, iri: str | None) -> np.ndarray:
        if iri is not None and iri in self._cache:
            return self._cache[iri]
        emb = self._embed_batch([label])[0]
        if iri is not None:
            self._cache[iri] = emb
        return emb


    ###
    # SELECTION
    ###


    @property
    def model_revision(self) -> str:
        """The resolved checkpoint commit for an embedding strategy, or "" for the naive ones.

        transformers stores the revision it actually loaded on the model config, so the
        sibling regime can be recorded in the retrieval trace. Naive strategies load no
        model and have no revision.
        """
        if self._model is None:
            return ""
        if self._configured_revision:
            return self._configured_revision
        config = getattr(self._model, "config", None)
        return str(getattr(config, "_commit_hash", "") or "")

    def select_siblings(
        self,
        entity: ClassEntity,
        max_count: int = DEFAULT_TOP_K,
        max_candidates: int | None = None,
        *,
        force_rank: bool = False,
    ) -> list[tuple[ClassEntity, float]]:
        """
        Return up to max_count sibling ClassEntity objs ranked by the configured strategy
        max_candidates overrides the per-call cost cap (default: this inst max_candidates)
        note that every result is a (ClassEntity, score) tuple ranked by descending score
        callers that require labels only (ie. template fns) must call _get_label(class_entity)

        PropertyEntity dispatches to select_property_siblings (shared declared domain) and
        InstanceEntity to select_instance_siblings (shared-type ranking via the instance
        type index) — see the module docstring for the semantics.
        """
        if isinstance(entity, InstanceEntity):
            return self.select_instance_siblings(
                entity, max_count=max_count, max_candidates=max_candidates,
                force_rank=force_rank,
            )
        if isinstance(entity, PropertyEntity):
            return self.select_property_siblings(
                entity, max_count=max_count, max_candidates=max_candidates,
                force_rank=force_rank,
            )
        if not isinstance(entity, ClassEntity):
            critical(f"Sibling selection is defined for classes, properties and instances only.")
            raise NotImplementedError(
                f"SiblingSelector.select_siblings received {type(entity).__name__}."
            )

        cap = max_candidates if max_candidates is not None else self.max_candidates

        if VERBOSE and VERY_VERBOSE:
            debug(f"(select_siblings) max_candidates set to {str(cap)} and max_count set to {str(max_count)}.")

        # cls -> direct parent/s -> unionised children, sorted for determinism
        siblings = sorted(_gather_siblings(entity), key=_stable_entity_key)
        if not siblings:
            return []
        if len(siblings) > cap:
            siblings = siblings[:cap]

        # dont rank if |S| <= k: a cost optimisation for prompt-context templates, which
        # only need *some* siblings to show. force_rank disables it — negative construction
        # would otherwise silently receive siblings in alphabetical order with a flat 1.0
        # score under the configured embedding strategy's name.
        if len(siblings) <= max_count and not force_rank:
            return [(sib, 1.0) for sib in siblings]

        # otherwise, rank by employed strategy
        return self._rank(entity, siblings, max_count)


    def select_instance_siblings(
        self,
        entity: InstanceEntity,
        max_count: int = DEFAULT_TOP_K,
        max_candidates: int | None = None,
        *,
        force_rank: bool = False,
    ) -> list[tuple[object, float]]:
        """Instances sharing a type with ``entity``, ranked by the configured strategy.

        Instances have no class hierarchy to walk, so "sibling" here means "another
        individual of the same type". Specific types are preferred over ``owl:Thing`` and
        friends, and which was used is carried on each candidate so the rate is reportable
        (see instance_index).
        """
        types = list(entity.get_type_uris() or ())
        if not types:
            return []
        index = self._instance_index(entity.onto)
        if index is None:
            return []
        cap = max_candidates if max_candidates is not None else self.max_candidates
        # The cap is applied inside the index so an oversized bucket is never materialised:
        # dbkwik's largest is ~44k members, and the caller keeps 50.
        candidates, _specificity = index.candidates(
            types, exclude_uri=str(entity.uri), limit=cap,
        )
        if not candidates:
            return []
        if len(candidates) <= max_count and not force_rank:
            return [(candidate, 1.0) for candidate in candidates]
        return self._rank(entity, candidates, max_count)


    def select_property_siblings(
        self,
        entity: PropertyEntity,
        max_count: int = DEFAULT_TOP_K,
        max_candidates: int | None = None,
        *,
        force_rank: bool = False,
    ) -> list[tuple[object, float]]:
        """Properties of the same kind sharing a declared domain (and comparable range).

        On the current KG corpus this returns ``[]`` for every property query: the OAEI
        Knowledge Graph ontologies contain no ``rdfs:domain`` triples, and ``kg-abox``-derived
        predicates get empty domain and range by construction
        (``PropertyEntity._init_undeclared``). Every OPROP/DPROP negative there falls back to
        the donor rule — reported per lane, not hidden. Whether a domain-free,
        ABox-evidence-based property-sibling notion should exist is an open
        experimental-design question.
        """
        domains = set(entity.get_domain_names() or ())
        if not domains:
            return []
        onto = getattr(entity, "onto", None)
        if onto is None:
            return []
        is_data = bool(getattr(entity, "is_data_property", False))
        ranges = set(entity.get_range_names() or ())

        # Pool-fetch exceptions propagate: the sibling_fn wrapper (rag_fewshot.make_sibling_fn)
        # records them as 'sibling-lookup-failed:<type>' rather than the false 'no-siblings'.
        # list(): the accessors can return single-use generators (see access.py), and the
        # failure count below needs the pool size.
        pool = list(onto.getDataProperties() if is_data else onto.getObjectProperties())

        resolved = []
        construction_failures = 0
        for raw in pool:
            if hasattr(raw, "get_domain_names"):
                resolved.append(raw)                      # already a PropertyEntity
                continue
            try:
                resolved.append(PropertyEntity(str(getattr(raw, "iri", "") or ""), onto))
            except Exception:
                construction_failures += 1
        if construction_failures:
            warn(f"select_property_siblings: {construction_failures}/{len(pool)} pool "
                 f"entries failed PropertyEntity construction and were dropped.")

        candidates = property_sibling_candidates(entity, resolved)
        if not candidates:
            return []
        candidates.sort(key=_stable_entity_key)
        cap = max_candidates if max_candidates is not None else self.max_candidates
        if len(candidates) > cap:
            candidates = candidates[:cap]
        if len(candidates) <= max_count and not force_rank:
            return [(candidate, 1.0) for candidate in candidates]
        return self._rank(entity, candidates, max_count)


    def _instance_index(self, onto):
        """Memoised per-ontology type index, built once, never per query.

        A failed build is also memoised, and re-raised rather than degraded to None:
        raising lets the sibling_fn wrapper record the true 'sibling-lookup-failed:<type>'
        reason per negative, and memoising avoids re-attempting the expensive graph walk
        on every INST query.
        """
        if onto is None:
            return None
        key = id(onto)
        if key in self._instance_indexes:
            cached = self._instance_indexes[key]
            if isinstance(cached, Exception):
                raise cached  # memoised build failure
            return cached
        try:
            index = build_instance_type_index(onto)
        except Exception as exc:
            self._instance_indexes[key] = exc
            raise
        self._instance_indexes[key] = index
        info(f"Instance type index built: {len(index.by_type)} types "
             f"(fingerprint {index.fingerprint[:12]}).")
        return index


    ###
    # RANKING STRATEGIES
    ###

    def _rank(self, entity: ClassEntity, siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        if self.strategy is SiblingSelectionStrategy.ALPHANUMERIC:
            return self._rank_alphanumeric(siblings, k)
        if self.strategy is SiblingSelectionStrategy.SHORTEST_LABEL:
            return self._rank_shortest_label(siblings, k)
        # embedding strategies share an implementation
        return self._rank_by_embedding(entity, siblings, k)


    @staticmethod
    def _rank_alphanumeric(siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        ranked = sorted(siblings, key=_stable_entity_key)
        return [(sib, 1.0) for sib in ranked[:k]]


    @staticmethod
    def _rank_shortest_label(siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        ranked = sorted(
            siblings,
            key=lambda s: (len(_get_label(s)), *_stable_entity_key(s)),
        )
        return [(sib, 1.0) for sib in ranked[:k]]


    def _rank_by_embedding(self, entity: ClassEntity, siblings: list, k: int) -> list[tuple[ClassEntity, float]]:
        """
        collects labels + IRIs and batches the uncached entities in a single forward pass;
        then score entity emb @ sib emb (for all sib embs), sort by descending score &
        return top-k as (ClassEntity, score) pairs (so callers can look siblings up by IRI)
        """
        entity_label = _get_label(entity)
        entity_iri = _iri_of(entity) or None
        entity_emb = self._embed_with_cache(entity_label, entity_iri)

        sib_payload: list[tuple[ClassEntity, str, str | None]] = []
        for sib in siblings:
            sib_payload.append(
                (sib, _get_label(sib), _iri_of(sib) or None)
            )

        uncached_idx = []
        for idx, (_sib, _label, iri) in enumerate(sib_payload):
            if iri is None or iri not in self._cache:
                uncached_idx.append(idx)

        if uncached_idx:
            uncached_labels = [sib_payload[idx][1] for idx in uncached_idx]
            embs = self._embed_batch(uncached_labels)
            for slot, row in zip(uncached_idx, embs):
                _sib, _label, iri = sib_payload[slot]
                if iri is not None:
                    self._cache[iri] = row

        scored: list[tuple[ClassEntity, float]] = []

        for sib, label, iri in sib_payload:
            if iri is not None and iri in self._cache:
                emb = self._cache[iri]
            else:
                # no IRI fallback, re-embeds single label
                emb = self._embed_batch([label])[0]
            # score := dot product entity_emb \cdot sib_emb
            # ^^^^ cosine-sim (since embs are L_2 norm'd)
            scored.append((sib, float(entity_emb @ emb)))

        # maximise by score (sort by DESC), then return top-k
        scored.sort(key=lambda lt: (-lt[1], *_stable_entity_key(lt[0])))

        return scored[:k]
