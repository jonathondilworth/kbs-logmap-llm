"""
logmap_llm.oracle.rag.corpus

Typed example corpora for RAG retrieval. Maintains disjoint CLS/OPROP/DPROP/INST pools of
positive examples (and optional explicit negatives), all keyed by complete IRIs — never
deduplicated by local name, so the same local name in different namespaces stays distinct.

Positives come only from authorised sources: training alignments (Source.GOLD) or separately
labelled high-confidence LogMap anchors (Source.ANCHOR, a pseudo-label), tracked distinctly
so the trace can tell them apart.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable
import hashlib

from .types import Example, EntityKind, Source


def _iri_pair_key(src: str, tgt: str) -> frozenset:
    """Direction-agnostic key over complete IRIs (dedup / exclusion). Not local-name based."""
    return frozenset({src, tgt})


@dataclass
class TypedCorpus:
    """Per-kind pools of positive and explicit-negative examples."""
    positives: dict[EntityKind, list[Example]] = field(
        default_factory=lambda: {k: [] for k in EntityKind})
    negatives: dict[EntityKind, list[Example]] = field(
        default_factory=lambda: {k: [] for k in EntityKind})
    dataset_sha: str = ""
    language: str = "en"

    # ---- construction -----------------------------------------------------

    def add_pairs(
        self,
        pairs: Iterable[tuple[str, str]],
        kind: EntityKind,
        source: Source,
        embed_text_fn: Callable[[str, str], str],
        payload_fn: Callable[[str, str], object] | None = None,
        label: bool = True,
        id_prefix: str | None = None,
    ) -> int:
        """
        Add (src_iri, tgt_iri) pairs of a single kind. Deduplicates by complete-IRI pair-key
        within the (kind, label) pool. Returns the number of new examples added.
        """
        pool = self.positives[kind] if label else self.negatives[kind]
        seen = {ex.unordered for ex in pool}
        prefix = id_prefix or f"{kind.value}:{source.value}:{'pos' if label else 'neg'}"
        added = 0
        for src, tgt in pairs:
            src, tgt = str(src), str(tgt)
            key = _iri_pair_key(src, tgt)
            if key in seen:
                continue
            seen.add(key)
            ex = Example(
                example_id=f"{prefix}:{added}:{_short_hash(src, tgt)}",
                src_iri=src, tgt_iri=tgt, kind=kind, label=label, source=source,
                embed_text=embed_text_fn(src, tgt),
                payload=(payload_fn(src, tgt) if payload_fn else None),
            )
            pool.append(ex)
            added += 1
        return added

    # ---- access -----------------------------------------------------------

    def positive_pool(self, kind: EntityKind) -> list[Example]:
        return list(self.positives.get(kind, []))

    def negative_pool(self, kind: EntityKind) -> list[Example]:
        return list(self.negatives.get(kind, []))

    def known_positive_keys(self, kind: EntityKind) -> set[frozenset]:
        """Complete-IRI pair-keys of all known positives of this kind (exclude from negatives)."""
        return {ex.unordered for ex in self.positives.get(kind, [])}

    def is_empty(self, kind: EntityKind) -> bool:
        return not self.positives.get(kind) and not self.negatives.get(kind)

    # ---- hashing ----------------------------------------------------------

    def corpus_hash(self, kind: EntityKind | None = None) -> str:
        """
        Deterministic hash over the corpus content. Feeds the index cache key so a changed
        corpus can never silently reuse stale vectors. Order-independent (sorted).
        """
        h = hashlib.sha256()
        h.update(f"dataset={self.dataset_sha};lang={self.language};".encode())
        kinds = [kind] if kind is not None else list(EntityKind)
        for k in kinds:
            h.update(f"|kind={k.value}|".encode())
            rows = []
            for ex in list(self.positives.get(k, [])) + list(self.negatives.get(k, [])):
                rows.append(f"{ex.example_id}\t{ex.src_iri}\t{ex.tgt_iri}\t{int(ex.label)}"
                            f"\t{ex.source.value}\t{ex.embed_text}")
            for row in sorted(rows):
                h.update(row.encode("utf-8"))
                h.update(b"\n")
        return h.hexdigest()


def _short_hash(*parts: str) -> str:
    return hashlib.blake2b("\x1f".join(parts).encode("utf-8"), digest_size=6).hexdigest()
