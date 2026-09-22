"""Embedding index for a typed example pool: builds (or loads from a keyed on-disk
cache) an L2-normalised embedding matrix aligned to a list of Examples, and ranks a
query embedding against it with a deterministic tie-break.

Cache key = sha256(dataset_sha, encoder repo+revision, encoder preprocessing_version,
retriever preprocessing_version, language, entity kind, corpus_hash, encoder runtime versions).
A stale/corrupt/mismatched cache is rebuilt and the reason recorded — never silently served.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import hashlib
import os
import tempfile

import numpy as np

from .encoder import Encoder
from .types import EntityKind


def compute_index_hash(*, dataset_sha: str, encoder_repo: str, encoder_revision: str,
                       encoder_preprocessing: str, retriever_preprocessing: str,
                       language: str, kind: EntityKind, corpus_hash: str,
                       encoder_runtime: str) -> str:
    """Content address for a typed embedding index.

    `encoder_runtime` must be part of the key: runs identical in every other component
    can differ in `torch`/`transformers` versions and produce different embeddings, so
    omitting it would let one run silently serve another's cache.
    """
    h = hashlib.sha256()
    for part in (dataset_sha, encoder_repo, encoder_revision, encoder_preprocessing,
                 retriever_preprocessing, language, kind.value, corpus_hash,
                 encoder_runtime):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


@dataclass
class EmbeddingIndex:
    examples: list                 # list[Example], aligned row-for-row with `matrix`
    matrix: np.ndarray             # (n, dim), L2-normalised rows
    index_hash: str
    rebuilt_reason: Optional[str] = None   # None if served from a valid cache; else why we rebuilt

    # construction / caching

    @classmethod
    def build(cls, examples: list, encoder: Encoder, index_hash: str,
              cache_dir: Optional[str] = None) -> "EmbeddingIndex":
        example_ids = [ex.example_id for ex in examples]
        if cache_dir is not None:
            cached = cls._try_load(cache_dir, index_hash, example_ids, examples)
            if cached is not None:
                return cached
        rebuilt_reason = None
        if cache_dir is not None:
            rebuilt_reason = "cache-miss-or-invalid"
        texts = [ex.embed_text for ex in examples]
        matrix = encoder.encode(texts) if texts else np.zeros((0, 1), dtype=np.float64)
        idx = cls(examples=list(examples), matrix=matrix, index_hash=index_hash,
                  rebuilt_reason=rebuilt_reason)
        if cache_dir is not None:
            idx._save(cache_dir)
        return idx

    @classmethod
    def _cache_path(cls, cache_dir: str, index_hash: str) -> str:
        return os.path.join(cache_dir, f"ragindex-{index_hash}.npz")

    @classmethod
    def _try_load(cls, cache_dir: str, index_hash: str, example_ids: list,
                  examples: list) -> Optional["EmbeddingIndex"]:
        path = cls._cache_path(cache_dir, index_hash)
        if not os.path.isfile(path):
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                stored_key = str(data["index_hash"].item()) if "index_hash" in data else None
                stored_ids = list(data["example_ids"]) if "example_ids" in data else None
                matrix = data["matrix"]
            # validate: key + example_ids + row count must all match, else stale/corrupt
            if stored_key != index_hash:
                return None
            if stored_ids is None or [str(x) for x in stored_ids] != [str(x) for x in example_ids]:
                return None
            if matrix.shape[0] != len(examples):
                return None
        except Exception:
            # corrupt file -> treat as miss (caller rebuilds)
            return None
        return cls(examples=list(examples), matrix=np.asarray(matrix, dtype=np.float64),
                   index_hash=index_hash, rebuilt_reason=None)

    def _save(self, cache_dir: str) -> None:
        os.makedirs(cache_dir, exist_ok=True)
        path = self._cache_path(cache_dir, self.index_hash)
        # Unique temp file per writer so concurrent cache warmers cannot clobber each
        # other's partial files; each complete archive is published atomically (one
        # valid equivalent archive may replace another).
        fd, tmp = tempfile.mkstemp(
            dir=cache_dir,
            prefix=f".ragindex-{self.index_hash}-",
            suffix=".partial",
        )
        try:
            # np.savez appends ".npz" to string paths, so write via the open handle.
            with os.fdopen(fd, "wb") as fh:
                np.savez(
                    fh,
                    index_hash=np.array(self.index_hash),
                    example_ids=np.array(
                        [ex.example_id for ex in self.examples], dtype=object
                    ).astype(str),
                    matrix=self.matrix,
                )
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        finally:
            # os.replace consumed tmp on success; this cleans up interrupted or
            # failed writers without touching another process's file.
            try:
                os.unlink(tmp)
            except FileNotFoundError:
                pass

    # ranking

    def rank(self, query_vec: np.ndarray, exclude_ids: Optional[set] = None,
             exclude_keys: Optional[set] = None) -> list:
        """
        Return [(Example, similarity)] sorted by (-similarity, example_id) — a stable,
        fully deterministic order. `exclude_ids` drops examples by id; `exclude_keys`
        drops by complete-IRI pair-key (query, reverse, M_ask, known positives, ...).
        """
        exclude_ids = exclude_ids or set()
        exclude_keys = exclude_keys or set()
        if self.matrix.shape[0] == 0:
            return []
        q = np.asarray(query_vec, dtype=np.float64).reshape(-1)
        sims = self.matrix @ q  # cosine (both normalised)
        scored = []
        for i, ex in enumerate(self.examples):
            if ex.example_id in exclude_ids:
                continue
            if ex.unordered in exclude_keys:
                continue
            scored.append((float(sims[i]), ex))
        # deterministic: highest similarity first, ties broken by example_id
        scored.sort(key=lambda t: (-t[0], t[1].example_id))
        return [(ex, sim) for sim, ex in scored]
