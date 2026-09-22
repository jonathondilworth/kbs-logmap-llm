"""
logmap_llm.oracle.rag.encoder

Similarity encoders behind a single Protocol so the retriever is testable on CPU without
torch/transformers/GPU. Production uses a CLS-pooled transformer encoder; unit tests use the
deterministic HashingEncoder.

Contract: encode(texts) -> np.ndarray of shape (len(texts), dim), L2-normalised rows, so a
plain dot product is cosine similarity. `repo`, `revision`, `preprocessing_version` feed the
index cache key.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable
import hashlib

import numpy as np


@runtime_checkable
class Encoder(Protocol):
    repo: str
    revision: str
    preprocessing_version: str
    #: The library versions that actually produced the embeddings. Part of the index cache
    #: key, because a cache rebuilt under a different `transformers` holds different vectors
    #: while every other key component is unchanged (see index.compute_index_hash).
    runtime_versions: str

    def encode(self, texts: list[str]) -> np.ndarray: ...


def _torch_runtime() -> str:
    """`torch=<v>;transformers=<v>` for the modules already imported by an encoder."""
    import torch
    import transformers

    return f"torch={torch.__version__};transformers={transformers.__version__}"


def _l2_normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return mat / norms


def cls_pool_preprocessing_version(max_length: int) -> str:
    """ClsPooledEncoder's preprocessing identity (an index-cache-key component).

    Module-level so the CPU test suite can pin the format without torch. The string
    must change whenever truncation changes, or runs with different max_length would
    share one cache entry holding different vectors.
    """
    return f"cls-pool-max{int(max_length)}-v1"


class HashingEncoder:
    """
    Deterministic, dependency-free encoder for tests and CPU smoke runs.

    Embeds text as a hashed character n-gram bag projected to a fixed dimension, L2-normalised.
    It is a genuine similarity signal (shared substrings -> higher cosine) and is fully
    deterministic given (text, dim, ngram, seed) — no RNG, no external model. It is not a
    semantic model; it exists so the retriever's selection logic can be exercised offline.
    """

    def __init__(self, dim: int = 256, ngram: tuple[int, int] = (2, 4), seed: int = 0):
        self.dim = int(dim)
        self.ngram = ngram
        self.seed = int(seed)
        self.repo = "local-hashing-encoder"
        self.revision = f"dim{self.dim}-ng{ngram[0]}_{ngram[1]}-s{seed}"
        self.preprocessing_version = "lower-strip-v1"
        # Pure Python and hashlib: no library whose version could change the vectors.
        self.runtime_versions = "pure-python"

    def _preprocess(self, text: str) -> str:
        return " ".join(str(text).lower().split())

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        s = self._preprocess(text)
        if not s:
            return vec
        lo, hi = self.ngram
        for n in range(lo, hi + 1):
            if n > len(s):
                break
            for i in range(len(s) - n + 1):
                gram = s[i:i + n]
                h = hashlib.blake2b(f"{self.seed}:{gram}".encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if (h[4] & 1) else -1.0
                vec[idx] += sign
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float64)
        mat = np.vstack([self._vector(t) for t in texts])
        return _l2_normalise(mat)


class ClsPooledEncoder:
    """
    Production encoder wrapping a CLS-pooled transformer checkpoint. torch/transformers are
    imported lazily, so importing this module never requires them; only instantiating
    ClsPooledEncoder does. Not used by the unit test suite.
    """

    def __init__(self, model_name_or_path: str,
                 revision: str | None = None, device: str | None = None, batch_size: int = 64,
                 max_length: int = 64):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as e:  # pragma: no cover - exercised only in production envs
            raise RuntimeError(
                "ClsPooledEncoder requires torch + transformers; use HashingEncoder for CPU tests."
            ) from e
        self._torch = torch
        self.repo = model_name_or_path
        # An unpinned revision is recorded as such, not resolved: the checkpoint identity
        # then rests on the caller having pinned it (build_rag_encoder enforces that).
        self.revision = revision or "unpinned"
        self.batch_size = batch_size
        self.max_length = max_length
        # Index-cache-key component (index.compute_index_hash). Includes max_length because
        # truncation changes the vectors; runs differing only in max_length must not share
        # a cache entry.
        self.preprocessing_version = cls_pool_preprocessing_version(self.max_length)
        self.runtime_versions = _torch_runtime()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, revision=revision)
        self._model = AutoModel.from_pretrained(model_name_or_path, revision=revision).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if len(texts) == 0:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float64)
        torch = self._torch
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = [str(t) for t in texts[i:i + self.batch_size]]
                enc = self._tokenizer(batch, padding=True, truncation=True,
                                      max_length=self.max_length, return_tensors="pt").to(self.device)
                # CLS-pooling (the convention this checkpoint family was trained under)
                cls = self._model(**enc).last_hidden_state[:, 0, :]
                out.append(cls.cpu().numpy())
        return _l2_normalise(np.vstack(out).astype(np.float64))


class SbertEncoder:
    """Sentence-transformer-compatible encoder with explicit masked-mean pooling.

    The pooling rule, truncation limit, model revision, and normalisation are
    visible parts of the experimental identity rather than library defaults.
    """

    def __init__(self, model_name_or_path: str, revision: str,
                 device: str | None = None, batch_size: int = 64,
                 max_length: int = 128):
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except Exception as exc:  # pragma: no cover - production dependency path
            raise RuntimeError(
                "SbertEncoder requires torch + transformers; use HashingEncoder for CPU tests."
            ) from exc

        if not str(revision).strip():
            raise ValueError("SbertEncoder requires an immutable model revision")
        if int(max_length) < 1:
            raise ValueError("SbertEncoder max_length must be positive")

        self._torch = torch
        self.repo = str(model_name_or_path)
        self.revision = str(revision)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.preprocessing_version = f"sbert-maskmean-max{self.max_length}-v1"
        self.runtime_versions = _torch_runtime()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.repo, revision=self.revision
        )
        self._model = AutoModel.from_pretrained(
            self.repo, revision=self.revision
        ).to(self.device).eval()

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.config.hidden_size), dtype=np.float64)
        torch = self._torch
        output = []
        with torch.no_grad():
            for offset in range(0, len(texts), self.batch_size):
                batch = [str(value) for value in texts[offset:offset + self.batch_size]]
                encoded = self._tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.max_length, return_tensors="pt",
                ).to(self.device)
                hidden = self._model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                output.append(pooled.cpu().numpy())
        return _l2_normalise(np.vstack(output).astype(np.float64))
