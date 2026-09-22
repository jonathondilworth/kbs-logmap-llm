"""Deterministic record of the resolved Python environment a batch was generated in.

``core_sha256`` hashes only ``*.py`` under ``logmap_llm`` (the Java matcher is bound
separately via ``input_fingerprints``), so it does not bind the Python dependency set;
this payload records the package versions that can change a result.  It is an optional
top-level manifest key covered by the seal, so it cannot be edited after generation,
but it is deliberately not part of run identity: ``condition_id``, ``execution_hash``,
``_job_id``, ``_alignment_id`` and ``plan_hash`` never include it, so hosts with
different virtualenvs produce the same job identities, and cross-host comparisons
compare manifests modulo ``environment`` and ``seal``.  Defined here rather than in
``campaign/`` because ``plan.py`` cannot import from ``campaign/``.
"""
from __future__ import annotations

import platform
from importlib import metadata
from pathlib import Path
from typing import Any

#: Packages whose version can plausibly change a *result*: the encoders and their runtime,
#: the ontology and RDF layers, the oracle transport and its validation, the numeric and
#: dataframe layer, and the figure renderer. Deliberately fixed and sorted so the payload is
#: stable and reviewable rather than a full freeze of the virtualenv.
TRACKED_PACKAGES = (
    "JPype1",
    "matplotlib",
    "numpy",
    "openai",
    "owlready2",
    "pandas",
    "pydantic",
    "rdflib",
    "torch",
    "transformers",
)

SCHEMA_VERSION = 1


def resolved_packages() -> dict[str, str]:
    """Installed versions for the tracked names; absent packages are omitted.

    A missing package is normal — the ``rag`` extra is optional — and is never an error.
    ``importlib.metadata`` normalises distribution names, so ``JPype1`` and ``jpype1``
    resolve identically.
    """
    versions: dict[str, str] = {}
    for name in sorted(TRACKED_PACKAGES):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return versions


def environment_payload(repo: Path | None = None) -> dict[str, Any]:
    """The deterministic environment record embedded in a sealed manifest."""
    record: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "packages": resolved_packages(),
    }
    if repo is not None:
        lock = Path(repo) / "uv.lock"
        if lock.is_file():
            import hashlib

            digest = hashlib.sha256()
            with lock.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            record["uv_lock_sha256"] = digest.hexdigest()
    return record


def environments_differ(left: Any, right: Any) -> list[str]:
    """Names of tracked packages whose versions differ, for a legible operator message."""
    left_packages = (left or {}).get("packages", {}) if isinstance(left, dict) else {}
    right_packages = (right or {}).get("packages", {}) if isinstance(right, dict) else {}
    names = set(left_packages) | set(right_packages)
    return sorted(
        name for name in names if left_packages.get(name) != right_packages.get(name)
    )
