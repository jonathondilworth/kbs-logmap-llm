'''
IO utils
general-purpose IO module (at the project level)
'''

import json
import os
import tempfile
from pathlib import Path
import hashlib

from logmap_llm.utils.logging import debug, warning
from logmap_llm.constants import (
    DEFAULT_ENTROPY_CACHE_DIR,
    JSON_DATA,
    VERBOSE,
)


def atomic_json_write_strict(path: Path, obj: JSON_DATA, *, indent: int | None = None) -> None:
    """Atomically publish required JSON and propagate every failure.

    Unlike :func:`atomic_json_write`, this is not a best-effort cache helper.
    Completion and stage artifacts must never be reported as written after an
    I/O or serialisation failure.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(
                obj,
                fp,
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
                indent=indent,
            )
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_json_write(path: Path, obj: JSON_DATA) -> None:
    '''
    Accepts a Path (expects `*/*.json`) and any valid python object that constitutes valid JSON data,
    tries to dump the file (with the suffix `.tmp`); if successful, will write to the specified 'path'
    (used in caching operations, eg. entropy caching).
    '''
    # mkstemp gives each writer its own temp file on the same filesystem, so os.replace()
    # is atomic and concurrent writers to the same target don't corrupt each other.
    # Cache writes stay best-effort (a failure is logged, not fatal).
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp_write_loc:
            json.dump(obj, tmp_write_loc, ensure_ascii=False, sort_keys=True, allow_nan=False)  # fail loudly on NaN
        os.replace(tmp, path)  # atomic publish
    # TypeError: json.dump raises it for unserialisable objects; keep it inside the
    # best-effort contract so the temp file is cleaned up.
    except (OSError, ValueError, TypeError) as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        warning(f"atomic_json_write: failed writing {path} ({type(e).__name__}: {e}); "
                f"skipping (best-effort cache).")


def atomic_write_text_strict(path: Path, write_fn) -> None:
    """Atomically publish a required text artifact produced by ``write_fn(handle)``.

    The tmp-fsync-replace discipline of :func:`atomic_json_write_strict`, for artifacts
    that are not JSON (predictions CSV, refined-alignment TSV); a crash mid-write never
    leaves a truncated file at the final path. ``write_fn`` receives a text handle opened
    for writing (e.g. ``lambda fp: df.to_csv(fp, index=False)``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=path.name + ".", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fp:
            write_fn(fp)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def compute_entropy_disk_cache_path(local_onto_fp: str, uri_pattern: str, entropy_cache_path: Path | str = DEFAULT_ENTROPY_CACHE_DIR) -> Path | None:
    '''
    Returns a content-addressed path for persisting entropies to disk, or None if the
    ontology was not loaded from a local file (eg. resolved via HTTP \\w a URL).

    Changes to the source file invalidate the cache: the path embeds a SHA-256 hash
    (of the utf-8 encoded string):

        "'onto_path'|'onto_file_size'|'onto_file_modified'|'uri_matching_pattern'"

    TODO: migrate to a ProjectPaths obj (similar to PipelinePaths).
    '''
    entropy_cache_path = Path(entropy_cache_path)
    try:
        onto_path = Path(local_onto_fp).resolve()
        if not onto_path.is_file():
            return None
        onto_stat = onto_path.stat()
    except (OSError, AttributeError):
        if VERBOSE:
            debug("(compute_entropy_disk_cache_path) Encountered OSError or AttributeError.")
        return None

    encoded_hash_input = (
        f"{onto_path}|{onto_stat.st_size}|{onto_stat.st_mtime_ns}|{uri_pattern}"
    ).encode()

    if VERBOSE:
        debug(f"(compute_entropy_disk_cache_path) ENCODED HASH INPUT: {encoded_hash_input}")

    hash_digest = hashlib.sha256(encoded_hash_input).hexdigest()[:16]

    full_cache_filepath = (entropy_cache_path / f"{onto_path.stem}_{hash_digest}.json").expanduser().resolve()

    if VERBOSE:
        debug(f"(compute_entropy_disk_cache_path) COMPUTED HASH DIGEST (OUTPUT): {hash_digest}")
        debug(f"(compute_entropy_disk_cache_path) entropy_cache_path: {full_cache_filepath}")

    return full_cache_filepath
