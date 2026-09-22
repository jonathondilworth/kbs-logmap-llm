"""
logmap_llm.ontology.cache — owlready2 quadstore persistence cache.

Caches the parsed RDF quadstore to a SQLite file on disk so that
repeated pipeline runs against the same ontology skip the expensive
OWL/RDF parse.  Uses a build-once gate with copy-on-open strategy
to avoid SQLite write-lock contention between concurrent processes.

Canonical cache location:  ~/.cache/logmap-llm/owlready2/
Process-private copies:    /tmp/logmap-llm-owlcache-*/
"""
from __future__ import annotations

import atexit
import contextlib
import fcntl
import hashlib
import os
import subprocess
import re
import shutil
import tempfile
import time
from pathlib import Path

import owlready2

from logmap_llm.utils.logging import info, success, warning

# prefix used for process-private temp directories under /tmp
_TEMP_DIR_PREFIX = 'logmap-llm-owlcache-'

# global registry of temp directories created by this process
_temp_dirs_to_cleanup: list[str] = []

# default canonical cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.environ.get(
        'XDG_CACHE_HOME', 
        os.path.expanduser('~/.cache')
    ),
    'logmap-llm',
    'owlready2',
)


def _register_temp_cleanup(temp_dir: str) -> None:
    _temp_dirs_to_cleanup.append(temp_dir)


def _cleanup_temp_dir(temp_dir: str) -> None:
    """Remove one private quadstore copy, retaining failed paths for atexit."""
    shutil.rmtree(temp_dir, ignore_errors=True)
    if not os.path.exists(temp_dir):
        with contextlib.suppress(ValueError):
            _temp_dirs_to_cleanup.remove(temp_dir)


def _cleanup_temp_dirs() -> None:
    for d in tuple(_temp_dirs_to_cleanup):
        _cleanup_temp_dir(d)


atexit.register(_cleanup_temp_dirs)


def _sanitise_filename(name: str, max_len: int = 80) -> str:
    sanitised = re.sub(r'[^a-zA-Z0-9]', '_', name)
    return sanitised[:max_len]


def _canonical_cache_path(onto_filepath: str, cache_dir: str,
                          stub_import_iris: tuple[str, ...] = ()) -> Path:
    resolved = str(Path(onto_filepath).resolve())
    # The import policy is part of the cache IDENTITY: a quadstore built with an import stubbed out
    # has different triples from one built with it resolved. Without this, changing the policy would
    # silently serve the old quadstore (the same class of trap as CR-01's mtime-only validity).
    key = resolved + ("|stub:" + ",".join(sorted(stub_import_iris)) if stub_import_iris else "")
    short_hash = hashlib.sha256(key.encode()).hexdigest()[:12]
    base_name = Path(onto_filepath).name
    sanitised = _sanitise_filename(base_name)
    cache_filename = f'{sanitised}_{short_hash}.sqlite3'
    return Path(cache_dir) / cache_filename


def _is_cache_valid(onto_filepath: str, cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    if cache_path.stat().st_size == 0:
        return False
    source_mtime = Path(onto_filepath).stat().st_mtime
    cache_mtime = cache_path.stat().st_mtime
    return cache_mtime >= source_mtime


def _format_age(seconds: float) -> str:
    if seconds < 60:
        return f'{seconds:.0f}s'
    elif seconds < 3600:
        return f'{seconds / 60:.0f}m'
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f'{hours}h {minutes}m'


def _robot_command() -> list[str] | None:
    """ROBOT as `java -jar $ROBOT_JAR` (or ~/.local/bin/robot.jar), else `robot` on PATH; None if absent."""
    jar = os.environ.get('ROBOT_JAR') or str(Path.home() / '.local' / 'bin' / 'robot.jar')
    if Path(jar).is_file():
        heap = os.environ.get('ROBOT_JAVA_HEAP', '64g')
        return [os.environ.get('JAVA', 'java'), f'-Xmx{heap}', '-jar', jar]
    robot = shutil.which('robot')
    return [robot] if robot else None


def _convert_for_owlready2(onto_filepath: str, cache_dir: str, parse_error: Exception) -> Path:
    """Convert an ontology owlready2 cannot parse (OWL functional / Manchester syntax) to OWL/XML with ROBOT, next
    to the quadstore cache, reusing a conversion that is newer than the source. OWL/XML rather than RDF/XML because
    RDF/XML cannot serialise property IRIs whose local part is not an XML name (SNOMED CT's numeric identifiers)."""
    source = Path(onto_filepath)
    target = Path(cache_dir) / f'{_sanitise_filename(source.stem)}_{hashlib.sha256(str(source.resolve()).encode()).hexdigest()[:12]}.owx'
    if target.exists() and target.stat().st_size > 0 and target.stat().st_mtime >= source.stat().st_mtime:
        success(f'Reusing OWL/XML conversion of {source.name} for owlready2')
        return target
    command = _robot_command()
    if command is None:
        raise RuntimeError(
            f'owlready2 cannot parse {source.name} ({parse_error}) and ROBOT is not available to convert it '
            f'to OWL/XML (set ROBOT_JAR or put `robot` on PATH)') from parse_error
    warning(f'owlready2 cannot parse {source.name} ({parse_error}); converting to OWL/XML with ROBOT ...')
    started = time.time()
    tmp = target.with_suffix('.owx.tmp')
    result = subprocess.run(command + ['convert', '--input', str(source), '--output', str(tmp), '--format', 'owx'],
                            capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists():
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise RuntimeError(f'ROBOT conversion of {source.name} failed: {result.stderr.strip()[-2000:]}') from parse_error
    os.replace(tmp, target)
    success(f'Converted {source.name} to OWL/XML in {time.time() - started:.0f}s ({target.stat().st_size / 2**20:.0f} MB)')
    return target


def _build_or_wait_for_cache(onto_filepath: str, cache_dir: str,
                             stub_import_iris: tuple[str, ...] = ()) -> Path:
    cache_path = _canonical_cache_path(onto_filepath, cache_dir, stub_import_iris)
    lock_path = cache_path.with_suffix('.sqlite3.lock')
    os.makedirs(cache_dir, exist_ok=True)
    onto_display = Path(onto_filepath).name

    if _is_cache_valid(onto_filepath, cache_path):
        age = time.time() - cache_path.stat().st_mtime
        success(f'Using cached quadstore for {onto_display} (cache age: {_format_age(age)})')
        return cache_path

    lock_fd = open(lock_path, 'w')
    try:
        info(f'Acquiring cache lock for {onto_display} ...', important=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        if _is_cache_valid(onto_filepath, cache_path):
            age = time.time() - cache_path.stat().st_mtime
            success(f'Using cached quadstore for {onto_display} '
                    f'(built by another process, age: {_format_age(age)})')
            return cache_path

        info(f'Building owlready2 cache for {onto_display} ...', important=True)
        build_start = time.time()

        temp_fd, temp_build_path = tempfile.mkstemp(suffix='.sqlite3.tmp', dir=cache_dir)
        os.close(temp_fd)

        try:
            build_world = owlready2.World(filename=temp_build_path)
            # STUB-IMPORTS (opt-in, empty by default -> byte-inert for every other track).
            # owlready2 follows owl:imports over HTTP and has no way to say "local only": only_local
            # is NOT propagated to imports (namespace.py:1053 calls .load() with no args), and with an
            # empty onto_path it fails on the TOP-LEVEL file instead. Pre-registering the import IRI in
            # this World and marking it loaded makes Ontology.load() short-circuit, so the import is
            # simply not followed.
            # This is what unblocks Circular-Economy: CEON.rdf imports three CEON modules and one of
            # them, .../qudt/0.1/, was NEVER PUBLISHED — it 404s at w3id.org, at the authors' GitHub
            # Pages, and at raw.githubusercontent. It is unresolvable for everyone, not just offline.
            # Stubbing is honest here and was verified, not assumed: loading CEON with the imports
            # stubbed yields the identical class/property/label set as vendoring the two that DO
            # resolve (214 classes / 317 properties; qudt:QuantityKind still carries its label
            # "Quantity Kind"). CEON.rdf is a materialised Protege merge and the imports are vestigial
            # — LogMap's OWLAPI matcher already produced this alignment without qudt resolving.
            for _iri in stub_import_iris:
                _stub = build_world.get_ontology(_iri)
                _stub.loaded = True
            try:
                build_world.get_ontology(str(onto_filepath)).load()
            except owlready2.OwlReadyOntologyParsingError as parse_error:
                # owlready2 reads RDF/XML, OWL/XML and N-Triples only. Releases in OWL functional or
                # Manchester syntax (e.g. SNOMED CT from the snomed-owl-toolkit) fail here although the
                # OWL API-based matcher loaded them fine, so convert once with ROBOT (OWL/XML) and load that copy.
                converted = _convert_for_owlready2(onto_filepath, cache_dir, parse_error)
                build_world.close()
                build_world = owlready2.World(filename=temp_build_path)
                for _iri in stub_import_iris:
                    _stub = build_world.get_ontology(_iri)
                    _stub.loaded = True
                build_world.get_ontology(str(converted)).load()
            build_world.save()
            build_world.close()
            os.replace(temp_build_path, str(cache_path))

            elapsed = time.time() - build_start
            size_mb = cache_path.stat().st_size / (1024 * 1024)
            success(f'Cache built for {onto_display} in {elapsed:.1f}s ({size_mb:.0f} MB)')
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(temp_build_path)
            raise

        return cache_path
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def _copy_to_private_temp(cache_path: Path) -> tuple[str, str]:
    temp_dir = tempfile.mkdtemp(prefix=_TEMP_DIR_PREFIX)
    _register_temp_cleanup(temp_dir)
    private_path = os.path.join(temp_dir, cache_path.name)
    try:
        shutil.copy2(str(cache_path), private_path)
    except Exception:
        _cleanup_temp_dir(temp_dir)
        raise
    info(f'Working copy: {private_path}', important=True)
    return private_path, temp_dir


class _PrivateCachedWorld(owlready2.World):
    """World whose process-private quadstore is released when it closes."""

    def __init__(self, filename: str, temp_dir: str):
        self._private_temp_dir = temp_dir
        self._private_cache_closed = False
        super().__init__(filename=filename)

    def close(self):
        if not self._private_cache_closed:
            try:
                super().close()
            finally:
                self._private_cache_closed = True
        _cleanup_temp_dir(self._private_temp_dir)


def _get_cached_world(onto_filepath: str, cache_dir: str,
                      stub_import_iris: tuple[str, ...] = ()) -> owlready2.World:
    canonical = _build_or_wait_for_cache(onto_filepath, cache_dir, stub_import_iris)
    private_path, temp_dir = _copy_to_private_temp(canonical)
    try:
        return _PrivateCachedWorld(filename=private_path, temp_dir=temp_dir)
    except Exception:
        _cleanup_temp_dir(temp_dir)
        raise


class OntologyCache:
    """High-level cache interface injected into OntologyAccess."""

    def __init__(self, cache_dir: str | Path | None = None,
                 stub_import_iris: tuple[str, ...] | list[str] | None = None):
        self.cache_dir = str(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        # Empty by default: every existing track's cache key and quadstore are unchanged.
        self.stub_import_iris = tuple(stub_import_iris or ())

    def get_cached_world(self, urionto: str):
        """Return an owlready2.World backed by a cached quadstore."""
        return _get_cached_world(urionto, self.cache_dir, self.stub_import_iris)
