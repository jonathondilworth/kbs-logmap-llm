"""The prompt builder's ontology cache falls back to a ROBOT OWL/XML conversion when owlready2 cannot parse the
input (OWL functional syntax, e.g. SNOMED CT releases). Skipped when ROBOT is not installed."""
from __future__ import annotations

import shutil
from pathlib import Path

import owlready2
import pytest

from logmap_llm.ontology import cache as cache_mod

FUNCTIONAL = """Prefix(:=<http://example.org/fs#>)
Prefix(owl:=<http://www.w3.org/2002/07/owl#>)
Prefix(rdfs:=<http://www.w3.org/2000/01/rdf-schema#>)
Ontology(<http://example.org/fs>
Declaration(Class(:A))
Declaration(Class(:B))
Declaration(ObjectProperty(<http://example.org/fs#116680003>))
SubClassOf(:B :A)
AnnotationAssertion(rdfs:label :A "Alpha"@en)
)
"""


def test_functional_syntax_falls_back_to_owlxml(tmp_path: Path) -> None:
    if cache_mod._robot_command() is None:
        pytest.skip("ROBOT not available")
    onto = tmp_path / "fs.owl"
    onto.write_text(FUNCTIONAL, encoding="utf-8")
    with pytest.raises(owlready2.OwlReadyOntologyParsingError):
        owlready2.World(filename=str(tmp_path / "direct.sqlite3")).get_ontology(str(onto)).load()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_mod._build_or_wait_for_cache(str(onto), str(cache_dir))
    assert cache_path.exists() and cache_path.stat().st_size > 0
    converted = list(cache_dir.glob("*.owx"))
    assert len(converted) == 1, "one OWL/XML conversion kept next to the cache"
    world = owlready2.World(filename=str(cache_path))
    loaded = world.get_ontology("http://example.org/fs").load()
    classes = {c.iri for c in loaded.classes()}
    assert {"http://example.org/fs#A", "http://example.org/fs#B"} <= classes
    # second build reuses the conversion (no ROBOT call needed): remove robot from the environment to prove it
    shutil.rmtree(cache_dir / "unused", ignore_errors=True)
    cache_path.unlink()
    cache_path2 = cache_mod._build_or_wait_for_cache(str(onto), str(cache_dir))
    assert cache_path2.exists() and len(list(cache_dir.glob("*.owx"))) == 1
