"""Guards for the annotation-predicate resource.

The IRIs that drive label/synonym extraction live in
``logmap_llm/ontology/data/annotation_uris.json``: each string must match the RDF
byte-for-byte, so it is data, not authored code. If the build config does not ship
``ontology/data/*.json`` as package data, an installed copy loads with no annotation
predicates and every prompt degrades to bare URIs; these tests make that failure
loud and immediate.
"""
from __future__ import annotations

import json

import pytest

from logmap_llm.ontology.access import (
    ANNOTATION_URIS_PATH,
    AnnotationURIs,
    load_annotation_uri_groups,
)

REQUIRED_GROUPS = ("preferred_label", "synonym", "lexical_extra")


def test_resource_is_present():
    """The JSON resource ships with the package."""
    assert ANNOTATION_URIS_PATH.is_file(), (
        f"{ANNOTATION_URIS_PATH} is missing. If this is an installed copy, the build "
        "config does not declare ontology/data/*.json as package data."
    )


def test_resource_is_valid_json_with_all_groups():
    raw = json.loads(ANNOTATION_URIS_PATH.read_text(encoding="utf-8"))
    for group in REQUIRED_GROUPS:
        assert group in raw, f"missing group: {group}"
        assert isinstance(raw[group], list) and raw[group], f"group {group} must be non-empty"
        assert all(isinstance(v, str) and v.strip() for v in raw[group])


def test_groups_load_as_non_empty_sets():
    groups = load_annotation_uri_groups()
    assert set(groups) == set(REQUIRED_GROUPS)
    for group in REQUIRED_GROUPS:
        assert groups[group], f"group {group} loaded empty"


def test_missing_resource_fails_loudly(tmp_path):
    """A missing resource raises rather than silently yielding zero predicates."""
    with pytest.raises(FileNotFoundError, match="annotation predicate resource missing"):
        load_annotation_uri_groups(tmp_path / "does_not_exist.json")


def test_malformed_resource_fails_loudly(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_annotation_uri_groups(bad)


def test_wrong_shape_fails_loudly(tmp_path):
    bad = tmp_path / "wrong.json"
    bad.write_text(json.dumps({"preferred_label": "not-a-list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="must be a list of IRI strings"):
        load_annotation_uri_groups(bad)


def test_annotation_uris_builds_expected_sets():
    """The three public sets are populated and the lexical set is a true superset."""
    uris = AnnotationURIs()
    assert uris.mainLabelURIs
    assert uris.synonymLabelURIs
    assert uris.mainLabelURIs <= uris.lexicalAnnotationURIs
    assert uris.synonymLabelURIs <= uris.lexicalAnnotationURIs
    # the lexical set carries entries beyond the two label groups
    assert uris.lexicalAnnotationURIs - (uris.mainLabelURIs | uris.synonymLabelURIs)


def test_accessors_return_the_backing_sets():
    uris = AnnotationURIs()
    assert uris.get_annotation_uris_for_preferred_labels() == uris.mainLabelURIs
    assert uris.get_annotation_uris_for_synonyms() == uris.synonymLabelURIs


def test_every_predicate_is_an_absolute_iri():
    groups = load_annotation_uri_groups()
    for group, values in groups.items():
        for value in values:
            assert value.startswith(("http://", "https://")), f"{group}: {value!r}"


# --------------------------------------------------------------------------
# Semantic classification of predicates. A predicate lives in the group matching
# its semantics: a definition is a gloss (lexical context), never an alternative
# name, and a preferred-label predicate is a preferred label whichever vocabulary
# declares it. Misclassification injects wrong content into the prompts' synonym
# clause.
# --------------------------------------------------------------------------

_HAS_DEFINITION = "http://www.geneontology.org/formats/oboInOwl#hasDefinition"
_BIRNLEX_PREF = "http://bioontology.org/projects/ontologies/birnlex#preferred_label"


def test_definitions_are_lexical_context_not_synonyms():
    groups = load_annotation_uri_groups()
    assert _HAS_DEFINITION in groups["lexical_extra"]
    assert _HAS_DEFINITION not in groups["synonym"]
    assert _HAS_DEFINITION not in groups["preferred_label"]


def test_birnlex_preferred_label_is_a_preferred_label():
    groups = load_annotation_uri_groups()
    assert _BIRNLEX_PREF in groups["preferred_label"]
    assert _BIRNLEX_PREF not in groups["synonym"]


def test_misclassified_predicates_stay_in_the_lexical_superset():
    """Reclassification must not remove either predicate from the lexical set."""
    uris = AnnotationURIs()
    assert _HAS_DEFINITION in uris.lexicalAnnotationURIs
    assert _BIRNLEX_PREF in uris.lexicalAnnotationURIs
