"""An undeclared ABox predicate must still get a property prompt.

The `kg-abox` build indexes predicates that are used in the ABox but never declared as owl
properties; `resolve_entity` hands the property template an `InstanceEntity` with no `.prop`,
`get_domain_names()` or `get_range_names()`. Such a predicate has, by definition, no declared
domain or range, so the template must degrade to a name+synonyms prompt rather than drop the
candidate.
"""
import pytest

from logmap_llm.oracle.prompts.context import PromptContext
from logmap_llm.oracle.prompts.templates import oupt_prop_domain_range_synonyms

#: Templates take an explicit prompt context; these tests render the default one.
CTX = PromptContext()


class _UndeclaredAboxPredicate:
    """Shaped like ontology.object.InstanceEntity: names + synonyms, and nothing else.

    Deliberately defines no .prop, get_domain_names, get_range_names, get_domain_synonyms or
    get_range_synonyms — that absence is the case under test.
    """

    def __init__(self, uri: str, name: str, synonyms=()):
        self.uri = uri
        self._name = name
        self._synonyms = set(synonyms)

    def get_preferred_names(self):
        return {self._name}

    def get_synonyms(self):
        return set(self._synonyms)


class _DeclaredProperty:
    def __init__(self, uri, name, domains=(), ranges=()):
        self.uri = uri
        self._name = name
        self._domains = set(domains)
        self._ranges = set(ranges)
        self.is_data_property = False

    def get_preferred_names(self):
        return {self._name}

    def get_synonyms(self):
        return set()

    def get_domain_names(self):
        return set(self._domains)

    def get_range_names(self):
        return set(self._ranges)

    def get_domain_synonyms(self):
        return set()

    def get_range_synonyms(self):
        return set()


def test_undeclared_abox_predicate_still_gets_a_prompt():
    """This pair must build a prompt, not raise AttributeError and be silently skipped."""
    src = _UndeclaredAboxPredicate("http://dbkwik.../property/spouse", "spouse", {"married to"})
    tgt = _UndeclaredAboxPredicate("http://dbkwik.../property/spouse_", "spouse")

    prompt = oupt_prop_domain_range_synonyms(src, tgt, ctx=CTX)

    assert isinstance(prompt, str) and prompt.strip(), "no prompt was built — the oracle is never asked"
    assert "spouse" in prompt
    assert "married to" in prompt, "synonyms must survive the degradation"


def test_undeclared_predicate_omits_the_domain_range_clause_rather_than_faking_one():
    """Degrade to name+synonyms. It must not invent a domain/range it does not have."""
    src = _UndeclaredAboxPredicate("http://x/p/a", "starship")
    tgt = _UndeclaredAboxPredicate("http://y/p/b", "vessel")

    prompt = oupt_prop_domain_range_synonyms(src, tgt, ctx=CTX)

    lowered = prompt.lower()
    assert "domain" not in lowered and "range" not in lowered, (
        "an undeclared predicate has no declared domain/range; the clause must be omitted, not fabricated"
    )


def test_declared_property_still_gets_its_domain_range_clause():
    """Properties that do declare a domain/range keep their clause."""
    src = _DeclaredProperty("http://x#author", "author", domains={"Paper"}, ranges={"Person"})
    tgt = _DeclaredProperty("http://y#writtenBy", "writtenBy", domains={"Publication"}, ranges={"Author"})

    prompt = oupt_prop_domain_range_synonyms(src, tgt, ctx=CTX)

    assert "Paper" in prompt and "Person" in prompt
    assert "Publication" in prompt and "Author" in prompt


def test_entity_without_preferred_names_falls_back_to_uri_not_attributeerror():
    """`entity.prop.name` raises on an InstanceEntity; the fallback must derive a name from the URI."""

    class _Nameless:
        uri = "http://dbkwik.webdatacommons.org/starwars/property/homeworld"

        def get_preferred_names(self):
            return set()

        def get_synonyms(self):
            return set()

    prompt = oupt_prop_domain_range_synonyms(_Nameless(), _Nameless(), ctx=CTX)
    assert "homeworld" in prompt


def test_the_old_failure_mode_is_gone():
    """Pin the exact failure mode: no AttributeError from the property template."""
    src = _UndeclaredAboxPredicate("http://x/p/a", "spouse(s)_")
    tgt = _UndeclaredAboxPredicate("http://y/p/b", "spouse")
    try:
        oupt_prop_domain_range_synonyms(src, tgt, ctx=CTX)
    except AttributeError as e:  # pragma: no cover
        pytest.fail(f"regression: property template still raises on an undeclared ABox predicate: {e}")
