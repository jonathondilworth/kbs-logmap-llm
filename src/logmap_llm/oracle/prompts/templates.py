"""
logmap_llm.oracle.prompts.templates
Prompt template functions and registry, plus the build_oracle_user_prompts()
and build_oracle_user_prompts_bidirectional() orchestration functions.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Callable

from logmap_llm.ontology.object import OntologyEntryAttr, ClassNotFoundError, resolve_entity, resolve_entity_as
from logmap_llm.oracle.prompts.routing import mask_row_entity_type, resolve_pair_lane
from logmap_llm.oracle.prompts.context import PromptContext
from logmap_llm.ontology.sibling_retrieval import SiblingSelector, _get_label
from logmap_llm.utils.logging import info, debug, warn, warning
from logmap_llm.oracle.prompts.formatting import (
    format_hierarchy,
    get_deterministic_single_name,
    select_best_direct_entity_names,
    select_best_direct_entity_names_with_synonyms,
    select_best_sequential_hierarchy_with_synonyms,
    format_sibling_context,
    format_synonyms_parenthetical,
    format_domain_range_clause,
    format_restriction_context,
    format_relational_signature,
    format_property_characteristics,
    format_instance_type_clause,
    format_instance_attribute_clause,
    select_intersecting_properties,
    get_deterministic_single_name,
)
from logmap_llm.constants import (
    EntityType,
    PAIRS_SEPARATOR,
    RESPONSE_INSTRUCTION,
    DEFAULT_ANSWER_FORMAT,
    DEFAULT_RESPONSE_MODE,
    ANSWER_FORMATS,
    RESPONSE_MODES,
    VERBOSE,
)
from tqdm import tqdm

###
# TEMPLATE REGISTRY
###

@dataclass(frozen=True)
class TemplateSpec:
    fn: Callable
    entity_type: EntityType = EntityType.CLASS
    bidirectional: bool = False
    requires_siblings: bool = False


class TemplateRegistry:
    '''
    Container in which template functions (prompts) are bound and resolved; use the
    decorator (or register_fn) to register. Registration options:

        1. entity_type: which entity category the prompt handles
           (CLASS | OBJECTPROPERTY | DATAPROPERTY | INSTANCE; defaults to CLASS).
           TODO: distinct behaviours for OBJECTPROPERTY vs DATAPROPERTY.

        2. bidirectional=True: evaluate under 'equivalence by mutual subsumption'
           (A \sqsubseteq B \land B \sqsubseteq A \iff A \equiv B). This doubles the
           prompt count — each prompt is sent twice with src and tgt reversed.
           (defaults to False)

        3. requires_siblings=True: siblings are fetched and the top-k selected, either
           alphanumerically by label/preferred term or by embedding cosine similarity
           (CLS-pooled or Sentence-Transformers model, configurable). (defaults to False)
           TODO: sibling retrieval via LM-based ontology embeddings (not yet implemented).
    '''
    def __init__(self):
        self._templates: dict[str, TemplateSpec] = {}

    # decorator (helper):
    def register(self, name: str, entity_type: EntityType = EntityType.CLASS, bidirectional: bool = False, requires_siblings: bool = False) -> Callable:
        def decorator(fn: Callable) -> Callable:
            self._templates[name] = TemplateSpec(
                fn=fn,
                entity_type=entity_type,
                bidirectional=bidirectional,
                requires_siblings=requires_siblings
            )
            return fn
        return decorator

    # bind:
    def register_fn(self, name: str, fn: Callable, entity_type: EntityType = EntityType.CLASS, bidirectional: bool = False, requires_siblings: bool = False) -> None:
        self._templates[name] = TemplateSpec(
            fn=fn, 
            entity_type=entity_type,
            bidirectional=bidirectional,
            requires_siblings=requires_siblings
        )

    # resolve:
    def get(self, name: str) -> TemplateSpec:
        if name not in self._templates:
            raise KeyError(f"Template '{name}' not found. Available: {list(self._templates.keys())}")
        return self._templates[name]

    ###
    # callable helpers:
    ###

    def is_bidirectional(self, name: str) -> bool:
        return self.get(name).bidirectional

    def requires_siblings(self, name: str) -> bool:
        return self.get(name).requires_siblings

    def get_by_entity_type(self, et: EntityType) -> dict[str, TemplateSpec]:
        return {k: v for k, v in self._templates.items() if v.entity_type == et}

    def __contains__(self, name: str) -> bool:
        return name in self._templates

    def __len__(self) -> int:
        return len(self._templates)

    def keys(self) -> list[str]:
        return list(self._templates.keys())


registry = TemplateRegistry()


###
# HELPERS (FUNCTIONS)
###

def _retrieve_siblings(entity, sibling_selector: SiblingSelector=None, max_count=2) -> list[str] | list[tuple[str, float]]:
    """Return (preferred label, score) tuples for up to max_count siblings; the
    no-selector fallthrough scores each sibling 1.0."""
    if sibling_selector is not None:
        ranked = sibling_selector.select_siblings(entity, max_count=max_count)
        return [
            (
                _get_label(sibling), 
                score
            ) for sibling, score in ranked
        ]

    sib_entries = entity.get_siblings(max_count=max_count)
    return [
        (
            min(sibling.get_preferred_names()) if sibling.get_preferred_names() 
            else str(sibling.thing_class.name), 1.0
        )
        for sibling in sib_entries
    ]


def _get_merged_entropies(src_inst, tgt_inst) -> dict:
    src_ent = src_inst.onto.compute_predicate_entropies()
    tgt_ent = tgt_inst.onto.compute_predicate_entropies()
    merged = {}
    merged.update(src_ent)
    for uri, entropy in tgt_ent.items():
        if uri not in merged or entropy > merged[uri]:
            merged[uri] = entropy
    return merged


###
# LEGACY TEMPLATES
# NOTE: these are not actively used and are currently untested.
# Name-set interpolations are wrapped in sorted() so the prompt text does not vary
# with PYTHONHASHSEED across processes.
###

@registry.register("all_data_dummy")
def prompt_all_data_dummy(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    return f"""
    **Task Description:**
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. **Source Entity:**
    **All Entity names:** {sorted(src_entity.get_preferred_names())}
    **Parent Entity Namings:** {src_entity.get_parents_preferred_names()}
    **Child Entity Namings:** {src_entity.get_children_preferred_names()}

    2. **Target Entity:**
    **All Entity names:** {sorted(tgt_entity.get_preferred_names())}
    **Parent Entity Namings:** {tgt_entity.get_parents_preferred_names()}
    **Child Entity Namings:** {tgt_entity.get_children_preferred_names()}

    Write "Yes" if the entities refer to the same concepts, and "No" otherwise.
    """.strip()


@registry.register("only_names")
def prompt_only_names(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    return f"""
    Given two entities from different ontologies with their names, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {sorted(src_entity.get_all_entity_names())}

    2. Target Entity:
    All Entity names: {sorted(tgt_entity.get_all_entity_names())}

    Response with True or False
    """.strip()


@registry.register("with_hierarchy")
def prompt_with_hierarchy(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    return f"""
    Given two entities from different ontologies with their names, parent relationships, and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {sorted(src_entity.get_all_entity_names())}
    Parent Entity Namings: {src_entity.get_parents_preferred_names()}
    Child Entity Namings: {src_entity.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {sorted(tgt_entity.get_all_entity_names())}
    Parent Entity Namings: {tgt_entity.get_parents_preferred_names()}
    Child Entity Namings: {tgt_entity.get_children_preferred_names()}

    Response with True or False
    """.strip()


@registry.register("only_with_parents")
def prompt_only_with_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    return f"""
    Given two entities from different ontologies with their names and parent relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {sorted(src_entity.get_all_entity_names())}
    Parent Entity Namings: {src_entity.get_parents_preferred_names()}

    2. Target Entity:
    All Entity names: {sorted(tgt_entity.get_all_entity_names())}
    Parent Entity Namings: {tgt_entity.get_parents_preferred_names()}

    Response with True or False
    """.strip()


@registry.register("only_with_children")
def prompt_only_with_children(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    return f"""
    Given two entities from different ontologies with their names and child relationships, determine if these concepts are the same:

    1. Source Entity:
    All Entity names: {sorted(src_entity.get_all_entity_names())}
    Child Entity Namings: {src_entity.get_children_preferred_names()}

    2. Target Entity:
    All Entity names: {sorted(tgt_entity.get_all_entity_names())}
    Child Entity Namings: {tgt_entity.get_children_preferred_names()}

    Response with True or False
    """.strip()



###
# ORIGINAL CLASS EQUIVALENCE TEMPLATES
###



@registry.register("one_level_of_parents_structured")
def oupt_one_level_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    """Ontological prompt that uses ontology-focused language."""
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct{ctx.forced_domain_string}ontology.",
        "Your task is to assess whether they represent the **same ontological concept**, considering both their semantic meaning and hierarchical position.",
        f'\n1. Source entity: "{src_entity_names}"',
        f"\t- Direct ontological parent: {src_parent}",
        f'\n2. Target entity: "{tgt_entity_names}"',
        f"\t- Direct ontological parent: {tgt_parent}",
        f'\nAre these entities **ontologically equivalent** within their respective ontologies? {ctx.response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents_structured")
def oupt_two_levels_of_parents_structured(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    """Ontological prompt that uses ontology-focused language and takes hierarchical relationships into account."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2))
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2))
    
    prompt_lines = [
        f"Analyze the following entities, each originating from a distinct{ctx.forced_domain_string}ontology.",
        "Each is represented by its **ontological lineage**, capturing its hierarchical placement from the most general to the most specific level.",
        f"\n1. Source entity ontological lineage:\n{src_hierarchy}",
        f"\n2. Target entity ontological lineage:\n{tgt_hierarchy}",
        f'\nBased on their **ontological positioning, hierarchical relationships, and semantic alignment**, do these entities represent the **same ontological concept**? {ctx.response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("one_level_of_parents")
def oupt_one_level_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names) = select_best_direct_entity_names(src_entity, tgt_entity)
    
    prompt_lines = [
        ctx.domain_preamble,
        (f'The first one is "{src_entity_names}"' + (f', which belongs to the broader category "{src_parent}"' if src_parent else "")),
        (f'The second one is "{tgt_entity_names}"' + (f', which belongs to the broader category "{tgt_parent}"' if tgt_parent else "")),
        (f'\nDo they mean the same thing? {ctx.response_instruction}'),
    ]
    return "\n".join(prompt_lines)



@registry.register("two_levels_of_parents")
def oupt_two_levels_of_parents(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)
    
    prompt_lines = [
        ctx.domain_preamble,
        (f'The first one is "{src_hierarchy[0]}"' + (f', which belongs to the broader category "{src_hierarchy[1]}"' if len(src_hierarchy) > 1 else "") + (f', under the even broader category "{src_hierarchy[2]}"' if len(src_hierarchy) > 2 else "")),
        (f'The second one is "{tgt_hierarchy[0]}"' + (f', which belongs to the broader category "{tgt_hierarchy[1]}"' if len(tgt_hierarchy) > 1 else "") + (f', under the even broader category "{tgt_hierarchy[2]}"' if len(tgt_hierarchy) > 2 else "")),
        (f'\nDo they mean the same thing? {ctx.response_instruction}'),
    ]
    return "\n".join(prompt_lines)



@registry.register("one_level_of_parents_and_synonyms")
def oupt_one_level_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    """Natural language prompt that includes synonyms for a more intuitive comparison."""
    (src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms) = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(src_synonyms))) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in sorted(tgt_synonyms))) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    prompt_lines = [
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nDo they mean the same thing? {ctx.response_instruction}',
    ]
    return "\n".join(prompt_lines)



@registry.register("eacl_direct_entity_with_synonyms")
def oupt_eacl_direct_entity_with_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    """EACL 2026 P^NLF_S prompt, byte-faithful to rai-ukraine-kga-llm/src/prompts/prompts.py::prompt_direct_entity_with_synonyms:
    fixed "biomedical" preamble, a "Thing" parent is not suppressed, a missing parent renders as "None", and the closing
    instruction is always 'Respond with "True" or "False".' whatever the configured response mode (the EACL runs paired
    this text with a structured {answer: bool} response_format). Synonym order is sorted here (the EACL code iterated a set)."""
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity, add_thing=True)
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    prompt_lines = [
        "We have two entities from different biomedical ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}, which falls under the category "{src_parent}".',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}, which falls under the category "{tgt_parent}".',
        '\nDo they mean the same thing? Respond with "True" or "False".',
    ]
    return "\n".join(prompt_lines)

@registry.register("two_levels_of_parents_and_synonyms")
def oupt_two_levels_of_parents_and_synonyms(src_entity: OntologyEntryAttr, tgt_entity: OntologyEntryAttr, *, ctx: PromptContext) -> str:
    """Natural language prompt asking whether two ontology entities (with synonyms and hierarchy) represent the same concept (True/False)."""
    src_hierarchy = format_hierarchy(src_entity.get_parents_by_levels(max_level=2), True)
    tgt_hierarchy = format_hierarchy(tgt_entity.get_parents_by_levels(max_level=2), True)
    
    (src_syns, tgt_syns, src_parents_syns, tgt_parents_syns) = select_best_sequential_hierarchy_with_synonyms(src_entity, tgt_entity, max_level=2)

    def describe_entity(hierarchy, entity_syns, parent_syns):
        name_part = f'"{hierarchy[0]}"'
        if entity_syns:
            alt = ", ".join(f'"{s}"' for s in sorted(entity_syns))
            name_part += f", also known as {alt}"
        parts = [name_part]
        labels = ["belongs to broader category", "under the even broader category", "under the even broader category"]
        for i, parent_name in enumerate(hierarchy[1:]):
            text = f'{labels[i]} "{parent_name}"'
            if parent_syns[i]:
                alt = ", ".join(f'"{s}"' for s in sorted(parent_syns[i]))
                text += f" (also known as {alt})"
            parts.append(text)
        return ", ".join(parts)

    src_desc = describe_entity(src_hierarchy, src_syns, src_parents_syns)
    tgt_desc = describe_entity(tgt_hierarchy, tgt_syns, tgt_parents_syns)
    
    prompt_lines = [
        ctx.domain_preamble,
        f"The first one is {src_desc}.",
        f"The second one is {tgt_desc}.",
        f'\nDo they mean the same thing? {ctx.response_instruction}',
    ]
    return "\n".join(prompt_lines)



###
# NEW CLASS-BASED PROMPTS
###



@registry.register("synonyms_only")
def oupt_synonyms_only(src_entity, tgt_entity, *, ctx: PromptContext):
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        "We have two entities from different ontologies.",
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nDo they mean the same thing? {ctx.response_instruction}',
    ])



###
# SUBSUMPTION TEMPLATES
# (ie. bidirectional=True)
###

### TODO: rename; its not only labels, it also contains synonyms.
@registry.register("sub_labels_only", bidirectional=True)
def oupt_sub_labels_only(src_entity, tgt_entity, *, ctx: PromptContext):
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? ',
        f'\n{ctx.response_instruction}',
    ])



@registry.register("sub_parents_synonyms", bidirectional=True)
def oupt_sub_parents_synonyms(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"? ',
        f'\n{ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("sub_syns_conj_parent", bidirectional=True)
def oupt_sub_syns_conj_parent(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        f'\nIf something is a "{src_entity_names}", is it also a "{tgt_entity_names}"{tgt_qual}? ',
        f'\n{ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("single_subs")
def oupt_single_subs(src_entity, tgt_entity, *, ctx: PromptContext):
    _, _, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("single_subs_with_parents")
def oupt_single_subs_with_parents(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}".',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}".',
        f'\nAre both statements true? {ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("single_subs_with_conj_parent")
def oupt_single_subs_with_conj_parent(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.',
        '\nFor these to be the same concept, BOTH of the following must be true:',
        f'1. Every "{src_entity_names}" is also a "{tgt_entity_names}"{tgt_qual}.',
        f'2. Every "{tgt_entity_names}" is also a "{src_entity_names}"{src_qual}.',
        f'\nAre both statements true? {ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("equiv_test_for_ancestral_disjointness", bidirectional=True)
def oupt_equiv_test_for_ancestral_disjointness(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nCan something be a "{src_entity_names}"{src_qual} at the same time as being a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{ctx.response_instruction}',
    ])



@registry.register("equiv_with_ancestral_disjointness", bidirectional=True)
def oupt_equiv_with_ancestral_disjointness(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_qual = f' and a "{src_parent}"' if src_parent else ""
    tgt_qual = f' and a "{tgt_parent}"' if tgt_parent else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}.',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}.',
        f'\nIf something is a "{src_entity_names}"{src_qual}, is it also a "{tgt_entity_names}"{tgt_qual}?',
        f'\n{ctx.response_instruction}',
    ])



@registry.register("deductive_equiv")
def oupt_deductive_equiv(src_entity, tgt_entity, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    facts = []
    facts.append(f'- "{src_entity_names}" is a kind of "{src_parent}" (Ontology 1)' if src_parent else f'- "{src_entity_names}" is a concept in Ontology 1')
    
    if src_synonyms:
        facts.append(f'- "{src_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in src_synonyms) + ' (Ontology 1)')
    
    facts.append(f'- "{tgt_entity_names}" is a kind of "{tgt_parent}" (Ontology 2)' if tgt_parent else f'- "{tgt_entity_names}" is a concept in Ontology 2')
    
    if tgt_synonyms:
        facts.append(f'- "{tgt_entity_names}" is also known as ' + " and ".join(f'"{s}"' for s in tgt_synonyms) + ' (Ontology 2)')
    
    return "\n".join([
        ctx.domain_preamble,
        "Consider the following facts:",
        "\n".join(facts),
        f'\nGiven these facts, can you conclude that "{src_entity_names}" (Ontology 1) and "{tgt_entity_names}" (Ontology 2) refer to the same real-world concept? {ctx.response_instruction}',
    ])



### TODO: scheduled for removal
@registry.register("falsification_equiv", requires_siblings=True)
def oupt_falsification_equiv(src_entity, tgt_entity, sibling_selector=None, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.{tgt_sibling_clause}',
        f'\nFirst consider: could something classified as a "{src_parent}" in one ontology plausibly also be classified as a "{tgt_parent}" in another? If the categories suggest entirely different domains, the concepts are unlikely to match. If the categories are compatible or overlapping, the concepts may well match.',
        f'\nDo "{src_entity_names}" (a "{src_parent}") and "{tgt_entity_names}" (a "{tgt_parent}") refer to the same concept? {ctx.response_instruction}',
    ])



###
# SIBLING-AWARE PROMPT TEMPLATES
###



@registry.register("equiv_parents_siblings", requires_siblings=True)
def oupt_equiv_parents_siblings(src_entity, tgt_entity, sibling_selector=None, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, _, _ = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""
    
    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""

    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_parent_clause}.{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {ctx.response_instruction}',
    ])



@registry.register("equiv_parents_synonyms_siblings", requires_siblings=True)
def oupt_equiv_parents_synonyms_siblings(src_entity, tgt_entity, sibling_selector=None, *, ctx: PromptContext):
    src_parent, tgt_parent, src_entity_names, tgt_entity_names, src_synonyms, tgt_synonyms = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)
    
    src_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in src_synonyms)) if src_synonyms else ""
    tgt_synonyms_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_synonyms)) if tgt_synonyms else ""
    
    src_siblings = _retrieve_siblings(src_entity, sibling_selector)
    tgt_siblings = _retrieve_siblings(tgt_entity, sibling_selector)
    
    src_sibling_clause = (" " + format_sibling_context(src_siblings, src_parent)) if src_siblings else ""
    tgt_sibling_clause = (" " + format_sibling_context(tgt_siblings, tgt_parent)) if tgt_siblings else ""

    src_parent_clause = f', which falls under the category "{src_parent}"' if src_parent and src_parent != "Thing" else ""
    tgt_parent_clause = f', which falls under the category "{tgt_parent}"' if tgt_parent and tgt_parent != "Thing" else ""
    
    return "\n".join([
        ctx.domain_preamble,
        f'The first one is "{src_entity_names}"{src_synonyms_text}{src_parent_clause}.{src_sibling_clause}',
        f'The second one is "{tgt_entity_names}"{tgt_synonyms_text}{tgt_parent_clause}.{tgt_sibling_clause}',
        f'\nDo they mean the same thing? {ctx.response_instruction}',
    ])



###
# PROPERTY TEMPLATES (OBJECTPROPERTY) -- EntityType.OBJECTPROPERTY
###



@registry.register("prop_labels_only", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_labels_only(src_prop, tgt_prop, *, ctx: PromptContext):
    src_name = get_deterministic_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_deterministic_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first property is "{src_name}".',
        f'The second property is "{tgt_name}".',
        f'Do these properties represent the same relationship? {ctx.response_instruction}',
    ]
    
    return "\n".join(prompt_lines)



@registry.register("prop_domain_range", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_domain_range(src_prop, tgt_prop, *, ctx: PromptContext):
    src_name = get_deterministic_single_name(src_prop.get_preferred_names()) or _entity_fallback_name(src_prop)
    tgt_name = get_deterministic_single_name(tgt_prop.get_preferred_names()) or _entity_fallback_name(tgt_prop)

    src_desc = format_domain_range_clause(src_name, _undeclared_safe(src_prop, "get_domain_names"),
                                          _undeclared_safe(src_prop, "get_range_names"),
                                          is_data_property=getattr(src_prop, "is_data_property", False))
    tgt_desc = format_domain_range_clause(tgt_name, _undeclared_safe(tgt_prop, "get_domain_names"),
                                          _undeclared_safe(tgt_prop, "get_range_names"),
                                          is_data_property=getattr(tgt_prop, "is_data_property", False))

    prompt_lines = [
        ctx.domain_preamble,
        f'The first property is {src_desc}.',
        f'The second property is {tgt_desc}.',
        f'Do these properties represent the same relationship? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)



def _undeclared_safe(entity, meth: str) -> set:
    """Domain/range accessor that tolerates an entity which is not a *declared* property.

    Predicates used in the ABox but never declared as OWL properties resolve to an
    `InstanceEntity`, which has no `get_domain_names`/`get_range_names`. An undeclared
    predicate has no declared domain or range, so return an empty set — the prompt falls
    back to name + synonyms rather than the candidate being silently dropped.
    """
    fn = getattr(entity, meth, None)
    if not callable(fn):
        return set()
    try:
        return fn() or set()
    except Exception:
        return set()


def _entity_fallback_name(entity) -> str:
    """Last-resort display name. `InstanceEntity` has no `.prop`, so `entity.prop.name` raises."""
    for attr in ("prop", "cls", "inst"):
        obj = getattr(entity, attr, None)
        name = getattr(obj, "name", None)
        if name:
            return str(name)
    uri = str(getattr(entity, "uri", "") or "")
    return uri.rsplit("/", 1)[-1].rsplit("#", 1)[-1] or uri


@registry.register("prop_domain_range_synonyms", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_domain_range_synonyms(src_prop, tgt_prop, *, ctx: PromptContext):
    src_name = get_deterministic_single_name(src_prop.get_preferred_names()) or _entity_fallback_name(src_prop)
    tgt_name = get_deterministic_single_name(tgt_prop.get_preferred_names()) or _entity_fallback_name(tgt_prop)
    
    src_syn_text = format_synonyms_parenthetical(src_prop.get_synonyms(), src_name)
    tgt_syn_text = format_synonyms_parenthetical(tgt_prop.get_synonyms(), tgt_name)
    
    src_line = f'The first property is "{src_name}"{src_syn_text}'
    tgt_line = f'The second property is "{tgt_name}"{tgt_syn_text}'
    
    src_dom, src_rng = _undeclared_safe(src_prop, "get_domain_names"), _undeclared_safe(src_prop, "get_range_names")
    tgt_dom, tgt_rng = _undeclared_safe(tgt_prop, "get_domain_names"), _undeclared_safe(tgt_prop, "get_range_names")

    if src_dom or src_rng:
        src_dr = format_domain_range_clause(src_name, src_dom, src_rng, domain_synonyms=_undeclared_safe(src_prop, "get_domain_synonyms"), range_synonyms=_undeclared_safe(src_prop, "get_range_synonyms"), include_synonyms=True, is_data_property=getattr(src_prop, "is_data_property", False))
        src_line += src_dr[len(f'"{src_name}"'):]

    if tgt_dom or tgt_rng:
        tgt_dr = format_domain_range_clause(tgt_name, tgt_dom, tgt_rng, domain_synonyms=_undeclared_safe(tgt_prop, "get_domain_synonyms"), range_synonyms=_undeclared_safe(tgt_prop, "get_range_synonyms"), include_synonyms=True, is_data_property=getattr(tgt_prop, "is_data_property", False))
        tgt_line += tgt_dr[len(f'"{tgt_name}"'):]
    
    prompt_lines = [
        ctx.domain_preamble,
        f'{src_line}',
        f'{tgt_line}',
        f'Do these properties represent the same relationship? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)



@registry.register("prop_with_inverse_and_chars", entity_type=EntityType.OBJECTPROPERTY)
def oupt_prop_with_inverse_and_chars(src_prop, tgt_prop, *, ctx: PromptContext):
    src_name = get_deterministic_single_name(src_prop.get_preferred_names()) or str(src_prop.prop.name)
    tgt_name = get_deterministic_single_name(tgt_prop.get_preferred_names()) or str(tgt_prop.prop.name)

    src_desc = format_domain_range_clause(src_name, src_prop.get_domain_names(), src_prop.get_range_names(),
                                          is_data_property=getattr(src_prop, "is_data_property", False))
    tgt_desc = format_domain_range_clause(tgt_name, tgt_prop.get_domain_names(), tgt_prop.get_range_names(),
                                          is_data_property=getattr(tgt_prop, "is_data_property", False))

    src_chars = src_prop.get_characteristics()
    src_inverse = src_prop.get_deterministic_inverse_name()

    tgt_chars = tgt_prop.get_characteristics()
    tgt_inverse = tgt_prop.get_deterministic_inverse_name()

    src_char_text = " " + format_property_characteristics(src_chars, src_inverse) if (src_chars or src_inverse) else ""
    tgt_char_text = " " + format_property_characteristics(tgt_chars, tgt_inverse) if (tgt_chars or tgt_inverse) else ""

    prompt_lines = [
        ctx.domain_preamble,
        f'The first property is {src_desc}.{src_char_text}',
        f'The second property is {tgt_desc}.{tgt_char_text}',
        f'Do these properties represent the same relationship? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)



###
# CONFERENCE-INSPIRED ENRICHED TEMPLATES
###



@registry.register("class_with_restrictions")
def oupt_class_with_restrictions(src_entity, tgt_entity, *, ctx: PromptContext):
    _, _, src_name, tgt_name, src_syns, tgt_syns = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)

    src_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in src_syns)) if src_syns else ""
    tgt_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_syns)) if tgt_syns else ""

    src_restrictions = src_entity.get_restrictions()
    tgt_restrictions = tgt_entity.get_restrictions()

    src_restr_text = " " + format_restriction_context(src_restrictions) if src_restrictions else ""
    tgt_restr_text = " " + format_restriction_context(tgt_restrictions) if tgt_restrictions else ""

    prompt_lines = [
        ctx.domain_preamble,
        f'The first one is "{src_name}"{src_syn_text}.{src_restr_text}',
        f'The second one is "{tgt_name}"{tgt_syn_text}.{tgt_restr_text}',
        f'Do they mean the same thing? {ctx.response_instruction}',
    ]

    return "\n".join(prompt_lines)



@registry.register("class_role_signature")
def oupt_class_role_signature(src_entity, tgt_entity, *, ctx: PromptContext):
    _, _, src_name, tgt_name, src_syns, tgt_syns = select_best_direct_entity_names_with_synonyms(src_entity, tgt_entity)

    src_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in src_syns)) if src_syns else ""
    tgt_syn_text = (", also known as " + ", ".join(f'"{s}"' for s in tgt_syns)) if tgt_syns else ""

    src_sig = src_entity.get_relational_signature()
    tgt_sig = tgt_entity.get_relational_signature()

    src_sig_text = " " + format_relational_signature(src_sig) if (src_sig.get('as_domain') or src_sig.get('as_range')) else ""
    tgt_sig_text = " " + format_relational_signature(tgt_sig) if (tgt_sig.get('as_domain') or tgt_sig.get('as_range')) else ""

    prompt_lines = [
        ctx.domain_preamble,
        f'The first one is "{src_name}"{src_syn_text}.{src_sig_text}',
        f'The second one is "{tgt_name}"{tgt_syn_text}.{tgt_sig_text}',
        f'Do they mean the same thing? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)



###
# INSTANCE TEMPLATES -- EntityType.INSTANCE
###



@registry.register("inst_labels_only", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_only(src_inst, tgt_inst, *, ctx: PromptContext):
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    prompt_lines = [
        ctx.domain_preamble,
        f'The first is "{src_label}".',
        f'The second is "{tgt_label}"',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)



@registry.register("inst_labels_with_types", entity_type=EntityType.INSTANCE)
def oupt_inst_labels_with_types(src_inst, tgt_inst, *, ctx: PromptContext):
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri

    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())

    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "")

    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]

    return "\n".join(prompt_lines)




@registry.register("inst_types_and_attributes", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)




@registry.register("inst_full_context", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_data_sel, tgt_data_sel = select_intersecting_properties(src_inst.get_data_properties(), tgt_inst.get_data_properties(), max_properties=max_properties)
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(src_inst.get_object_properties(), tgt_inst.get_object_properties(), max_properties=max_properties)
    
    def _build(label, type_clause, data_clause, object_clause):
        built_description = (
            f'"{label}"' + (f', {type_clause}' if type_clause else "") 
            + (f' and has attributes: {data_clause}' if data_clause else "") 
            + (f' and has relationships: {object_clause}' if object_clause else "")
        )
        return built_description
    
    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)



@registry.register("inst_types_and_attributes_intersect", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes_intersect(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties, intersection_only=True)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)



@registry.register("inst_full_context_intersect", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context_intersect(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    src_data_sel, tgt_data_sel = select_intersecting_properties(src_inst.get_data_properties(), tgt_inst.get_data_properties(), max_properties=max_properties, intersection_only=True)
    src_obj_sel, tgt_obj_sel = select_intersecting_properties(src_inst.get_object_properties(), tgt_inst.get_object_properties(), max_properties=max_properties, intersection_only=True)
    
    def _build(label, type_clause, data_clause, object_clause):
        built_description = (
            f'"{label}"' + (f', {type_clause}' if type_clause else "") 
            + (f' and has attributes: {data_clause}' if data_clause else "") 
            + (f' and has relationships: {object_clause}' if object_clause else "")
        )
        return built_description

    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)



@registry.register("inst_types_and_attributes_entropy", entity_type=EntityType.INSTANCE)
def oupt_inst_types_and_attributes_entropy(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    entropies = _get_merged_entropies(src_inst, tgt_inst)
    
    src_selected, tgt_selected = select_intersecting_properties(src_inst.get_all_properties(), tgt_inst.get_all_properties(), max_properties=max_properties, intersection_only=True, predicate_entropies=entropies)
    
    src_attr = fmt_fn(src_selected, max_properties)
    tgt_attr = fmt_fn(tgt_selected, max_properties)
    
    src_desc = f'"{src_label}"' + (f', {src_type_clause}' if src_type_clause else "") + (f' and {src_attr}' if src_attr else "")
    tgt_desc = f'"{tgt_label}"' + (f', {tgt_type_clause}' if tgt_type_clause else "") + (f' and {tgt_attr}' if tgt_attr else "")
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)



@registry.register("inst_full_context_entropy", entity_type=EntityType.INSTANCE)
def oupt_inst_full_context_entropy(src_inst, tgt_inst, max_properties=3, fmt_fn=None, *, ctx: PromptContext):
    if fmt_fn is None: 
        fmt_fn = ctx.instance_fmt_fn
    
    src_label = get_deterministic_single_name(src_inst.get_preferred_names()) or src_inst.uri
    tgt_label = get_deterministic_single_name(tgt_inst.get_preferred_names()) or tgt_inst.uri
    
    src_type_clause = format_instance_type_clause(src_inst.get_type_names())
    tgt_type_clause = format_instance_type_clause(tgt_inst.get_type_names())
    
    entropies = _get_merged_entropies(src_inst, tgt_inst)
    
    src_data_sel, tgt_data_sel = select_intersecting_properties(
        src_props=src_inst.get_data_properties(),
        tgt_props=tgt_inst.get_data_properties(), 
        max_properties=max_properties, 
        intersection_only=True, 
        predicate_entropies=entropies,
    )

    src_obj_sel, tgt_obj_sel = select_intersecting_properties(
        src_props=src_inst.get_object_properties(), 
        tgt_props=tgt_inst.get_object_properties(), 
        max_properties=max_properties, 
        intersection_only=True, 
        predicate_entropies=entropies
    )
    
    def _build(label, type_clause, data_clause, object_clause):
        built_description = (
            f'"{label}"' + (f', {type_clause}' if type_clause else "") 
            + (f' and has attributes: {data_clause}' if data_clause else "") 
            + (f' and has relationships: {object_clause}' if object_clause else "")
        )
        return built_description
    
    src_desc = _build(src_label, src_type_clause, fmt_fn(src_data_sel, max_properties), fmt_fn(src_obj_sel, max_properties))
    tgt_desc = _build(tgt_label, tgt_type_clause, fmt_fn(tgt_data_sel, max_properties), fmt_fn(tgt_obj_sel, max_properties))
    
    prompt_lines = [
        ctx.domain_preamble,
        f'The first is {src_desc}.',
        f'The second is {tgt_desc}.',
        f'Do these refer to the same entity? {ctx.response_instruction}'
    ]
    
    return "\n".join(prompt_lines)




###
# REGISTRY FUNCTION
###

def get_oracle_user_prompt_template_function(
    oupt_name: str, ctx: PromptContext | None = None,
) -> Callable:
    """Resolve a template by name, optionally bound to a prompt context.

    Passing `ctx` is the ordinary path: it returns the `(src, tgt) -> str` callable every
    consumer expects. Every template requires `ctx` as a keyword-only argument, so an
    unbound callable raises TypeError when used rather than rendering under stale state.
    """
    fn = registry.get(oupt_name).fn
    return fn if ctx is None else partial(fn, ctx=ctx)


def bind_ctx(fn: Callable | None, ctx: PromptContext) -> Callable | None:
    """Return `fn` bound to `ctx`, or `fn` unchanged if it is already bound.

    A lane template still needing `ctx` in the per-candidate loop raises `TypeError`,
    which that loop's broad except converts into a warning and a dropped candidate — an
    unbound lane empties silently. Binding is idempotent and cheap, so the builder does
    it defensively.
    """
    if fn is None:
        return None
    try:
        parameters = inspect.signature(fn).parameters
    except (TypeError, ValueError):       # builtins / C callables: nothing to inspect
        return fn
    ctx_parameter = parameters.get("ctx")
    if ctx_parameter is None or ctx_parameter.default is not inspect.Parameter.empty:
        return fn                          # not context-taking, or already bound
    return partial(fn, ctx=ctx)



###
# PROMPT-BUILDING ORCHESTRATION
###

def build_oracle_user_prompts(
    oupt_name, onto_src_filepath, onto_tgt_filepath, m_ask_df,
    OA_source=None, OA_target=None, sibling_selector=None,
    property_prompt_function=None, instance_prompt_function=None,
    property_prompt_name=None, instance_prompt_name=None,
    data_property_prompt_function=None, data_property_prompt_name=None,
    *, ctx: PromptContext) -> dict:

    if OA_source is None or OA_target is None:
        raise ValueError("You must provide params (OA_source, OA_target).")

    ###
    # CLS PROMPT:
    ###

    prompt_function = get_oracle_user_prompt_template_function(oupt_name, ctx)

    if sibling_selector is not None and registry.requires_siblings(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

    ###
    # PROPERTY PROMPT (IF REQUIRED)
    ###

    # An unknown lane template name raises here, matching the class-template resolution
    # above, rather than leaving the lane function None and silently emptying the lane.
    if property_prompt_name and property_prompt_function is None:
        property_prompt_function = get_oracle_user_prompt_template_function(
            property_prompt_name, ctx)

    ###
    # INSTANCE PROMPT (IF REQUIRED)
    ###

    if instance_prompt_name and instance_prompt_function is None:
        instance_prompt_function = get_oracle_user_prompt_template_function(
            instance_prompt_name, ctx)

    ###
    # DATA-PROPERTY PROMPT (OPTIONAL) — DPROP rows use this when configured,
    # otherwise they fall back to the object-property template.
    ###

    if data_property_prompt_name and data_property_prompt_function is None:
        data_property_prompt_function = get_oracle_user_prompt_template_function(
            data_property_prompt_name, ctx)

    ###
    # CONTEXT BINDING (defensive)
    ###
    # Caller-supplied functions bypass the three `is None` branches above and may arrive
    # unbound; an unbound one raises TypeError inside the per-candidate try below, which
    # swallows it and drops the candidate. Binding here is idempotent.

    property_prompt_function = bind_ctx(property_prompt_function, ctx)
    instance_prompt_function = bind_ctx(instance_prompt_function, ctx)
    data_property_prompt_function = bind_ctx(data_property_prompt_function, ctx)

    ###
    # REPORTING
    ###

    info(f"Prompt template function obtained: {oupt_name}")
    if VERBOSE:
        debug(f"CLS PROMPT FN: {prompt_function.__repr__()}")

    if property_prompt_function:
        info(f"Property prompt template: {property_prompt_name}")
        if VERBOSE:
            debug(f"PROP PROMPT FN: {property_prompt_function.__repr__()}")

    if instance_prompt_function:
        info(f"Instance prompt template: {instance_prompt_name}")            
        if VERBOSE:
            debug(f"INSTANCE PROMPT FN: {instance_prompt_function.__repr__()}")

    ###
    # CONSTRUCT M_ASK ORACLE USER PROMPTS
    #  + LOG SKIPPED MAPPINGS
    #  + COUNT CLASS, PROPERTY, AND INSTANCE PROMPTS
    ###

    m_ask_oracle_user_prompts: dict = {}
    skipped_mappings: list = []
    n_class, n_prop, n_inst = 0, 0, 0

    for row in tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], desc="Building the prompts"):
        
        row_series = row[1]
        src_uri, tgt_uri = row_series.iloc[0], row_series.iloc[1]
        # Authoritative LogMap typing takes precedence over ontology-derived typing.
        mask_et = mask_row_entity_type(row_series)

        try:

            # OPROP/DPROP is authoritative output from LogMap. Resolve that lane before
            # class-first ontology dispatch: an undeclared ABox predicate may not be an RDF
            # subject at all, yet still needs a property prompt rather than a skipped row.
            if mask_et in ("OPROP", "DPROP"):
                src_entity, src_type = resolve_entity_as(src_uri, OA_source, mask_et)
                tgt_entity, tgt_type = resolve_entity_as(tgt_uri, OA_target, mask_et)
            else:
                src_entity, src_type = resolve_entity(src_uri, OA_source)
                tgt_entity, tgt_type = resolve_entity(tgt_uri, OA_target)

            lane = resolve_pair_lane(mask_et, src_type, tgt_type)

            # When LogMap authoritatively tags the lane, class-first resolve_entity may have
            # mis-resolved an OWL2-punned URI as a class. Re-resolve any side that disagrees
            # with the lane so the correct entity type + template are used; a side that
            # genuinely cannot resolve to the lane keeps its resolved type and the
            # `src_type != tgt_type` guard below still skips it as inconsistent.
            mask_authoritative = mask_et in ("CLS", "OPROP", "DPROP", "INST")
            if mask_authoritative and lane == "instance":
                if src_type != lane:
                    src_entity, src_type = resolve_entity_as(src_uri, OA_source, lane)
                if tgt_type != lane:
                    tgt_entity, tgt_type = resolve_entity_as(tgt_uri, OA_target, lane)

            ###
            # INSTANCE ALIGNMENT PROMPTS
            ###

            if lane == "instance":

                if src_type != tgt_type:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": f"Mixed types: {src_type}/{tgt_type}"})
                    continue

                if instance_prompt_function is None:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": "No instance template"})
                    continue

                oracle_user_prompt = instance_prompt_function(src_entity, tgt_entity); n_inst += 1

            ###
            # PROPERTY ALIGNMENT PROMPTS (OPROP + DPROP)
            ###

            elif lane == "property":

                if src_type != tgt_type:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": f"Mixed types: {src_type}/{tgt_type}"})
                    continue

                # Route DPROP rows to the data-property template when configured; otherwise
                # fall back to the (datatype-aware) object-property template.
                is_dprop = (mask_et == "DPROP") or getattr(src_entity, "is_data_property", False)
                chosen_prop_fn = (
                    data_property_prompt_function
                    if (is_dprop and data_property_prompt_function is not None)
                    else property_prompt_function
                )

                if chosen_prop_fn is None:
                    skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": "No property template"})
                    continue

                oracle_user_prompt = chosen_prop_fn(src_entity, tgt_entity); n_prop += 1

            else:
                # CLASS ALIGNMENT PROMPTS
                oracle_user_prompt = prompt_function(src_entity, tgt_entity)
                n_class += 1
            
            # index oracle user prompt within m_ask_oracle_user_prompts dict:
            m_ask_oracle_user_prompts[src_uri + PAIRS_SEPARATOR + tgt_uri] = oracle_user_prompt

        except (ClassNotFoundError, Exception) as e:
            skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": str(e)})
            tqdm.write(f"  WARNING: Skipping mapping - {e}")

    if skipped_mappings:
        warning(f"Skipped {len(skipped_mappings)} mappings.")

    info(f"Prompts built: {n_class} cls, {n_prop} prop, {n_inst} inst ({len(m_ask_oracle_user_prompts)} total)", important=True)
    
    return m_ask_oracle_user_prompts



def build_oracle_user_prompts_bidirectional(
        oupt_name, onto_src_filepath, onto_tgt_filepath,
        m_ask_df, OA_source=None, OA_target=None,
        sibling_selector=None,
        property_prompt_function=None, instance_prompt_function=None,
        property_prompt_name=None, instance_prompt_name=None,
        data_property_prompt_function=None, data_property_prompt_name=None,
        *, ctx: PromptContext):
    """Build forward + reverse subsumption prompts for every CLASS candidate.

    Hybrid lanes: an M_ask row that LogMap tags OPROP/DPROP/INST is not a class candidate and
    has no subsumption reading, so when a property/instance template is configured those rows
    are routed through the unidirectional builder exactly as in forward mode (one prompt, one
    verdict). Without lane templates they are skipped, as before (a skipped candidate is a
    reject downstream). Untagged rows are treated as class candidates.
    """
    if OA_source is None or OA_target is None:
        raise ValueError("You must provide params (OA_source, OA_target).")
    lane_templates_configured = any((
        property_prompt_name, property_prompt_function,
        instance_prompt_name, instance_prompt_function,
        data_property_prompt_name, data_property_prompt_function,
    ))
    lane_row_positions: list[int] = []

    prompt_function = get_oracle_user_prompt_template_function(oupt_name, ctx)

    if sibling_selector is not None and registry.requires_siblings(oupt_name):
        prompt_function = partial(prompt_function, sibling_selector=sibling_selector)

    info(f"Prompt template function obtained: {oupt_name} (bidirectional mode)")

    m_ask_oracle_user_prompts = {}
    skipped_mappings = []
    n_non_equiv_skipped = 0
    n_equiv_candidates = 0

    # Ask every row and route by entity type, exactly as the unidirectional builder does; the
    # relation column is LogMap's hypothesised relation, not ground truth, and is never used to
    # filter — filtering on it would make the two arms answer different question sets. A skipped
    # candidate is treated as a reject downstream and excluded from the confusion matrix, so
    # non-class rows are recorded as skipped only as the unidirectional builder would also skip
    # them, keeping the arms comparable.
    for position, row in enumerate(tqdm(m_ask_df.iterrows(), total=m_ask_df.shape[0], desc="Building bidirectional prompts")):

        row_series = row[1]
        src_uri, tgt_uri = row_series.iloc[0], row_series.iloc[1]
        if lane_templates_configured and mask_row_entity_type(row_series) in ("OPROP", "DPROP", "INST"):
            lane_row_positions.append(position)   # forward property/instance lane, built below
            continue

        try:
            src_e = OntologyEntryAttr(src_uri, OA_source)
            tgt_e = OntologyEntryAttr(tgt_uri, OA_target)
            base_key = src_uri + PAIRS_SEPARATOR + tgt_uri
            # forward = (src, tgt) -> "is every src a tgt?"; reverse = (tgt, src) -> the converse.
            # Accepted iff both hold; consult_oracle_bidirectional persists both directions so the
            # AND can be decomposed after the fact. Render both directions before inserting either,
            # so a reverse-side failure cannot leave a dangling forward-only key.
            forward_prompt = prompt_function(src_e, tgt_e)
            reverse_prompt = prompt_function(tgt_e, src_e)
            m_ask_oracle_user_prompts[base_key] = forward_prompt
            m_ask_oracle_user_prompts[base_key + PAIRS_SEPARATOR + "REVERSE"] = reverse_prompt
            n_equiv_candidates += 1
        except (ClassNotFoundError, Exception) as e:
            # Non-class rows (OPROP/DPROP/INST) land here: OntologyEntryAttr cannot resolve them
            # as classes. That matches the unidirectional builder on a track with no property
            # template declared, so the arms stay comparable — but it is recorded, not silent.
            skipped_mappings.append({"src": src_uri, "tgt": tgt_uri, "reason": str(e)})

    if skipped_mappings:
        warning(f"[WARNING] Skipped {len(skipped_mappings)} mappings that could not be built as CLASS "
                f"candidates (expected: non-class rows on tracks with no property/instance template). "
                f"NOTE: a skipped candidate is treated as a REJECT downstream and is EXCLUDED from the "
                f"confusion matrix — check this count against the unidirectional arm before comparing.")

    n_lane_prompts = 0

    if lane_row_positions:

        lane_df = m_ask_df.iloc[lane_row_positions]

        lane_prompts = build_oracle_user_prompts(

            oupt_name, onto_src_filepath, onto_tgt_filepath, lane_df,

            OA_source=OA_source, OA_target=OA_target, sibling_selector=None,

            property_prompt_function=property_prompt_function,

            instance_prompt_function=instance_prompt_function,

            property_prompt_name=property_prompt_name,

            instance_prompt_name=instance_prompt_name,

            data_property_prompt_function=data_property_prompt_function,

            data_property_prompt_name=data_property_prompt_name,

            ctx=ctx,

        )

        n_lane_prompts = len(lane_prompts)

        m_ask_oracle_user_prompts.update(lane_prompts)

    info(f"  Built {n_equiv_candidates * 2} prompts ({n_equiv_candidates} forward + {n_equiv_candidates} reverse) "

         f"from {m_ask_df.shape[0]} M_ask rows; {len(skipped_mappings)} not buildable as class candidates; "

         f"{n_lane_prompts} forward property/instance prompts for {len(lane_row_positions)} typed non-class rows.")

    return m_ask_oracle_user_prompts, n_equiv_candidates, n_non_equiv_skipped
