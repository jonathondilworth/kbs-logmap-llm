'''
Bridging between the worlds of Python and Java (LogMap).
'''

import math
from pathlib import Path

import pandas as pd

from logmap_llm.constants import (
    COL_CONFIDENCE,
    COL_ENTITY_TYPE,
    COL_RELATION,
    COL_SOURCE_ENTITY_URI,
    COL_TARGET_ENTITY_URI,
    DEFAULT_CONFIDENCE_FALLBACK,
    M_ASK_COLUMNS,
    EntityType,
    EntityRelation,
    PAIRS_SEPARATOR,
    VERBOSE,
)

from logmap_llm.utils.data import (
    dedupe_m_ask_by_uri_pair,
    filter_accepted_predictions,
)
from logmap_llm.utils.logging import debug, warning

# Entity type representations:
# LogMap represents its 5 entity types as integers in <MappingObjectStr>
# objects and as strings in its m_ask output file. LogMap-LLM maps the
# integers to those same strings and uses the strings alone, both in memory
# and in LogMap-LLM output files.

entityType_int_2_str = {
    0: EntityType.CLASS.value,
    1: EntityType.DATAPROPERTY.value,
    2: EntityType.OBJECTPROPERTY.value,
    3: EntityType.INSTANCE.value,
    4: EntityType.UNKNOWN.value,
}

entityType_str_2_int = {
    EntityType.CLASS.value: 0,
    EntityType.DATAPROPERTY.value: 1,
    EntityType.OBJECTPROPERTY.value: 2,
    EntityType.INSTANCE.value: 3,
    EntityType.UNKNOWN.value: 4,
}

# Relation representations:
# LogMap represents its 3 relations as integers in <MappingObjectStr> objects
# and as strings in its m_ask output file; as above, LogMap-LLM maps the
# integers to those same strings and uses the strings alone.

#  0 - < - subClassOf    (src_entity subClassOf tgt_entity)
# -1 - > - superClassOf  (src_entity superClassOf tgt_entity)
# -2 - = - equivalence   (src_entity equivalent tgt_entity)

relation_int_2_str = {
    0: EntityRelation.SUBCLASSOF.value,
    -1: EntityRelation.SUPERCLASSOF.value,
    -2: EntityRelation.EQUIVALENCE.value,
}

relation_str_2_int = {
    EntityRelation.SUBCLASSOF.value: 0,
    EntityRelation.SUPERCLASSOF.value: -1,
    EntityRelation.EQUIVALENCE.value: -2,
}

# Java -> Python output format for M_ask (mappings to ask):
# LogMap dumps its uncertain mappings (M_ask) — normally destined for a human
# oracle — to a headerless .txt file so they can be forwarded to an LLM or any
# other valid oracle. Rows are structured according to the following header
# (column names are defined in logmap_llm.constants):

# source_entity_uri|target_entity_uri|relation|confidence|entityType

# Example M_ask row:
#   http://example.org/a#Widget|http://example.org/b#Gadget|=|0.73|CLS

def get_m_ask_column_names() -> list[str]:
    """
    Returns M_ASK_COLUMNS as a mutable list
    """
    return list(M_ASK_COLUMNS)


def load_m_ask_from_file(filepath: Path) -> pd.DataFrame:
    """
    Load a (headerless, pipe-delimited) LogMap m_ask file as a pd.DataFrame
    and set the column names to those defined above (M_ASK_COLUMNS). Duplicate
    (source, target) rows are collapsed (see dedupe_m_ask_by_uri_pair).
    """
    try:
        m_ask_df = pd.read_csv(filepath, sep=PAIRS_SEPARATOR, header=None)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=get_m_ask_column_names())
    m_ask_df.columns = get_m_ask_column_names()
    return dedupe_m_ask_by_uri_pair(m_ask_df)


###
# Functions to enable roundtripping between LogMap (Java) and LogMap-LLM (Python)
###

# Confidence coercion at the Java boundary: when the oracle accepts a mapping
# but no usable logprob token was emitted, calculate_logprobs_confidence
# returns float('nan'); NaN is not a semantically safe value for LogMap's
# MappingObjectStr(..., double conf, ...), so it is coerced to a concrete
# default (1.0, i.e. confident 'True'; see constants.py). This only fires for
# oracle predictions that accept the mapping.

def _coerce_confidence_for_java(raw_conf, row_index: int | None = None) -> float:
    """
    Normalises a Python confidence value to a finite float for 'double conf'.
    Handles NaN | None | any non-float value (should only trigger when mapping=True).
    """
    if raw_conf is None:
        if VERBOSE:
            debug(f"(_coerce_confidence_for_java) None confidence at row {row_index}; hits fallback.")
        return DEFAULT_CONFIDENCE_FALLBACK

    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        if VERBOSE:
            warning(f"(_coerce_confidence_for_java) non-numeric confidenceat row {row_index}; using fallback")
        return DEFAULT_CONFIDENCE_FALLBACK

    if math.isnan(conf) or math.isinf(conf):
        if VERBOSE:
            debug(f"(_coerce_confidence_for_java) NaN/inf confidence at row {row_index}; using fallback.")
        return DEFAULT_CONFIDENCE_FALLBACK

    return conf


###
# Bridging interface
###


def java_mappings_2_python(
    mappings_java,
    *,
    dedupe_uri_pairs: bool = False,
) -> pd.DataFrame:
    '''
    Convert mappings from LogMap's Java representation (a java.util.HashSet of
    <MappingObjectStr> objects) to a pandas DataFrame.

    dedupe_uri_pairs collapses duplicate directional URI pairs; enable it only
    at the M_ask boundary, where asking the same candidate twice is undesirable
    — full alignments must preserve rows that differ by relation or type.
    '''

    # convert java.util.HashSet to Object[] array
    mappings_java = mappings_java.toArray()

    src_entity_uris: list[str] = []
    tgt_entity_uris: list[str] = []
    relations: list[str] = []
    confidences: list[float] = []
    entity_types: list[str] = []

    for mapping_java in mappings_java:

        src_entity_uris.append(str(mapping_java.getIRIStrEnt1()))
        tgt_entity_uris.append(str(mapping_java.getIRIStrEnt2()))

        # convert LogMap's integer relation to its string representation
        relation_int = mapping_java.getMappingDirection()

        if relation_int not in relation_int_2_str:
            raise ValueError(f'Relation {relation_int} not recognised')

        relations.append(relation_int_2_str[relation_int])

        confidences.append(float(mapping_java.getConfidence()))

        # convert LogMap's integer entity type to its string representation
        entityType_int = mapping_java.getTypeOfMapping()
        if entityType_int not in entityType_int_2_str:
            raise ValueError(f'Entity type {entityType_int} not recognised')

        entity_types.append(entityType_int_2_str[entityType_int])

    mappings_df = pd.DataFrame(data={
        COL_SOURCE_ENTITY_URI: src_entity_uris,
        COL_TARGET_ENTITY_URI: tgt_entity_uris,
        COL_RELATION: relations,
        COL_CONFIDENCE: confidences,
        COL_ENTITY_TYPE: entity_types,
    })
    # Rows arrive in java.util.HashSet iteration order, which is unspecified and
    # would otherwise reach the persisted predictions CSV and refined TSV row
    # order — and, via the keep='first' dedupe below, even decide which lane of
    # a mixed-use predicate survives as the asked candidate. Sorting canonically
    # at the boundary makes every downstream ordering a function of content,
    # not of JVM hashing.
    mappings_df = mappings_df.sort_values(
        by=[COL_SOURCE_ENTITY_URI, COL_TARGET_ENTITY_URI, COL_RELATION, COL_ENTITY_TYPE],
        kind="mergesort",
    ).reset_index(drop=True)
    if dedupe_uri_pairs:
        return dedupe_m_ask_by_uri_pair(mappings_df)
    return mappings_df



def python_oracle_mapping_predictions_2_java(m_ask_df_ext):
    '''
    Convert Python LLM Oracle mapping predictions (LogMap's m_ask output as a
    DataFrame, extended with Oracle predictions) to a java.util.HashSet of
    LogMap <MappingObjectStr> objects.

    LogMap expects to receive only True mappings from an Oracle, so as well as
    converting datatypes this function filters out mappings predicted False.
    '''
    # Keep Java imports at the actual Java boundary. Pure mapping-file helpers can
    # then be imported by planners/tests before a JVM has been started.
    from java.util import HashSet  # type: ignore
    from uk.ac.ox.krr.logmap2.mappings.objects import MappingObjectStr  # type: ignore

    # keep only mappings with an Oracle prediction of True
    # (excludes prediction values of False and 'error')
    accepted = filter_accepted_predictions(m_ask_df_ext)

    m_ask_oracle_preds_true: list = []

    for row in accepted.itertuples():

        iri1 = row.source_entity_uri
        iri2 = row.target_entity_uri
        conf = _coerce_confidence_for_java(row.Oracle_confidence, row_index=row.Index)

        # integer representations recognised by LogMap
        if row.relation not in relation_str_2_int:
            raise ValueError(f'Entity relation {row.relation} not recognised')

        relation_int = relation_str_2_int[row.relation]

        if row.entityType not in entityType_str_2_int:
            raise ValueError(f'Entity type {row.entityType} not recognised')

        entityType_int = entityType_str_2_int[row.entityType]

        mos = MappingObjectStr(iri1, iri2, conf, relation_int, entityType_int)
        m_ask_oracle_preds_true.append(mos)

    m_ask_oracle_preds_java = HashSet(m_ask_oracle_preds_true)

    return m_ask_oracle_preds_java
