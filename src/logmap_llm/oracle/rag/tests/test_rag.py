"""
Query-specific RAG few-shot retrieval — test suite.

Runs on CPU with the deterministic HashingEncoder + HeuristicTokenCounter (no torch/JVM/GPU).
Covers: k=0/odd/even, empty/undersized pools, all four entity kinds in one run, forward &
bidirectional, concurrency isolation, leakage & reverse-pair exclusion, duplicate URIs &
same-local-name-different-namespace, missing labels, hard-negative validity, stale/corrupt
index cache, fixed-seed determinism, token-budget overflow, exact True/False & Yes/No formats,
static-random & static-hard back-compat, and explicit recorded fallback (no silent zero-shot).
"""
from __future__ import annotations

import concurrent.futures
import os

import pytest

from logmap_llm.oracle.rag import (
    RagRetriever, RagConfig, FallbackPolicy, Mode, EntityKind, Source, Direction,
    QueryMapping, TypedCorpus, HashingEncoder, HeuristicTokenCounter, RagRetrievalError,
)


# --------------------------------------------------------------------------
# fixtures / builders
# --------------------------------------------------------------------------

def _local(iri: str) -> str:
    for sep in ("#", "/"):
        if sep in iri:
            return iri.rsplit(sep, 1)[-1]
    return iri


# render_fn(src_iri, tgt_iri, payload) -> user prompt. payload may carry labels.
def render_fn(src, tgt, payload):
    if isinstance(payload, dict):
        sl = payload.get("src", _local(src))
        tl = payload.get("tgt", _local(tgt))
    else:
        sl, tl = _local(src), _local(tgt)
    return f"Is <{sl}> equivalent to <{tl}> ?"


CLS_POS = [
    # (src_iri, tgt_iri, src_label, tgt_label) — embed_text made from labels
    ("http://a#Lamp", "http://b#Lamp", "lamp", "lamp"),
    ("http://a#Glow", "http://b#LampFixture", "glow fixture", "lamp fixture"),
    ("http://a#Shelf", "http://b#Shelf", "shelf", "shelf"),
    ("http://a#Bench", "http://b#Seating", "bench", "seating module"),
    ("http://a#Clock", "http://b#Chronometer", "clock", "chronometer"),
    ("http://a#Stool", "http://b#Pad", "stool", "padded module"),
    ("http://a#Table", "http://b#Slab", "table", "slab module"),
    ("http://a#Door", "http://b#Panel", "door", "panelled board"),
    ("http://a#Rug", "http://b#Mat", "rug", "matted board"),
    ("http://a#Mirror", "http://b#Glass", "eye glass", "glassy module"),
]
OPROP_POS = [
    ("http://a#hasPart", "http://b#partOfInverse", "has part", "has part"),
    ("http://a#contains", "http://b#includes", "contains", "includes"),
    ("http://a#locatedIn", "http://b#inRegion", "located in", "in region"),
    ("http://a#connectedTo", "http://b#linkedWith", "connected to", "linked with"),
]
DPROP_POS = [
    ("http://a#hasName", "http://b#label", "has name", "label"),
    ("http://a#hasAge", "http://b#ageValue", "has age", "age value"),
    ("http://a#hasWeight", "http://b#weightKg", "has weight", "weight kg"),
    ("http://a#hasCode", "http://b#codeStr", "has code", "code string"),
]
INST_POS = [
    ("http://a#smith", "http://b#jordanSmith", "jordan smith", "jordan t smith"),
    ("http://a#reese", "http://b#mxReese", "reese", "mx reese"),
    ("http://a#morgan", "http://b#caseyLee", "casey lee morgan", "morgan"),
    ("http://a#sam", "http://b#robotSam", "sam robot", "sam"),
]


def build_corpus(dataset_sha="testsha", language="en"):
    c = TypedCorpus(dataset_sha=dataset_sha, language=language)
    for kind, rows in [(EntityKind.CLS, CLS_POS), (EntityKind.OPROP, OPROP_POS),
                       (EntityKind.DPROP, DPROP_POS), (EntityKind.INST, INST_POS)]:
        pairs = [(s, t) for (s, t, sl, tl) in rows]
        labels = {(s, t): (sl, tl) for (s, t, sl, tl) in rows}
        c.add_pairs(
            pairs, kind=kind, source=Source.GOLD,
            embed_text_fn=lambda s, t, L=labels: f"{L[(s, t)][0]} | {L[(s, t)][1]}",
            payload_fn=lambda s, t, L=labels: {"src": L[(s, t)][0], "tgt": L[(s, t)][1]},
        )
    return c


def make_retriever(config=None, corpus=None, cache_dir=None):
    corpus = corpus or build_corpus()
    config = config or RagConfig(mode=Mode.QUERY_RAG, k=4)
    return RagRetriever(
        corpus=corpus, encoder=HashingEncoder(dim=128), render_fn=render_fn,
        config=config, token_counter=HeuristicTokenCounter(), cache_dir=cache_dir,
    )


def cls_query(src="http://q#Lamp", tgt="http://q#Glow", text="lamp glow module"):
    return QueryMapping(src_iri=src, tgt_iri=tgt, kind=EntityKind.CLS, embed_text=text)


# --------------------------------------------------------------------------
# k handling
# --------------------------------------------------------------------------

def test_k_zero_returns_empty():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=0))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=0, corpus_id="c")
    assert res.examples == ()
    assert res.trace.effective_mode == "zero_shot"
    assert res.trace.effective_k == 0


def test_zero_shot_mode_ignores_k():
    r = make_retriever(RagConfig(mode=Mode.ZERO_SHOT, k=8))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    assert res.examples == ()
    assert res.trace.effective_mode == "zero_shot"


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 8])
def test_odd_even_k_counts(k):
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=k))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=k, corpus_id="c")
    # non-bidirectional: at most k examples; with a full pool we expect exactly k
    assert len(res.examples) == k
    assert res.trace.effective_k == k


# --------------------------------------------------------------------------
# empty / undersized pools + explicit fallback
# --------------------------------------------------------------------------

def test_empty_pool_zero_shot_fallback_recorded():
    c = build_corpus()
    c.positives[EntityKind.INST] = []  # wipe INST pool
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), corpus=c)
    q = QueryMapping("http://q#a", "http://q#b", EntityKind.INST, "some instance")
    res = r.retrieve(q, EntityKind.INST, "=", "inst_labels_only", k=4, corpus_id="c")
    assert res.examples == ()
    assert res.trace.effective_mode == "zero_shot"
    assert res.trace.fallback_reason and "empty" in res.trace.fallback_reason.lower()


def test_empty_pool_strict_raises():
    c = build_corpus()
    c.positives[EntityKind.INST] = []
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4, fallback=FallbackPolicy.strict()), corpus=c)
    q = QueryMapping("http://q#a", "http://q#b", EntityKind.INST, "x")
    with pytest.raises(RagRetrievalError):
        r.retrieve(q, EntityKind.INST, "=", "inst", k=4, corpus_id="c")


def test_undersized_reduce_k_recorded():
    # tiny corpus (2 positives) but request k=8 -> fewer, recorded
    c = TypedCorpus(dataset_sha="s")
    c.add_pairs([("http://a#X", "http://b#X"), ("http://a#Y", "http://b#Y")],
                kind=EntityKind.CLS, source=Source.GOLD,
                embed_text_fn=lambda s, t: f"{_local(s)} {_local(t)}")
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8), corpus=c)
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    assert len(res.examples) < 8
    assert res.trace.fallback_reason and "undersized" in res.trace.fallback_reason.lower()


def test_undersized_strict_raises():
    c = TypedCorpus(dataset_sha="s")
    c.add_pairs([("http://a#X", "http://b#X")], kind=EntityKind.CLS, source=Source.GOLD,
                embed_text_fn=lambda s, t: "x")
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8,
                                 fallback=FallbackPolicy(on_undersized="error")), corpus=c)
    with pytest.raises(RagRetrievalError):
        r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c")


# --------------------------------------------------------------------------
# all four kinds in one run — typed pools never mix
# --------------------------------------------------------------------------

def test_all_four_kinds_typed_pools():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=2))
    queries = {
        EntityKind.CLS: cls_query(),
        EntityKind.OPROP: QueryMapping("http://q#hasPart", "http://q#partOf", EntityKind.OPROP, "has part member"),
        EntityKind.DPROP: QueryMapping("http://q#hasName", "http://q#name", EntityKind.DPROP, "has name label"),
        EntityKind.INST: QueryMapping("http://q#smith", "http://q#doctorSmith", EntityKind.INST, "jordan smith doctor"),
    }
    for kind, q in queries.items():
        res = r.retrieve(q, kind, "=", "tmpl", k=2, corpus_id="c")
        assert len(res.examples) == 2
        assert all(e.kind == kind for e in res.examples), f"{kind} query leaked another kind"


# --------------------------------------------------------------------------
# forward vs bidirectional
# --------------------------------------------------------------------------

def test_forward_only_single_direction():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4, bidirectional=False))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=4, corpus_id="c")
    assert all(e.direction == Direction.FORWARD for e in res.examples)


def test_bidirectional_forward_and_reverse():
    # inline corpus with no payload so render uses (src,tgt) directly and the swap is checkable
    c = TypedCorpus(dataset_sha="s")
    c.add_pairs([("http://a#Alpha", "http://b#Beta"), ("http://a#Gamma", "http://b#Delta"),
                 ("http://a#Eps", "http://b#Zeta")], kind=EntityKind.CLS, source=Source.GOLD,
                embed_text_fn=lambda s, t: f"{_local(s)} {_local(t)}")
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4, bidirectional=True), corpus=c)
    res = r.retrieve(cls_query(text="alpha beta"), EntityKind.CLS, "=", "sub_labels_only", k=4, corpus_id="c")
    dirs = [e.direction for e in res.examples]
    assert Direction.FORWARD in dirs and Direction.REVERSE in dirs
    # a reverse example swaps src/tgt in the rendered prompt vs its forward partner
    fwd = next(e for e in res.examples if e.direction == Direction.FORWARD)
    rev = next(e for e in res.examples if e.direction == Direction.REVERSE
               and e.example_id == fwd.example_id)
    assert render_fn(fwd.src_iri, fwd.tgt_iri, None) == fwd.prompt_text
    assert render_fn(rev.tgt_iri, rev.src_iri, None) == rev.prompt_text
    assert fwd.prompt_text != rev.prompt_text


# --------------------------------------------------------------------------
# concurrency isolation
# --------------------------------------------------------------------------

def test_concurrent_requests_match_serial():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4))
    r.warmup()
    queries = [QueryMapping(f"http://q#s{i}", f"http://q#t{i}", EntityKind.CLS,
                            ["lamp", "shelf", "clock", "bench", "stool", "door"][i % 6])
               for i in range(40)]
    serial = [tuple(e.example_id for e in r.retrieve(q, EntityKind.CLS, "=", "syn", 4, "c").examples)
              for q in queries]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        threaded = list(ex.map(
            lambda q: tuple(e.example_id for e in r.retrieve(q, EntityKind.CLS, "=", "syn", 4, "c").examples),
            queries))
    assert threaded == serial


# --------------------------------------------------------------------------
# leakage / exclusion
# --------------------------------------------------------------------------

def test_query_and_reverse_excluded():
    # make the query be a corpus positive; it (and its reverse) must never be returned
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8))
    q = QueryMapping("http://a#Lamp", "http://b#Lamp", EntityKind.CLS, "lamp lamp")
    res = r.retrieve(q, EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    key = frozenset({"http://a#Lamp", "http://b#Lamp"})
    assert all(frozenset({e.src_iri, e.tgt_iri}) != key for e in res.examples)
    # reverse form also excluded
    qrev = QueryMapping("http://b#Lamp", "http://a#Lamp", EntityKind.CLS, "lamp lamp")
    res2 = r.retrieve(qrev, EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    assert all(frozenset({e.src_iri, e.tgt_iri}) != key for e in res2.examples)


def test_mask_exclusion_keys_honoured():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8))
    mask = {frozenset({"http://a#Shelf", "http://b#Shelf"}),
            frozenset({"http://a#Bench", "http://b#Seating"})}
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c", exclude_keys=mask)
    for e in res.examples:
        assert frozenset({e.src_iri, e.tgt_iri}) not in mask


def test_known_positives_never_negatives():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    known = {frozenset({s, t}) for (s, t, _, _) in CLS_POS}
    for e in res.examples:
        if not e.label:  # a negative
            assert frozenset({e.src_iri, e.tgt_iri}) not in known


# --------------------------------------------------------------------------
# dedup / namespace distinctness
# --------------------------------------------------------------------------

def test_duplicate_pairs_deduped():
    c = TypedCorpus(dataset_sha="s")
    added = c.add_pairs([("http://a#X", "http://b#X"), ("http://a#X", "http://b#X"),
                         ("http://b#X", "http://a#X")],  # reverse == same unordered key
                        kind=EntityKind.CLS, source=Source.GOLD, embed_text_fn=lambda s, t: "x")
    assert added == 1
    assert len(c.positive_pool(EntityKind.CLS)) == 1


def test_same_localname_different_namespace_distinct():
    c = TypedCorpus(dataset_sha="s")
    added = c.add_pairs([("http://ns1#Cell", "http://tgt#Cell"),
                         ("http://ns2#Cell", "http://tgt#Cell")],
                        kind=EntityKind.CLS, source=Source.GOLD, embed_text_fn=lambda s, t: "cell")
    assert added == 2  # different full IRIs -> distinct despite same local name


# --------------------------------------------------------------------------
# missing labels
# --------------------------------------------------------------------------

def test_missing_labels_no_crash():
    c = TypedCorpus(dataset_sha="s")
    c.add_pairs([("http://a#", "http://b#"), ("http://a#P", "http://b#Q")],
                kind=EntityKind.CLS, source=Source.GOLD,
                embed_text_fn=lambda s, t: "")  # empty embed text (missing labels)
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=2), corpus=c)
    q = QueryMapping("http://q#a", "http://q#b", EntityKind.CLS, "")  # empty query text too
    res = r.retrieve(q, EntityKind.CLS, "=", "syn", k=2, corpus_id="c")
    assert isinstance(res.examples, tuple)  # no exception; may be fewer than k


# --------------------------------------------------------------------------
# hard-negative validity
# --------------------------------------------------------------------------

def test_query_rag_hard_negatives_are_constructed_not_random():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=6,
                                 fallback=FallbackPolicy(allow_random_negatives=False)))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=6, corpus_id="c")
    negs = [e for e in res.examples if not e.label]
    assert negs, "expected some hard negatives"
    assert all(e.source == Source.CONSTRUCTED for e in negs)
    assert all(e.example_id.startswith("neg:hard:") for e in negs)
    # no random fallback should have been needed with a full pool
    assert not (res.trace.fallback_reason and "random" in res.trace.fallback_reason)


# --------------------------------------------------------------------------
# cache: stale / corrupt handling
# --------------------------------------------------------------------------

def test_cache_roundtrip_and_stale_rebuild(tmp_path):
    cache = str(tmp_path / "idxcache")
    corpus = build_corpus()
    r1 = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), corpus=corpus, cache_dir=cache)
    r1.warmup()  # writes cache
    files = os.listdir(cache)
    assert any(f.startswith("ragindex-") for f in files)
    # a second retriever over the same corpus loads from cache (rebuilt_reason None)
    r2 = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), corpus=build_corpus(), cache_dir=cache)
    idx = r2._positive_index(EntityKind.CLS)
    assert idx.rebuilt_reason is None
    # a different corpus (extra pair -> different corpus_hash -> different key) rebuilds
    corpus3 = build_corpus()
    corpus3.add_pairs([("http://a#New", "http://b#New")], kind=EntityKind.CLS,
                      source=Source.GOLD, embed_text_fn=lambda s, t: "new new")
    r3 = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), corpus=corpus3, cache_dir=cache)
    idx3 = r3._positive_index(EntityKind.CLS)
    assert idx3.rebuilt_reason is not None


def test_corrupt_cache_rebuilds(tmp_path):
    cache = str(tmp_path / "idxcache")
    r1 = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), cache_dir=cache)
    r1.warmup()
    # corrupt every cache file
    for f in os.listdir(cache):
        with open(os.path.join(cache, f), "wb") as fh:
            fh.write(b"not a real npz")
    r2 = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4), cache_dir=cache)
    idx = r2._positive_index(EntityKind.CLS)  # must not raise; rebuilds
    assert idx.rebuilt_reason is not None
    assert idx.matrix.shape[0] == len(idx.examples)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------

def test_fixed_seed_determinism():
    q = cls_query()
    outs = []
    for _ in range(3):
        r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=6, seed=42))
        res = r.retrieve(q, EntityKind.CLS, "=", "syn", k=6, corpus_id="c")
        outs.append(tuple((e.example_id, e.direction.value) for e in res.examples))
    assert outs[0] == outs[1] == outs[2]


def test_static_random_seed_determinism():
    q = cls_query()
    a = make_retriever(RagConfig(mode=Mode.STATIC_RANDOM, k=6, seed=7))
    b = make_retriever(RagConfig(mode=Mode.STATIC_RANDOM, k=6, seed=7))
    ra = a.retrieve(q, EntityKind.CLS, "=", "syn", 6, "c")
    rb = b.retrieve(q, EntityKind.CLS, "=", "syn", 6, "c")
    assert [e.example_id for e in ra.examples] == [e.example_id for e in rb.examples]


# --------------------------------------------------------------------------
# token budget
# --------------------------------------------------------------------------

def test_token_budget_overflow_truncates_examples_only():
    # each rendered example costs ~ (prompt+answer)/4 tokens; set a tiny budget
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=8, token_budget=20))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=8, corpus_id="c")
    assert res.trace.tokens_used <= 20
    assert len(res.examples) < 8
    assert res.trace.fallback_reason and "budget" in res.trace.fallback_reason.lower()


def test_generous_budget_keeps_all():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4, token_budget=100000))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=4, corpus_id="c")
    assert len(res.examples) == 4


# --------------------------------------------------------------------------
# answer formats (exact strings)
# --------------------------------------------------------------------------

@pytest.mark.parametrize("pos,neg", [
    ('{"answer": true}', '{"answer": false}'),
    ('True', 'False'),
    ('{"answer": "Yes"}', '{"answer": "No"}'),
    ('Yes', 'No'),
])
def test_exact_answer_format_strings(pos, neg):
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4, answer_pos=pos, answer_neg=neg))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=4, corpus_id="c")
    for e in res.examples:
        assert e.answer_text == (pos if e.label else neg)


# --------------------------------------------------------------------------
# static-mode backward compatibility
# --------------------------------------------------------------------------

def test_static_hard_backcompat():
    r = make_retriever(RagConfig(mode=Mode.STATIC_HARD, k=6))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=6, corpus_id="c")
    assert res.trace.effective_mode == "static_hard"
    negs = [e for e in res.examples if not e.label]
    assert all(e.source == Source.CONSTRUCTED for e in negs)


def test_static_random_backcompat_uses_random_negatives():
    r = make_retriever(RagConfig(mode=Mode.STATIC_RANDOM, k=6))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=6, corpus_id="c")
    assert res.trace.effective_mode == "static_random"
    negs = [e for e in res.examples if not e.label]
    assert all(e.example_id.startswith("neg:rand:") for e in negs)


# --------------------------------------------------------------------------
# trace completeness + no silent degradation
# --------------------------------------------------------------------------

def test_trace_has_required_fields():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4))
    tr = r.retrieve(cls_query(), EntityKind.CLS, "=", "class_role_signature", k=4, corpus_id="c").trace
    d = tr.to_dict()
    for f in ["requested_mode", "effective_mode", "requested_k", "effective_k", "entity_type",
              "relation", "prompt_family", "answer_format", "selected", "exclusions",
              "corpus_hash", "index_hash", "encoder_repo", "encoder_revision",
              "preprocessing_version", "token_budget", "tokens_used"]:
        assert f in d
    assert d["encoder_repo"] and d["index_hash"] and d["corpus_hash"]
    assert len(d["selected"]) == len(tr.selected)


def test_selected_trace_matches_examples():
    r = make_retriever(RagConfig(mode=Mode.QUERY_RAG, k=4))
    res = r.retrieve(cls_query(), EntityKind.CLS, "=", "syn", k=4, corpus_id="c")
    ids_examples = [e.example_id for e in res.examples]
    ids_trace = [s["example_id"] for s in res.trace.selected]
    assert ids_examples == ids_trace
