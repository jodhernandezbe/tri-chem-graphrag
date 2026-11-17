# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the ``analogues`` switch: Section 4.3 context similarity and the pool re-rank."""

import math
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.rag_graph.retrieval.context_profiles import (
    ContextProfiles,
    is_empty,
    rerank_by_context,
)
from src.rag_graph.retrieval.retriever_factory import RetrieverFactory
from src.rag_graph.retrieval.switches import (
    CONTEXT_POOL_SIZE,
    compute_code_idf,
    idf_from_counts,
    resolve_switch,
)

NAICS = {"q": {"3251": 2.0}, "near": {"3251": 3.0}, "far": {"4931": 5.0}, "x": {"3251": 1.0, "4931": 1.0}}
COU = {"q": {"reactant": 2.0}, "near": {"reactant": 1.0}, "far": {"fuel": 4.0}, "x": {"fuel": 1.0}}


def _store() -> ContextProfiles:
    return ContextProfiles(NAICS, COU)


def _hit(cas: str, score: float) -> Dict[str, Any]:
    return {"node": {"cas_number": cas}, "content": "", "score": score}


class TestContextSimilarity:

    def test_identical_profiles_score_one(self):
        store = _store()
        assert store.similarity(store.profile("q"), store.profile("q")) == pytest.approx(1.0)

    def test_disjoint_profiles_score_zero(self):
        store = _store()
        assert store.similarity(store.profile("q"), store.profile("far")) == 0.0

    def test_mean_of_sector_and_use_cosines(self):
        store = _store()
        # "near" shares both its sector and its use with "q" (single-key vectors), so both cosines are 1
        assert store.similarity(store.profile("q"), store.profile("near")) == pytest.approx(1.0)
        # "x" shares only the sector, with a partial cosine, and no use: strictly between 0 and 0.5
        value = store.similarity(store.profile("q"), store.profile("x"))
        assert 0.0 < value < 0.5

    def test_section_4_3_weighting(self):
        naics = {"a": {"k1": 3.0, "k2": 1.0}, "b": {"k1": 1.0}, "c": {"k2": 1.0}}
        store = ContextProfiles(naics, {})
        idf = {k: math.log((3 + 1) / (df + 1)) + 1 for k, df in (("k1", 2), ("k2", 2))}
        va = [math.log1p(3.0) * idf["k1"], math.log1p(1.0) * idf["k2"]]
        vb = [math.log1p(1.0) * idf["k1"], 0.0]
        expected = sum(x * y for x, y in zip(va, vb)) / (math.hypot(*va) * math.hypot(*vb))
        assert store.similarity(store.profile("a"), store.profile("b")) == pytest.approx(expected / 2)

    def test_from_rows_and_unknown_chemical(self):
        store = ContextProfiles.from_rows([{"cas": "a", "key": "3251", "n": 2}], [{"cas": "a", "key": "fuel", "n": 1}])
        assert store.profile("a") == ({"3251": 2.0}, {"fuel": 1.0})
        assert is_empty(store.profile("missing"))


class TestRerank:

    def test_context_order_replaces_molecular_order(self):
        hits = [_hit("far", 0.99), _hit("near", 0.80), _hit("x", 0.70)]
        out = rerank_by_context(hits, _store(), _store().profile("q"), top_k=2)
        assert [h["node"]["cas_number"] for h in out] == ["near", "x"]
        assert all("context_similarity" in h for h in out)
        assert out[0]["score"] == 0.80  # molecular score kept

    def test_no_visible_profile_keeps_molecular_order(self):
        hits = [_hit("far", 0.99), _hit("near", 0.80), _hit("x", 0.70)]
        out = rerank_by_context(hits, _store(), ({}, {}), top_k=2)
        assert out == hits[:2] and "context_similarity" not in out[0]

    def test_overrides_hide_a_candidates_profile(self):
        hits = [_hit("near", 0.9), _hit("x", 0.8)]
        out = rerank_by_context(hits, _store(), _store().profile("q"), top_k=2, overrides={"near": ({}, {})})
        assert out[0]["node"]["cas_number"] == "x" and out[1]["context_similarity"] == 0.0

    def test_ties_break_by_molecular_score(self):
        hits = [_hit("far", 0.5), _hit("far2", 0.9)]
        out = rerank_by_context(hits, _store(), _store().profile("q"), top_k=2)
        assert [h["node"]["cas_number"] for h in out] == ["far2", "far"]


def _doc(cas: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"cas_number": cas}
    doc.page_content = ""
    return doc


def _factory(store: ContextProfiles) -> RetrieverFactory:
    factory = RetrieverFactory.__new__(RetrieverFactory)
    factory._top_k = 2
    factory._chem_embedding_type = "Hybrid"
    pool = [(_doc("q"), 1.0), (_doc("far"), 0.9), (_doc("near"), 0.8), (_doc("x"), 0.7)]
    vector = MagicMock()
    vector.similarity_search_with_score.side_effect = lambda q, k: pool[:k]
    factory.get_vector_store = lambda t: vector  # type: ignore[method-assign]
    factory.context_profiles = lambda split=None: store  # type: ignore[method-assign]
    factory._vector = vector  # type: ignore[attr-defined]
    return factory


@pytest.fixture(autouse=True)
def _bypass_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.rag_graph.retrieval.retriever_factory.guarded_search",
        lambda store, q, k: store.similarity_search_with_score(q, k=k),
    )


class TestChemicalNeighbours:

    def test_config_default_is_molecular(self):
        assert resolve_switch("analogues") == "molecular"

    def test_molecular_takes_top_k_only(self):
        factory = _factory(_store())
        hits = factory.chemical_neighbours("CCO", analogues="molecular")
        assert [h["node"]["cas_number"] for h in hits] == ["q", "far"]
        assert factory._vector.similarity_search_with_score.call_args.kwargs["k"] == 2  # type: ignore[attr-defined]

    def test_context_reranks_a_wider_pool_and_keeps_top_k(self):
        factory = _factory(_store())
        hits = factory.chemical_neighbours("CCO", analogues="context", report_split="index", query_cas="q")
        assert factory._vector.similarity_search_with_score.call_args.kwargs["k"] == CONTEXT_POOL_SIZE  # type: ignore[attr-defined]
        assert [h["node"]["cas_number"] for h in hits] == ["q", "near"]

    def test_query_found_through_identical_hit(self):
        hits = _factory(_store()).chemical_neighbours("CCO", analogues="context")
        assert [h["node"]["cas_number"] for h in hits] == ["q", "near"]

    def test_unknown_query_keeps_molecular_order(self):
        store = ContextProfiles({"far": {"4931": 1.0}}, {})
        hits = _factory(store).chemical_neighbours("CCO", analogues="context", query_cas="not-in-graph")
        assert [h["node"]["cas_number"] for h in hits] == ["q", "far"]

    def test_explicit_visible_profile_is_used(self):
        hits = _factory(_store()).chemical_neighbours(
            "CCO", analogues="context", query_profile=({"4931": 1.0}, {"fuel": 1.0}), profile_overrides={"q": ({}, {})}
        )
        assert [h["node"]["cas_number"] for h in hits] == ["far", "x"]


class TestCodeIdf:

    def test_idf_definition(self):
        idf = compute_code_idf({"a": {"S1", "S2"}, "b": {"S1"}, "c": {"S1", "S3"}, "d": {"S1"}})
        assert idf["S1"] == 0.0 and idf["S2"] == pytest.approx(math.log(4)) and idf["S3"] == idf["S2"]

    def test_idf_from_counts_matches(self):
        profiles: Dict[str, List[str]] = {"a": ["S1", "S2"], "b": ["S1"], "c": ["S3"]}
        assert idf_from_counts({"S1": 2, "S2": 1, "S3": 1}, 3) == compute_code_idf(profiles)
