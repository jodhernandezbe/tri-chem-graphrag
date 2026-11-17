# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the per-call chemical embedding choice on the fused path and its embedding-keyed caches."""

from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

from src.rag_graph.retrieval.retriever_factory import RetrieverFactory, _FactorySource

HITS: Dict[str, List[str]] = {"MorganFingerprint": ["A", "B"], "MolDeBERTa": ["C", "D"], "Hybrid": ["E", "F"]}


def _doc(cas: str) -> MagicMock:
    doc = MagicMock()
    doc.metadata = {"cas_number": cas}
    return doc


def _factory() -> RetrieverFactory:
    factory = RetrieverFactory.__new__(RetrieverFactory)
    factory._chem_embedding_type = "Hybrid"
    factory.searched: List[str] = []  # type: ignore[misc]

    def store(embedding: str) -> MagicMock:
        mock = MagicMock()
        mock.embedding = embedding
        return mock

    factory.get_vector_store = store  # type: ignore[method-assign]
    factory.candidate_hits = lambda smiles, embedding=None: []  # type: ignore[method-assign]
    return factory


@pytest.fixture(autouse=True)
def _guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.rag_graph.retrieval.retriever_factory.guarded_search",
        lambda store, q, k: [(_doc(c), 0.9) for c in HITS[store.embedding]],
    )


def _cas(source: _FactorySource) -> List[str]:
    return [h[0] for h in source.chemical_hits("CCO", frozenset())]


class TestResolve:
    def test_default_is_configured(self) -> None:
        assert _factory().resolve_chemical_embedding() == "Hybrid"

    def test_override_wins_and_is_validated(self) -> None:
        factory = _factory()
        assert factory.resolve_chemical_embedding("MorganFingerprint") == "MorganFingerprint"
        with pytest.raises(ValueError):
            factory.resolve_chemical_embedding("NAICSEmbeddings")


class TestCachesKeyedByEmbedding:
    def test_sources_differ_per_embedding(self) -> None:
        factory = _factory()
        seen: Dict[str, List[str]] = {}
        for emb in HITS:
            seen[emb] = _cas(_FactorySource(factory, False, emb))
        assert seen == HITS

    def test_chemical_cache_key_contains_embedding(self) -> None:
        factory = _factory()
        a, b = _FactorySource(factory, False, "MorganFingerprint"), _FactorySource(factory, False, "MolDeBERTa")
        _cas(a)
        _cas(b)
        keys: List[Tuple[Any, ...]] = list(a._cache) + list(b._cache)  # pyright: ignore[reportPrivateUsage]
        assert {k[1] for k in keys} == {"MorganFingerprint", "MolDeBERTa"}

    def test_retrieve_fused_reuses_source_per_embedding_only(self) -> None:
        factory = _factory()
        created: List[Tuple[bool, str]] = []
        real = _FactorySource

        def build(f: RetrieverFactory, inc: bool, emb: str) -> _FactorySource:
            created.append((inc, emb))
            return real(f, inc, emb)

        import src.rag_graph.retrieval.retriever_factory as module

        original = module._FactorySource
        module._FactorySource = build  # type: ignore[misc, assignment]
        try:
            factory.get_cross_retriever = lambda: MagicMock(reduction_records=lambda split: [])  # type: ignore[method-assign]
            for emb in ("MorganFingerprint", "MolDeBERTa", "MorganFingerprint", None):
                factory.retrieve_fused(chemical_smiles="CCO", include_candidates=False, chemical_embedding=emb)
        finally:
            module._FactorySource = original  # type: ignore[misc]
        assert created == [(False, "MorganFingerprint"), (False, "MolDeBERTa"), (False, "Hybrid")]

    def test_retrieved_nodes_follow_the_override(self) -> None:
        factory = _factory()
        factory.get_cross_retriever = lambda: MagicMock(reduction_records=lambda split: [])  # type: ignore[method-assign]
        got = {
            emb: [
                n["node"]["cas_number"]
                for n in factory.retrieve_fused("CCO", include_candidates=False, chemical_embedding=emb).retrieved_nodes
            ]
            for emb in HITS
        }
        assert got == HITS
