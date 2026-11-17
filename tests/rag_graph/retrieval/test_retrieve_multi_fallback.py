# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the sector-fallback flag and the backward-compatible retrieve_multi 2-tuple."""

from typing import Any, List
from unittest.mock import MagicMock

import pytest

from src.rag_graph.retrieval.retriever_factory import RetrieverFactory


def _doc(**meta: Any) -> MagicMock:
    doc = MagicMock()
    doc.metadata = meta
    doc.page_content = ""
    return doc


def _factory(cross_result: List[Any]) -> RetrieverFactory:
    factory = RetrieverFactory.__new__(RetrieverFactory)
    factory._top_k = 5
    factory._chem_embedding_type = "Hybrid"
    chem, sector = MagicMock(), MagicMock()
    chem.similarity_search_with_score.return_value = [(_doc(cas_number="1-1-1"), 0.9)]
    sector.similarity_search_with_score.return_value = [(_doc(naics_code="8123"), 0.8)]
    factory.get_vector_store = lambda t: sector if t == "NAICSEmbeddings" else chem  # type: ignore[method-assign]
    cross = MagicMock()
    cross.traverse_cross.return_value = cross_result
    factory.get_cross_retriever = lambda: cross  # type: ignore[method-assign]
    graph = MagicMock()
    graph.traverse_all.return_value = [{"node_type": "Chemical"}]
    factory.get_graph_retriever = lambda t: graph  # type: ignore[method-assign]
    return factory


@pytest.fixture(autouse=True)
def _bypass_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.rag_graph.retrieval.retriever_factory.guarded_search",
        lambda store, q, k: store.similarity_search_with_score(q, k=k),
    )


class TestRetrieveMultiFallback:

    def test_empty_intersection_sets_flag(self):
        _, _, fallback = _factory([]).retrieve_multi_detailed("CCO", "Dry cleaning")
        assert fallback is True

    def test_non_empty_intersection_clears_flag(self):
        _, _, fallback = _factory([{"node_type": "Chemical_Sector"}]).retrieve_multi_detailed("CCO", "Dry cleaning")
        assert fallback is False

    def test_chemical_only_has_no_flag(self):
        _, _, fallback = _factory([]).retrieve_multi_detailed("CCO", "")
        assert fallback is False

    def test_retrieve_multi_still_returns_two_tuple(self):
        result = _factory([]).retrieve_multi("CCO", "Dry cleaning")
        assert isinstance(result, tuple) and len(result) == 2
