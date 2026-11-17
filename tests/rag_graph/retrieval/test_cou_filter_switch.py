# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the ``cou_filter`` switch (hard vs soft) and its deterministic answer notice."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.rag_graph.generation.rag_pipeline import COU_FALLBACK_NOTICE, RAGPipeline
from src.rag_graph.retrieval.retriever_factory import RetrieverFactory, has_records
from src.rag_graph.retrieval.switches import resolve_switch

WITH_RECORDS: List[Dict[str, Any]] = [{"node_type": "Chemical", "reports": [{"reduction_code": "S01"}]}]
WITHOUT_RECORDS: List[Dict[str, Any]] = [{"node_type": "Chemical", "reports": []}]


def _doc(**meta: Any) -> MagicMock:
    doc = MagicMock()
    doc.metadata = meta
    doc.page_content = ""
    return doc


def _factory(filtered: List[Dict[str, Any]], unfiltered: List[Dict[str, Any]]) -> RetrieverFactory:
    """Chemical-only factory whose traversal returns ``filtered`` with activity names and ``unfiltered`` without."""
    factory = RetrieverFactory.__new__(RetrieverFactory)
    factory._top_k = 5
    factory._chem_embedding_type = "Hybrid"
    chem = MagicMock()
    chem.similarity_search_with_score.return_value = [(_doc(cas_number="1-1-1"), 0.9)]
    factory.get_vector_store = lambda t: chem  # type: ignore[method-assign]
    factory._retrieve_activity_names = lambda cou: ["used_as_a_reactant"] if cou else []  # type: ignore[method-assign]
    graph = MagicMock()
    graph.traverse_all.side_effect = lambda nodes, names, split: filtered if names else unfiltered
    factory.get_graph_retriever = lambda t: graph  # type: ignore[method-assign]
    factory._graph_mock = graph  # type: ignore[attr-defined]
    return factory


@pytest.fixture(autouse=True)
def _bypass_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "src.rag_graph.retrieval.retriever_factory.guarded_search",
        lambda store, q, k: store.similarity_search_with_score(q, k=k),
    )


class TestSwitchResolution:

    def test_config_default_is_hard(self):
        assert resolve_switch("cou_filter") == "hard"

    def test_override_wins(self):
        assert resolve_switch("cou_filter", "soft") == "soft"

    def test_invalid_value_rejected(self):
        with pytest.raises(ValueError):
            resolve_switch("cou_filter", "medium")


class TestHasRecords:

    def test_detects_reports(self):
        assert has_records(WITH_RECORDS) and not has_records(WITHOUT_RECORDS) and not has_records([])


class TestHardFilter:

    def test_hard_keeps_empty_result_and_does_not_retry(self):
        factory = _factory(WITHOUT_RECORDS, WITH_RECORDS)
        res = factory.retrieve_multi_result("CCO", condition_of_use="zinc oxide use", cou_filter="hard")
        assert res.subgraphs == WITHOUT_RECORDS and res.cou_fallback is False
        assert factory._graph_mock.traverse_all.call_count == 1  # type: ignore[attr-defined]

    def test_default_is_hard(self):
        res = _factory(WITHOUT_RECORDS, WITH_RECORDS).retrieve_multi_result("CCO", condition_of_use="x")
        assert res.cou_fallback is False and res.subgraphs == WITHOUT_RECORDS


class TestSoftFilter:

    def test_soft_drops_filter_when_no_records(self):
        factory = _factory(WITHOUT_RECORDS, WITH_RECORDS)
        res = factory.retrieve_multi_result("CCO", condition_of_use="zinc oxide use", cou_filter="soft")
        assert res.subgraphs == WITH_RECORDS and res.cou_fallback is True
        assert factory._graph_mock.traverse_all.call_args.args[1] == []  # type: ignore[attr-defined]

    def test_soft_keeps_filtered_result_when_it_has_records(self):
        res = _factory(WITH_RECORDS, WITHOUT_RECORDS).retrieve_multi_result("CCO", condition_of_use="x", cou_filter="soft")
        assert res.subgraphs == WITH_RECORDS and res.cou_fallback is False

    def test_soft_without_condition_of_use_never_falls_back(self):
        res = _factory(WITHOUT_RECORDS, WITH_RECORDS).retrieve_multi_result("CCO", cou_filter="soft")
        assert res.cou_fallback is False

    def test_two_tuple_wrappers_still_work(self):
        factory = _factory(WITHOUT_RECORDS, WITH_RECORDS)
        nodes, subgraphs = factory.retrieve_multi("CCO", condition_of_use="x", cou_filter="soft")
        assert subgraphs == WITH_RECORDS and len(nodes) == 1
        assert len(factory.retrieve_multi_detailed("CCO", condition_of_use="x", cou_filter="soft")) == 3


def _generate(cou_fallback: bool, sector_fallback: bool = False) -> str:
    state: Dict[str, Any] = {
        "query": "q",
        "records": [{"Chemical": "x"}],
        "context": "ctx",
        "condition_of_use": "plasma etching",
        "cou_fallback": cou_fallback,
        "sector_query": "Dry cleaning",
        "sector_fallback": sector_fallback,
    }
    with patch("src.rag_graph.generation.rag_pipeline.ChatOllama") as chat:
        chat.return_value.invoke.return_value = MagicMock(content="LLM answer")
        return str(RAGPipeline()._generate(state)["answer"])  # type: ignore[arg-type]


class TestCouFallbackNotice:

    def test_notice_prepended(self):
        answer = _generate(True)
        assert answer == COU_FALLBACK_NOTICE.format(cou="plasma etching") + "\n\nLLM answer"
        assert "not restricted by condition of use" in answer

    def test_no_notice_by_default(self):
        assert _generate(False) == "LLM answer"

    def test_both_notices_stack(self):
        answer = _generate(True, sector_fallback=True)
        assert "stated sector" in answer and "stated condition of use" in answer and answer.endswith("LLM answer")
