# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the report-split leak parameter and the MostCommon baseline."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.rag_graph.retrieval import graph_retriever
from src.rag_graph.retrieval.graph_retriever import CrossEntityRetriever, GraphRetriever
from src.rag_graph.retrieval.switches import resolve_switch
from src.validation.anchor_builder import ActivityProfiles
from src.validation.recommendation_metrics import RECORD_CAP, RecommendationMetrics


def _metrics() -> RecommendationMetrics:
    metrics = RecommendationMetrics.__new__(RecommendationMetrics)
    metrics._factory = MagicMock()  # type: ignore[attr-defined]
    metrics._factory.retrieve_multi.return_value = ([], [])
    return metrics


def _profiles() -> ActivityProfiles:
    index = {
        "a": frozenset({"S1", "S2"}),
        "b": frozenset({"S1", "S3"}),
        "c": frozenset({"S1", "S2", "S4"}),
    }
    return ActivityProfiles(
        index_global=index,
        query_global={"q": frozenset({"S1", "S2"})},
        index_sector={},
        query_sector={},
        code_to_family={},
    )


class TestEvaluatorHidesWithheldReports:

    def test_legacy_row_expands_through_candidates_like_the_deployed_pipeline(self):
        metrics = _metrics()
        metrics._fetch_test_chemicals = lambda: [{"cas_number": "q", "smiles": "CCO"}]  # type: ignore[method-assign]
        metrics._fused_codes = lambda smiles: ["S1"]  # type: ignore[method-assign]
        metrics.evaluate(_profiles())
        retrieve: Any = metrics._factory.retrieve_multi  # type: ignore[attr-defined]
        assert retrieve.call_args.kwargs["include_candidates"] is True

    def test_recommended_codes_requests_index_split(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr("src.validation.recommendation_metrics.tabulate_records", lambda n, s, limit: [])
        metrics = _metrics()
        metrics._recommended_codes("CCO")
        assert metrics._factory.retrieve_multi.call_args.kwargs["report_split"] == "index"


class TestTraversalParameter:

    def _retriever(self, label: str = "Chemical") -> GraphRetriever:
        retriever = GraphRetriever.__new__(GraphRetriever)
        retriever.node_label, retriever.limit = label, 5
        retriever._graph = MagicMock()
        retriever._graph.query.return_value = []
        return retriever

    def test_default_passes_none(self):
        retriever = self._retriever()
        retriever.traverse({"cas_number": "1-1-1"})
        assert retriever._graph.query.call_args.args[1]["report_split"] is None

    def test_split_is_passed_as_parameter(self):
        retriever = self._retriever()
        retriever.traverse_all([{"cas_number": "1-1-1"}], report_split="index")
        assert retriever._graph.query.call_args.args[1]["report_split"] == "index"

    def test_cross_traversal_passes_split(self):
        cross = CrossEntityRetriever.__new__(CrossEntityRetriever)
        cross.limit = 5
        cross._graph = MagicMock()
        cross._graph.query.return_value = []
        cross.traverse_cross(["1-1-1"], ["8123"], report_split="index")
        assert cross._graph.query.call_args.args[1]["report_split"] == "index"

    @pytest.mark.parametrize("name", ["CHEMICAL_TRAVERSAL", "SECTOR_TRAVERSAL", "CROSS_ENTITY_TRAVERSAL"])
    def test_cypher_is_parameterised_and_null_safe(self, name: str):
        query = getattr(graph_retriever, name)
        assert "$report_split IS NULL OR r.split = $report_split" in query
        assert "'index'" not in query


class TestMostCommonBaseline:

    def test_ranks_by_index_frequency(self):
        result = _metrics()._most_common_baseline({"q": frozenset({"S1", "S2"})}, _profiles(), {"S1": 1.0, "S2": 1.0}, 1.0)
        assert result.method == "MostCommon"
        assert result.n_queries == 1
        assert result.raw_coverage_at_k[5] == 1.0
        assert result.code_precision_at_k[10] == 0.5

    def test_top_one_would_be_most_frequent_code(self):
        queries: Dict[str, Any] = {"q": frozenset({"S4"})}
        result = _metrics()._most_common_baseline(queries, _profiles(), {"S4": 2.0}, 2.0)
        # k=5 and above contain all four codes, so the rare S4 is covered
        assert result.raw_coverage_at_k[5] == 1.0

    def test_evaluate_includes_baseline_on_same_queries(self):
        metrics = _metrics()
        metrics._fetch_test_chemicals = lambda: [{"cas_number": "q", "smiles": "CCO"}]  # type: ignore[method-assign]
        metrics._recommended_codes = lambda smiles, **kwargs: ["S3"]  # type: ignore[method-assign]
        metrics._fused_codes = lambda smiles: ["S1"]  # type: ignore[method-assign]
        names: List[str] = [r.method for r in metrics.evaluate(_profiles())]
        assert names == ["Shipped (fused)", "Legacy", "MostCommon", "Random", "Oracle"]


class TestShippedDefaultAndLegacyRow:

    def _evaluated(self, **kwargs: Any) -> Dict[str, Any]:
        metrics = _metrics()
        metrics._fetch_test_chemicals = lambda: [{"cas_number": "q", "smiles": "CCO"}]  # type: ignore[method-assign]
        metrics._recommended_codes = lambda smiles, **kwargs: ["S3"]  # type: ignore[method-assign]
        metrics._fused_codes = lambda smiles: ["S1", "S2"]  # type: ignore[method-assign]
        return {r.method: r for r in metrics.evaluate(_profiles(), **kwargs)}

    def test_config_ships_fused_as_default(self):
        assert resolve_switch("retrieval") == "fused"

    def test_default_scores_fused_and_legacy_separately(self):
        rows = self._evaluated()
        assert rows["Shipped (fused)"].raw_coverage_at_k[5] == 1.0
        assert rows["Legacy"].raw_coverage_at_k[5] == 0.0

    def test_legacy_default_has_no_duplicate_legacy_row(self):
        assert [m for m in self._evaluated(retrieval="legacy") if m.startswith(("Shipped", "Legacy"))] == ["Shipped (legacy)"]

    def test_ranking_override_keeps_a_single_legacy_variant_row(self):
        metrics = _metrics()
        metrics._fetch_test_chemicals = lambda: [{"cas_number": "q", "smiles": "CCO"}]  # type: ignore[method-assign]
        metrics._recommended_codes = lambda smiles, cas, ranking, analogues, idf: ["S1"]  # type: ignore[method-assign]
        names = [r.method for r in metrics.evaluate(_profiles(), ranking="specificity")]
        assert names[0] == "ShippedRecommender" and "Legacy" not in names

    def test_fused_row_uses_index_split_and_shared_cap(self):
        metrics = _metrics()
        metrics._factory.retrieve_fused.return_value = MagicMock(records=[{"ReductionCode": "S1"}, {"ReductionCode": "S1"}])
        assert metrics._fused_codes("CCO") == ["S1"]
        kwargs = metrics._factory.retrieve_fused.call_args.kwargs
        assert kwargs["report_split"] == "index" and kwargs["limit"] == RECORD_CAP and "condition_of_use" not in kwargs
