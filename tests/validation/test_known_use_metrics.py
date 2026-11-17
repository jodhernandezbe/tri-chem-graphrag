# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the known-use test: query construction, per-chemical averaging and the four methods."""

from typing import Any, Dict, FrozenSet, List
from unittest.mock import MagicMock

from src.validation.anchor_builder import ActivityProfiles
from src.validation.known_use_metrics import (
    KnownUseMetrics,
    KnownUseQuery,
    build_queries,
    by_chemical,
    paired_p,
    query_scores,
)
from src.validation.recommendation_metrics import RECORD_CAP

WEIGHTS: Dict[str, float] = {"S1": 1.0, "S2": 1.0, "S3": 1.0}


def _profiles() -> ActivityProfiles:
    index = {"a": frozenset({"S1", "S2"}), "b": frozenset({"S1", "S3"}), "c": frozenset({"S1"})}
    return ActivityProfiles(index, {}, {}, {}, {})


def _query(cas: str, use: str, truth: FrozenSet[str]) -> KnownUseQuery:
    return KnownUseQuery(cas, "CCO", use, truth)


def _metrics(queries: List[KnownUseQuery]) -> KnownUseMetrics:
    metrics = KnownUseMetrics.__new__(KnownUseMetrics)
    metrics._factory = MagicMock()  # type: ignore[attr-defined]
    metrics._fetch_queries = lambda: queries  # type: ignore[method-assign]
    return metrics


class TestScoring:

    def test_query_scores_cover_every_k(self):
        scores = query_scores(frozenset({"S1"}), ["S1", "S2"], WEIGHTS, 1.0)
        assert set(scores) == {5, 10, 20} and scores[5]["raw"] == 1.0 and scores[5]["prec"] == 0.5

    def test_queries_are_averaged_within_a_chemical(self):
        queries = [
            _query("a", "u1", frozenset({"S1"})),
            _query("a", "u2", frozenset({"S2"})),
            _query("b", "u1", frozenset({"S1"})),
        ]
        per_query = [query_scores(q.truth, ["S1"], WEIGHTS, 1.0) for q in queries]
        chem = by_chemical(queries, per_query)
        assert [c[10]["raw"] for c in chem] == [0.5, 1.0]

    def test_paired_p_is_one_when_identical(self):
        assert paired_p([0.1, 0.2], [0.1, 0.2]) == 1.0


class TestBuildQueries:

    def test_two_reports_with_different_uses_give_different_truth(self):
        rows = [
            {"cas": "a", "smiles": "CCO", "codes": ["S1"], "acts": ["Use A"]},
            {"cas": "a", "smiles": "CCO", "codes": ["S2"], "acts": ["Use B"]},
        ]
        truth = {q.use: q.truth for q in build_queries(rows)}
        assert truth == {"Use A": frozenset({"S1"}), "Use B": frozenset({"S2"})}

    def test_reports_sharing_a_use_are_unioned(self):
        rows = [
            {"cas": "a", "smiles": "CCO", "codes": ["S1"], "acts": ["Use A", "Use B"]},
            {"cas": "a", "smiles": "CCO", "codes": ["S2"], "acts": ["Use A"]},
        ]
        truth = {q.use: q.truth for q in build_queries(rows)}
        assert truth == {"Use A": frozenset({"S1", "S2"}), "Use B": frozenset({"S1"})}


class TestEvaluate:

    def test_four_methods_with_shared_cap_and_use_passed_through(self):
        queries = [_query("a", "Use A", frozenset({"S1"})), _query("b", "Use B", frozenset({"S3"}))]
        metrics = _metrics(queries)
        fused: Any = metrics._factory.retrieve_fused  # type: ignore[attr-defined]
        fused.return_value = MagicMock(records=[{"ReductionCode": "S1"}, {"ReductionCode": "S3"}])
        metrics._factory.retrieve_multi_result.return_value = MagicMock(retrieved_nodes=[], subgraphs=[])  # type: ignore[attr-defined]
        results = {r.method: r for r in metrics.evaluate(_profiles())}
        assert list(results) == ["Fused", "Legacy", "MostCommon", "Random"]
        assert results["Fused"].raw_coverage_at_k[5] == 1.0 and results["Legacy"].raw_coverage_at_k[5] == 0.0
        assert results["Fused"].p_vs_fused is None and results["Legacy"].p_vs_fused is not None
        assert all(c.kwargs["report_split"] == "index" and c.kwargs["limit"] == RECORD_CAP for c in fused.call_args_list)
        assert {c.kwargs["condition_of_use"] for c in fused.call_args_list} == {"Use A", "Use B"}
        legacy: Any = metrics._factory.retrieve_multi_result  # type: ignore[attr-defined]
        assert all(c.kwargs["include_candidates"] is True for c in legacy.call_args_list)
