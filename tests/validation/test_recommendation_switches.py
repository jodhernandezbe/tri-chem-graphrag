# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the switch arguments of the recommendation evaluator."""

from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from src.validation.anchor_builder import ActivityProfiles
from src.validation.recommendation_metrics import RECORD_CAP, RecommendationMetrics


def _metrics() -> RecommendationMetrics:
    metrics = RecommendationMetrics.__new__(RecommendationMetrics)
    metrics._factory = MagicMock()  # type: ignore[attr-defined]
    metrics._factory.retrieve_multi.return_value = ([], [])
    return metrics


class TestRecommendedCodesSwitches:

    def test_defaults_add_no_extra_arguments(self, monkeypatch: pytest.MonkeyPatch):
        seen: Dict[str, Any] = {}
        monkeypatch.setattr("src.validation.recommendation_metrics.tabulate_records", lambda n, s, **kw: seen.update(kw) or [])
        metrics = _metrics()
        retrieve: Any = metrics._factory.retrieve_multi  # type: ignore[attr-defined]
        metrics._recommended_codes("CCO")
        assert seen == {"limit": RECORD_CAP}
        assert set(retrieve.call_args.kwargs) == {"chemical_smiles", "sector_query", "report_split", "include_candidates"}
        assert retrieve.call_args.kwargs["include_candidates"] is False

    def test_switches_reach_retriever_and_tabulator(self, monkeypatch: pytest.MonkeyPatch):
        seen: Dict[str, Any] = {}
        monkeypatch.setattr("src.validation.recommendation_metrics.tabulate_records", lambda n, s, **kw: seen.update(kw) or [])
        metrics = _metrics()
        retrieve: Any = metrics._factory.retrieve_multi  # type: ignore[attr-defined]
        metrics._recommended_codes("CCO", "1-1-1", "consensus_specificity", "context", {"S1": 1.0})
        kwargs = retrieve.call_args.kwargs
        assert kwargs["report_split"] == "index" and kwargs["analogues"] == "context" and kwargs["query_cas"] == "1-1-1"
        assert seen["ranking"] == "consensus_specificity" and seen["code_idf"] == {"S1": 1.0}

    def test_evaluate_builds_index_split_idf_for_non_default_ranking(self):
        metrics = _metrics()
        profiles = ActivityProfiles(
            index_global={"a": frozenset({"S1"}), "b": frozenset({"S1", "S2"})},
            query_global={"q": frozenset({"S2"})},
            index_sector={},
            query_sector={},
            code_to_family={},
        )
        metrics._fetch_test_chemicals = lambda: [{"cas_number": "q", "smiles": "CCO"}]  # type: ignore[method-assign]
        captured: Dict[str, Any] = {}

        def fake(smiles: str, cas: str, ranking: str, analogues: Any, idf: Dict[str, float]) -> list:
            captured.update(ranking=ranking, idf=idf)
            return ["S2"]

        metrics._recommended_codes = fake  # type: ignore[method-assign]
        metrics.evaluate(profiles, ranking="specificity")
        assert captured["ranking"] == "specificity" and captured["idf"]["S1"] == 0.0 and captured["idf"]["S2"] > 0
