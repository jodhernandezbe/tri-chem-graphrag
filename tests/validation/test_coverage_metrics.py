# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the redesigned activity-coverage metric helpers.

These guard the core fix: the primary metric must reward relevance, not breadth,
so that uniformly random retrieval no longer outscores informed retrieval.
"""

import math

from src.validation.anchor_builder import ActivityProfiles
from src.validation.retrieval_metrics import (
    _code_precision,
    _coverage,
    _idf_weights,
    _rank_codes,
    _weighted_coverage,
)


def _profiles(index_global: dict) -> ActivityProfiles:
    return ActivityProfiles(
        index_global={k: frozenset(v) for k, v in index_global.items()},
        query_global={},
        index_sector={},
        query_sector={},
        code_to_family={},
    )


class TestIdfWeights:

    def test_rare_code_weighted_higher_than_common(self):
        # 'common' appears in every chemical; 'rare' in one.
        index = {
            "a": frozenset({"common", "rare"}),
            "b": frozenset({"common"}),
            "c": frozenset({"common"}),
            "d": frozenset({"common"}),
        }
        weights = _idf_weights(index)
        assert weights["rare"] > weights["common"]

    def test_weights_always_positive(self):
        index = {"a": frozenset({"x"}), "b": frozenset({"x"})}
        weights = _idf_weights(index)
        assert all(w > 0 for w in weights.values())

    def test_empty_index_yields_empty_weights(self):
        assert _idf_weights({}) == {}


class TestWeightedCoverage:

    def test_full_recovery_is_one(self):
        weights = {"x": 1.0, "y": 2.0}
        assert _weighted_coverage(frozenset({"x", "y"}), {"x", "y"}, weights, 1.0) == 1.0

    def test_empty_query_is_zero(self):
        assert _weighted_coverage(frozenset(), {"x"}, {"x": 1.0}, 1.0) == 0.0

    def test_rare_miss_costs_more_than_common_miss(self):
        weights = {"common": 1.0, "rare": 5.0}
        query = frozenset({"common", "rare"})
        cover_common_only = _weighted_coverage(query, {"common"}, weights, 1.0)
        cover_rare_only = _weighted_coverage(query, {"rare"}, weights, 1.0)
        # Recovering the rare code yields more weighted coverage than the common one.
        assert cover_rare_only > cover_common_only

    def test_unseen_query_code_uses_fallback_weight(self):
        # 'z' is not in weights → fallback applies to the denominator.
        weights = {"x": 1.0}
        cov = _weighted_coverage(frozenset({"x", "z"}), {"x"}, weights, fallback=4.0)
        assert math.isclose(cov, 1.0 / (1.0 + 4.0))


class TestCodePrecision:

    def test_all_relevant_is_one(self):
        assert _code_precision(frozenset({"x", "y"}), {"x", "y"}) == 1.0

    def test_dumping_irrelevant_codes_is_penalized(self):
        # Retrieving 1 relevant + 3 irrelevant → precision 0.25.
        assert _code_precision(frozenset({"x"}), {"x", "a", "b", "c"}) == 0.25

    def test_empty_retrieval_is_zero(self):
        assert _code_precision(frozenset({"x"}), set()) == 0.0


class TestRankCodes:

    def test_codes_emitted_in_neighbour_order_deduplicated(self):
        profiles = _profiles({"a": {"S02", "S01"}, "b": {"S01", "S03"}, "c": {"S04"}})
        # Neighbour order a, b, c → within-chemical sorted, first occurrence wins.
        assert _rank_codes(["a", "b", "c"], profiles) == ["S01", "S02", "S03", "S04"]

    def test_unknown_chemical_contributes_nothing(self):
        profiles = _profiles({"a": {"S01"}})
        assert _rank_codes(["zzz", "a"], profiles) == ["S01"]

    def test_top_k_slice_reflects_ranking(self):
        profiles = _profiles({"a": {"S01"}, "b": {"S02"}, "c": {"S03"}})
        ranked = _rank_codes(["a", "b", "c"], profiles)
        assert ranked[:2] == ["S01", "S02"]


class TestPopularityFloor:
    """The redesigned metric must not reward a broad, indiscriminate retrieval."""

    def test_broad_retrieval_loses_on_precision(self):
        query = frozenset({"rare"})
        index = {f"c{i}": frozenset({"common"}) for i in range(9)}
        index["c9"] = frozenset({"common", "rare"})
        weights = _idf_weights(index)
        # A focused retrieval that returns exactly the right code.
        focused = {"rare"}
        # A broad retrieval that unions many chemicals' codes (popularity-style).
        broad = {"common", "rare", "x", "y", "z"}
        # Raw union coverage cannot distinguish them (both recover 'rare').
        assert _coverage(query, focused) == _coverage(query, broad) == 1.0
        # Precision does: focused wins decisively.
        assert _code_precision(query, focused) > _code_precision(query, broad)
        # Weighted coverage ties at full recovery, but precision breaks the popularity tie.
        assert _weighted_coverage(query, focused, weights, 1.0) == 1.0
