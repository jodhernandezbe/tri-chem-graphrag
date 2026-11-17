# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for weighted reciprocal rank fusion: the formula, defaults, validation and the retriever's use of weights."""

from typing import Dict, List
from unittest.mock import patch

import pytest

from src.rag_graph.retrieval.fused_retriever import (
    F1,
    F2,
    FusedOptions,
    FusedRetriever,
    reciprocal_rank_fusion,
)
from src.rag_graph.retrieval.switches import (
    parse_channel_weights,
    resolve_channel_weights,
)
from tests.rag_graph.retrieval.test_fused_retriever import _source

CHANNELS = {"a": {0: 2.0, 1: 1.0}, "b": {1: 5.0}}


def _codes(options: FusedOptions) -> List[str]:
    return [
        r["ReductionCode"]
        for r in FusedRetriever(_source()).retrieve("C", "sector text", "solvent use", options=options, limit=50).records
    ]


class TestWeightedFormula:
    def test_weighted_sum_of_reciprocal_ranks(self) -> None:
        fused = reciprocal_rank_fusion(CHANNELS, weights={"a": 1.0, "b": 0.25})
        assert fused[0] == pytest.approx(1 / 61)
        assert fused[1] == pytest.approx(1 / 62 + 0.25 / 61)

    def test_default_weights_equal_explicit_ones(self) -> None:
        explicit = reciprocal_rank_fusion(CHANNELS, weights={"a": 1.0, "b": 1.0})
        assert reciprocal_rank_fusion(CHANNELS) == explicit
        assert reciprocal_rank_fusion(CHANNELS, weights={}) == explicit

    def test_zero_weight_skips_the_channel(self) -> None:
        assert reciprocal_rank_fusion(CHANNELS, weights={"b": 0.0}) == reciprocal_rank_fusion({"a": CHANNELS["a"]})

    def test_weight_can_reverse_the_order(self) -> None:
        channels = {"a": {0: 2.0, 1: 1.0}, "b": {1: 2.0, 0: 1.0}}
        assert (
            reciprocal_rank_fusion(channels, weights={"a": 1.0, "b": 0.1})[0]
            > reciprocal_rank_fusion(channels, weights={"a": 1.0, "b": 0.1})[1]
        )


class TestWeightParsing:
    def test_empty_means_all_ones(self) -> None:
        assert parse_channel_weights(None) == {"chemical": 1.0, "use": 1.0, "sector": 1.0, "prior": 1.0}

    def test_partial_fills_the_rest(self) -> None:
        assert parse_channel_weights({"prior": 0.25})["use"] == 1.0

    @pytest.mark.parametrize("bad", [{"nope": 1.0}, {"prior": -0.1}, {"prior": "x"}, {"prior": float("nan")}])
    def test_invalid_is_rejected(self, bad: Dict[str, object]) -> None:
        with pytest.raises(ValueError):
            parse_channel_weights(bad)  # type: ignore[arg-type]

    def test_override_beats_config(self) -> None:
        assert resolve_channel_weights({"prior": 0.5})["prior"] == 0.5

    def test_config_is_read_when_no_override(self) -> None:
        cfg = {"RAG_GRAPH": {"RETRIEVAL": {"RETRIEVAL_WEIGHTS": {"prior": 0.1}}}}
        with patch("src.rag_graph.retrieval.switches.ConfigManager") as manager:
            manager.return_value.config = cfg
            assert resolve_channel_weights()["prior"] == 0.1


class TestRetrieverWeights:
    def test_default_is_unchanged_by_explicit_ones(self) -> None:
        ones = FusedOptions(weights={"chemical": 1.0, "use": 1.0, "sector": 1.0, "prior": 1.0})
        assert _codes(ones) == _codes(F1)

    def test_zero_prior_weight_equals_no_prior(self) -> None:
        assert _codes(FusedOptions(weights={"prior": 0.0})) == _codes(F2)

    def test_zero_prior_weight_removes_the_channel_label(self) -> None:
        rows = FusedRetriever(_source()).retrieve("C", "s", "u", options=FusedOptions(weights={"prior": 0.0}), limit=50).records
        assert all("prior" not in r["Channels"] for r in rows)
