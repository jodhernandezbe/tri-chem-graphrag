# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Tests for activity families, reduction bands and the read-only probability and efficiency layer.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import copy
import importlib.util
import pathlib
from typing import Any, Dict, List, Optional

import pytest

from src.rag_graph.retrieval.activity_families import (
    CODE_TO_FAMILY,
    FAMILY_NAMES,
    band_of,
    family_of,
)
from src.rag_graph.retrieval.p2_estimates import estimate_p2

XWALK = pathlib.Path(__file__).resolve().parents[3] / "analysis" / "surrogate_and_p2_likelihood" / "xwalk.py"


def _row(code: str, name: str, score: float, reduction: Optional[float]) -> Dict[str, Any]:
    return {"ReductionCode": code, "Reduction Activity": name, "FusedScore": score, "AnnualReduction": reduction}


ROWS: List[Dict[str, Any]] = [
    _row("W58", "OTHER PROCESS MODIFICATIONS", 0.4, 100.0),
    _row("W42", "SUBSTITUTED RAW MATERIALS", 0.3, 50.0),
    _row("W42", "SUBSTITUTED RAW MATERIALS", 0.1, None),
    _row("S11", "PRODUCT REFORMULATION", 0.2, 25.0),
]


class TestFamiliesAndBands:
    def test_both_series_map_to_families(self) -> None:
        assert (family_of("W58"), family_of("S23"), family_of("s11"), family_of("X99")) == ("F3", "F3", "F2", "F0")

    @pytest.mark.skipif(not XWALK.exists(), reason="analysis crosswalk not present")
    def test_analysis_crosswalk_reexports_the_src_map(self) -> None:
        spec = importlib.util.spec_from_file_location("xwalk_copy", XWALK)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.C2F == CODE_TO_FAMILY and module.FAM == FAMILY_NAMES
        assert len(CODE_TO_FAMILY) == 73 and set(CODE_TO_FAMILY.values()) == set(FAMILY_NAMES)

    @pytest.mark.parametrize(
        "percent,band",
        [(100.0, "R1"), (75.0, "R2"), (50.0, "R2"), (25.0, "R3"), (10.0, "below"), (None, None), (float("nan"), None)],
    )
    def test_band_of(self, percent: Optional[float], band: Optional[str]) -> None:
        assert band_of(percent) == band


class TestEstimate:
    def test_probabilities_sum_to_one_and_follow_score_mass(self) -> None:
        est = estimate_p2(ROWS)
        assert sum(g.probability for g in est.codes) == pytest.approx(1.0)
        assert sum(g.probability for g in est.families) == pytest.approx(1.0)
        assert [g.key for g in est.codes] == ["W42", "W58", "S11"]
        w42 = next(g for g in est.codes if g.key == "W42")
        assert w42.probability == pytest.approx(0.4) and w42.n_records == 2 and w42.n_with_band == 1

    def test_families_aggregate_codes_and_flag_catch_all(self) -> None:
        est = estimate_p2(ROWS)
        f3 = next(g for g in est.families if g.key == "F3")
        assert f3.probability == pytest.approx(0.4)
        assert next(g for g in est.codes if g.key == "W58").catch_all
        assert not next(g for g in est.codes if g.key == "W42").catch_all

    def test_band_distribution_ignores_missing_reduction(self) -> None:
        w42 = next(g for g in estimate_p2(ROWS).codes if g.key == "W42")
        assert w42.band_probabilities == {"R1": 0.0, "R2": 1.0, "R3": 0.0}

    def test_rows_are_not_modified_or_reordered(self) -> None:
        before = copy.deepcopy(ROWS)
        estimate_p2(ROWS)
        assert ROWS == before

    def test_limit_uses_leading_rows_only(self) -> None:
        est = estimate_p2(ROWS, limit=2)
        assert est.n_records == 2 and {g.key for g in est.codes} == {"W58", "W42"}

    def test_zero_scores_fall_back_to_equal_weights(self) -> None:
        rows = [_row("W58", "A", 0.0, 50.0), _row("W42", "B", 0.0, 50.0)]
        assert [g.probability for g in estimate_p2(rows).codes] == [0.5, 0.5]

    def test_empty_input(self) -> None:
        assert estimate_p2([]).codes == [] and estimate_p2([]).n_records == 0
