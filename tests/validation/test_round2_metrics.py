# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Tests for the round 2 metric definitions (grouping, tie rules, "not in list", sign test, chance level, freeze check).

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import json
from pathlib import Path

import pytest

from src.validation.round2_metrics import (
    LOSS,
    TIE,
    WIN,
    binomial_upper_p,
    chance_p0,
    check_freeze,
    compare_ranks,
    count_outcomes,
    exact_sign_p,
    grouped_order,
    mostcommon_order,
    paper_row,
    rank_of,
    sha256_file,
    summarize,
    tool_order,
)

NAMES = {
    "W58": "OTHER PROCESS MODIFICATIONS",
    "S25": "other process modifications made",
    "W42": "SUBSTITUTED RAW MATERIALS",
    "S11": "REFORMULATED",
}
UNIVERSE = ["W58", "S25", "W42", "S11"]


class TestLists:
    def test_tool_order_drops_zero_probability_and_breaks_ties_by_code(self) -> None:
        assert tool_order({"W42": 0.3, "S11": 0.3, "W58": 0.4, "S25": 0.0}, UNIVERSE) == ["W58", "S11", "W42"]

    def test_mostcommon_ties_by_ascending_code_string(self) -> None:
        assert mostcommon_order({"W42": 5, "S11": 5, "W58": 9}, UNIVERSE) == ["W58", "S11", "W42", "S25"]

    def test_catch_alls_move_after_specific_codes_case_insensitively(self) -> None:
        assert grouped_order(["W58", "W42", "S25", "S11"], NAMES) == ["W42", "S11", "W58", "S25"]

    def test_rank_of(self) -> None:
        assert rank_of(["a", "b"], "b") == 2 and rank_of(["a"], "z") is None


class TestCompare:
    @pytest.mark.parametrize(
        "tool,mc,expected",
        [
            (2, 5, WIN),
            (5, 2, LOSS),
            (3, 3, TIE),
            (None, 7, LOSS),
            (None, None, TIE),
            (None, 51, TIE),
            (4, None, WIN),
            (4, 60, WIN),
        ],
    )
    def test_outcomes(self, tool: int, mc: int, expected: str) -> None:
        assert compare_ranks(tool, mc) == expected

    def test_not_in_list_is_a_loss_only_when_mostcommon_has_it_within_the_budget(self) -> None:
        assert compare_ranks(None, 50) == LOSS and compare_ranks(None, 51) == TIE


class TestTests:
    def test_sign_p_matches_preregistered_values(self) -> None:
        assert exact_sign_p(9, 3) == pytest.approx(0.0730, abs=1e-3)
        assert exact_sign_p(8, 4) == pytest.approx(0.1938, abs=1e-3)
        assert exact_sign_p(11, 4) == pytest.approx(0.0592, abs=1e-3)
        assert exact_sign_p(0, 0) is None

    def test_binomial_and_chance(self) -> None:
        assert binomial_upper_p(0, 5, 0.3) == pytest.approx(1.0)
        assert binomial_upper_p(5, 5, 0.5) == pytest.approx(1 / 32)
        assert chance_p0(73) == pytest.approx(10 / 73)

    def test_count_outcomes(self) -> None:
        assert count_outcomes([WIN, LOSS, WIN, TIE]) == {"wins": 2, "losses": 1, "ties": 1}


class TestPaperRowAndSummary:
    def test_row_uses_grouped_lists_and_records_not_in_list(self) -> None:
        row = paper_row("W42", {"W58": 0.6, "W42": 0.3, "S11": 0.1}, UNIVERSE, {"W58": 9, "W42": 5, "S11": 1}, NAMES)
        assert row["tool_rank"] == 1 and row["mostcommon_rank"] == 1 and row["outcome"] == TIE
        missing = paper_row("S25", {"W58": 1.0}, UNIVERSE, {"W58": 9, "S25": 2}, NAMES)
        assert missing["tool_rank"] is None and missing["outcome"] == LOSS

    def test_summary(self) -> None:
        rows = [
            {"outcome": WIN, "tool_hit_top10": True, "mostcommon_hit_top10": False},
            {"outcome": LOSS, "tool_hit_top10": False, "mostcommon_hit_top10": False},
        ]
        out = summarize(rows, 73)
        assert (
            out["wins"] == 1 and out["losses"] == 1 and out["sign_p_one_sided"] == pytest.approx(0.75) and out["n_codes"] == 73
        )
        assert out["tool_hits_top10"] == 1 and out["chance_p0"] == pytest.approx(10 / 73)


class TestFreeze:
    def test_requires_matching_hash(self, tmp_path: Path) -> None:
        papers = tmp_path / "papers.json"
        papers.write_text("[]")
        manifest = tmp_path / "freeze_manifest.json"
        manifest.write_text(json.dumps({"papers_sha256": sha256_file(str(papers))}))
        assert check_freeze(str(manifest), str(papers))[1] == sha256_file(str(papers))
        manifest.write_text(json.dumps({"papers_sha256": "0" * 64}))
        with pytest.raises(ValueError):
            check_freeze(str(manifest), str(papers))

    def test_missing_manifest_is_refused(self, tmp_path: Path) -> None:
        papers = tmp_path / "papers.json"
        papers.write_text("[]")
        with pytest.raises(FileNotFoundError):
            check_freeze(str(tmp_path / "nope.json"), str(papers))
