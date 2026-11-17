# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Tests for catch-all grouping in the fused context and markdown table, and for the shipped prior weight.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

from typing import Any, Dict, List

from src.rag_graph.generation.rag_pipeline import (
    records_table_lines,
    records_to_context,
)
from src.rag_graph.retrieval.fused_retriever import CATCH_ALL_HEADING, split_catch_all
from src.rag_graph.retrieval.switches import resolve_channel_weights


def _row(name: str, tier: int = 1) -> Dict[str, Any]:
    return {
        "Chemical": "Benzene",
        "CAS": "71-43-2",
        "Similarity": 1.0,
        "Sector": "325 — Chemicals",
        "Reduction Activity": name,
        "Year": 2020,
        "State": "TX",
        "Tier": tier,
        "Channels": ["chemical"],
        "Uses": [],
    }


ROWS: List[Dict[str, Any]] = [
    _row("OTHER process modifications"),
    _row("Substituted solvent"),
    _row("other housekeeping"),
    _row("Improved seals"),
]


class TestSplitCatchAll:
    def test_groups_keep_order_and_original_rank(self) -> None:
        specific, catch_all = split_catch_all(ROWS)
        assert [(i, r["Reduction Activity"]) for i, r in specific] == [(2, "Substituted solvent"), (4, "Improved seals")]
        assert [i for i, _ in catch_all] == [1, 3]

    def test_prefix_is_case_insensitive_and_input_untouched(self) -> None:
        before = [r["Reduction Activity"] for r in ROWS]
        split_catch_all(ROWS)
        assert [r["Reduction Activity"] for r in ROWS] == before

    def test_no_catch_all(self) -> None:
        assert split_catch_all(ROWS[1:2])[1] == []


class TestPresentation:
    def test_context_lists_specific_first_then_heading(self) -> None:
        context = records_to_context(ROWS, "C1=CC=CC=C1")
        assert (
            context.index("Substituted solvent")
            < context.index(CATCH_ALL_HEADING)
            < context.index("OTHER process modifications")
        )
        assert "Rank 1 | OTHER process modifications" in context
        assert "Rank 4 | Improved seals" in context

    def test_table_has_heading_and_rank_column(self) -> None:
        text = "\n".join(records_table_lines(ROWS))
        assert text.index("Improved seals") < text.index(CATCH_ALL_HEADING) < text.index("OTHER process modifications")
        assert text.count("| Tier | Channels | Rank |") == 2

    def test_no_heading_without_catch_all(self) -> None:
        assert CATCH_ALL_HEADING not in "\n".join(records_table_lines(ROWS[1:2]))
        assert CATCH_ALL_HEADING not in records_to_context(ROWS[1:2])

    def test_record_list_order_is_unchanged(self) -> None:
        names = [r["Reduction Activity"] for r in ROWS]
        records_table_lines(ROWS)
        records_to_context(ROWS)
        assert [r["Reduction Activity"] for r in ROWS] == names


class TestShippedWeights:
    def test_prior_is_half_and_others_are_one(self) -> None:
        assert resolve_channel_weights() == {"chemical": 1.0, "use": 1.0, "sector": 1.0, "prior": 0.5}
