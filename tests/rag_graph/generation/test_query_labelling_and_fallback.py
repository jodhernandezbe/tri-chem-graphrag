# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the analogue-mislabel safeguard and the deterministic sector-fallback notice."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.rag_graph.generation.rag_pipeline import (
    RAGPipeline,
    fix_query_chemical_mislabel,
)

RECORDS: List[Dict[str, Any]] = [
    {"Chemical": "Allyl chloride", "Similarity": 0.98},
    {"Chemical": "Benzene", "Similarity": 1.0},
]


class TestFixQueryChemicalMislabel:

    def test_analogue_named_as_query_is_relabelled(self):
        out = fix_query_chemical_mislabel("for the query chemical (C=CCl, Allyl chloride) is", RECORDS)
        assert out == "for the closest analogue (C=CCl, Allyl chloride) is"

    def test_identical_structure_name_left_alone(self):
        text = "the query chemical (Benzene) was reported"
        assert fix_query_chemical_mislabel(text, RECORDS) == text

    def test_analogue_cited_with_similarity_to_query_untouched(self):
        text = "Allyl chloride, similarity 0.98 to the query chemical"
        assert fix_query_chemical_mislabel(text, RECORDS) == text

    def test_no_records_is_noop(self):
        assert fix_query_chemical_mislabel("the query chemical (X)", []) == "the query chemical (X)"


def _generate(sector_fallback: bool, sector: str) -> str:
    state: Dict[str, Any] = {
        "query": "q",
        "records": RECORDS,
        "context": "ctx",
        "sector_query": sector,
        "sector_fallback": sector_fallback,
    }
    with patch("src.rag_graph.generation.rag_pipeline.ChatOllama") as chat:
        chat.return_value.invoke.return_value = MagicMock(content="LLM answer")
        return str(RAGPipeline()._generate(state)["answer"])  # type: ignore[arg-type]


class TestSectorFallbackNotice:

    def test_notice_prepended_when_fallback(self):
        answer = _generate(True, "Dry cleaning")
        assert answer.startswith("Note: No qualifying records were found in the stated sector ('Dry cleaning')")
        assert "other industries" in answer and answer.endswith("LLM answer")

    def test_no_notice_without_fallback(self):
        assert _generate(False, "Dry cleaning") == "LLM answer"
