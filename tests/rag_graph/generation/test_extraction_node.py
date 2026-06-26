# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the structured chemical-extraction LangGraph node.

The deterministic RDKit validation runs for real; the ChatOllama structured-output
call is mocked so the tests need neither a live Ollama server nor Neo4j.
"""

from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from src.rag_graph.generation.rag_pipeline import ChemicalExtraction, RAGPipeline


def _state(query: str = "q", **overrides: Any) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "query": query,
        "chemical_smiles": "",
        "sector_query": "",
        "condition_of_use": "",
    }
    base.update(overrides)
    return base


def _run_extract(extraction: Optional[ChemicalExtraction], state: Dict[str, Any]) -> Dict[str, Any]:
    pipeline = RAGPipeline()
    with patch("src.rag_graph.generation.rag_pipeline.ChatOllama") as mock_chat:
        structured = MagicMock()
        structured.invoke.return_value = extraction if extraction is not None else ChemicalExtraction()
        mock_chat.return_value.with_structured_output.return_value = structured
        return pipeline._extract(state)  # type: ignore[arg-type]


class TestExtractionNode:

    def test_valid_smiles_canonicalizes_and_routes_to_retrieve(self) -> None:
        extraction = ChemicalExtraction(smiles="OCC")  # non-canonical ethanol
        updates = _run_extract(extraction, _state())
        assert updates["chemical_smiles"] == "CCO"  # RDKit canonical form
        assert updates["needs_clarification"] is False
        assert RAGPipeline()._route_after_extract(updates) == "retrieve"  # type: ignore[arg-type]

    def test_invalid_smiles_routes_to_clarify(self) -> None:
        extraction = ChemicalExtraction(smiles="XYZ123notSMILES")
        updates = _run_extract(extraction, _state())
        assert updates["needs_clarification"] is True
        assert updates["chemical_smiles"] == ""
        assert "not a valid chemical structure" in updates["clarification"]
        assert RAGPipeline()._route_after_extract(updates) == "clarify"  # type: ignore[arg-type]

    def test_no_smiles_routes_to_plan(self) -> None:
        extraction = ChemicalExtraction(industry_sector="plastics manufacturing")
        updates = _run_extract(extraction, _state())
        assert updates["chemical_smiles"] == ""
        assert updates["needs_clarification"] is False
        assert RAGPipeline()._route_after_extract(updates) == "plan"  # type: ignore[arg-type]

    def test_sector_and_condition_pass_through(self) -> None:
        extraction = ChemicalExtraction(
            smiles="CCO",
            industry_sector="chemical manufacturing",
            condition_of_use="industrial solvent",
        )
        updates = _run_extract(extraction, _state())
        assert updates["sector_query"] == "chemical manufacturing"
        assert updates["condition_of_use"] == "industrial solvent"

    def test_explicit_smiles_overrides_extracted(self) -> None:
        extraction = ChemicalExtraction(smiles="c1ccccc1")  # benzene
        updates = _run_extract(extraction, _state(chemical_smiles="CCO"))
        assert updates["chemical_smiles"] == "CCO"  # explicit caller value wins

    def test_failed_extraction_is_graceful(self) -> None:
        pipeline = RAGPipeline()
        with patch("src.rag_graph.generation.rag_pipeline.ChatOllama") as mock_chat:
            structured = MagicMock()
            structured.invoke.side_effect = RuntimeError("ollama down")
            mock_chat.return_value.with_structured_output.return_value = structured
            updates = pipeline._extract(_state())  # type: ignore[arg-type]
        assert updates["needs_clarification"] is False
        assert updates["chemical_smiles"] == ""
        assert pipeline._route_after_extract(updates) == "plan"  # type: ignore[arg-type]
