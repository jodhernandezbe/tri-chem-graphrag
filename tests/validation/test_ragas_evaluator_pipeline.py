# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for RAGASEvaluator running the shipped pipeline per embedding condition."""

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.validation.ragas_evaluator import RAGASEvaluator

RECORD = {
    "Chemical": "Toluene",
    "CAS": "108-88-3",
    "Similarity": 0.9,
    "Sector": "Paint",
    "Reduction Activity": "W13",
    "Year": 2020,
    "State": "OH",
}


def _evaluator(result: Dict[str, Any]) -> RAGASEvaluator:
    evaluator = RAGASEvaluator.__new__(RAGASEvaluator)
    evaluator._factory = MagicMock()  # type: ignore[attr-defined]
    evaluator._pipeline = MagicMock()  # type: ignore[attr-defined]
    evaluator._pipeline.run_structured.return_value = result
    return evaluator


def _ok(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"answer": "A", "records": records, "needs_clarification": False, "clarification": ""}


class TestRunFull:

    def test_embedding_reaches_pipeline_as_switch_and_factory_untouched(self):
        evaluator = _evaluator(_ok([RECORD]))
        evaluator._run_full("MorganFingerprint", {"chemical_smiles": "CCO"}, "Q?")
        kwargs = evaluator._pipeline.run_structured.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs["switches"] == {"chemical_embedding": "MorganFingerprint"}
        assert evaluator._factory.mock_calls == []  # type: ignore[attr-defined]

    def test_pipeline_receives_question_and_key_only(self):
        evaluator = _evaluator(_ok([RECORD]))
        evaluator._run_full("Hybrid", {"chemical_smiles": "CCO"}, "What?")
        kwargs = evaluator._pipeline.run_structured.call_args.kwargs  # type: ignore[attr-defined]
        assert kwargs == {
            "query": "What?",
            "chemical_smiles": "CCO",
            "industry_sector": "",
            "switches": {"chemical_embedding": "Hybrid"},
        }

    def test_answer_and_contexts_come_from_result(self):
        evaluator = _evaluator(_ok([RECORD]))
        answer, contexts = evaluator._run_full("Hybrid", {}, "Q")
        assert answer == "A"
        assert len(contexts) == 1 and "CAS: 108-88-3" in contexts[0] and "Reduction Activity: W13" in contexts[0]

    def test_clarification_raises(self):
        result = {"answer": "", "records": [], "needs_clarification": True, "clarification": "bad"}
        with pytest.raises(RuntimeError, match="bad"):
            _evaluator(result)._run_full("Hybrid", {}, "Q")


class TestPipelineInputs:

    def test_key_routing(self):
        f = RAGASEvaluator._pipeline_inputs
        assert f({"metadata": {"cas_number": "1"}}, "CCO") == {"chemical_smiles": "CCO"}
        assert f({"metadata": {"naics_code": "3"}}, "Paint") == {"industry_sector": "Paint"}
        assert f({"metadata": {}}, "Q") == {}


class TestCollectSamples:

    def test_full_and_flat_dispatch(self):
        evaluator = _evaluator(_ok([RECORD]))
        evaluator._retrieval_query = lambda pair: "CCO"  # type: ignore[method-assign]
        evaluator._run_flat = MagicMock(return_value=("F", ["x"]))  # type: ignore[method-assign]
        pairs = [{"question": "Q", "ground_truth": "G", "metadata": {"cas_number": "1"}}]
        samples, stats = evaluator._collect_samples("Hybrid", pairs)
        assert samples[0]["user_input"] == "Q" and samples[0]["response"] == "A" and stats["n_failed"] == 0
        assert set(samples[0]) == {"user_input", "response", "retrieved_contexts", "reference"}
        flat, _ = evaluator._collect_samples("flat", pairs)
        assert flat[0]["response"] == "F"
        evaluator._run_flat.assert_called_once_with("MolDeBERTa", "CCO", "Q")  # type: ignore[attr-defined]
