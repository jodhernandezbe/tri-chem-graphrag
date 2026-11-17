# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the fused-retrieval pipeline path: switch routing, the unmatched-use notice, context and table, legacy intact."""

from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.rag_graph.generation.rag_pipeline import (
    RAGPipeline,
    records_table_lines,
    records_to_context,
)
from src.rag_graph.retrieval.fused_retriever import USE_UNMATCHED_NOTICE, FusedResult

FUSED_ROW: Dict[str, Any] = {
    "Chemical": "Trichloroethylene",
    "CAS": "79-01-6",
    "Similarity": 1.0,
    "EntryPoint": "",
    "Sector": "332813 — Electroplating",
    "Reduction Activity": "SUBSTITUTED AN ORGANIC SOLVENT",
    "ReductionCode": "S02",
    "Year": 2020,
    "State": "TX",
    "Uses": ["Used as a degreasing solvent"],
    "Tier": 1,
    "Channels": ["chemical", "use", "prior"],
}
LEGACY_ROW: Dict[str, Any] = {k: v for k, v in FUSED_ROW.items() if k not in ("Tier", "Channels", "Uses")}


def _pipeline_state(switches: Dict[str, str], **extra: Any) -> Dict[str, Any]:
    pipe = RAGPipeline.__new__(RAGPipeline)
    state = pipe._initial_state("q", "ClC=C(Cl)Cl", "", "degreasing", switches)  # pyright: ignore[reportPrivateUsage]
    return {"pipe": pipe, "state": {**state, **extra}}


class TestRouting:
    def test_default_calls_the_fused_retriever(self) -> None:
        ctx = _pipeline_state({})
        factory = MagicMock()
        factory.retrieve_fused.return_value = FusedResult([FUSED_ROW], True, False, [])
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            ctx["pipe"]._retrieve(ctx["state"])  # pyright: ignore[reportPrivateUsage]
        factory.retrieve_fused.assert_called_once()
        factory.retrieve_multi_result.assert_not_called()

    def test_embedding_switch_reaches_the_fused_retriever(self) -> None:
        ctx = _pipeline_state({"chemical_embedding": "MorganFingerprint"})
        factory = MagicMock()
        factory.retrieve_fused.return_value = FusedResult([FUSED_ROW], True, False, [])
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            ctx["pipe"]._retrieve(ctx["state"])  # pyright: ignore[reportPrivateUsage]
        assert factory.retrieve_fused.call_args.kwargs["chemical_embedding"] == "MorganFingerprint"

    def test_no_embedding_switch_passes_none(self) -> None:
        ctx = _pipeline_state({})
        factory = MagicMock()
        factory.retrieve_fused.return_value = FusedResult([FUSED_ROW], True, False, [])
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            ctx["pipe"]._retrieve(ctx["state"])  # pyright: ignore[reportPrivateUsage]
        assert factory.retrieve_fused.call_args.kwargs["chemical_embedding"] is None

    def test_legacy_override_calls_the_legacy_retriever(self) -> None:
        ctx = _pipeline_state({"retrieval": "legacy"})
        factory = MagicMock()
        factory.retrieve_multi_result.return_value = MagicMock(
            retrieved_nodes=[], subgraphs=[], sector_fallback=False, cou_fallback=False
        )
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            ctx["pipe"]._retrieve(ctx["state"])  # pyright: ignore[reportPrivateUsage]
        factory.retrieve_multi_result.assert_called_once()
        factory.retrieve_fused.assert_not_called()

    def test_fused_switch_calls_the_fused_retriever(self) -> None:
        ctx = _pipeline_state({"retrieval": "fused"})
        factory = MagicMock()
        factory.retrieve_fused.return_value = FusedResult([FUSED_ROW], True, False, [])
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            out = ctx["pipe"]._retrieve(ctx["state"])  # pyright: ignore[reportPrivateUsage]
        factory.retrieve_multi_result.assert_not_called()
        assert out["records"] == [FUSED_ROW] and out["use_notice"] is False

    def test_tabulate_keeps_fused_records(self) -> None:
        ctx = _pipeline_state({"retrieval": "fused"}, records=[FUSED_ROW])
        assert ctx["pipe"]._tabulate(ctx["state"]) == {"records": [FUSED_ROW]}  # pyright: ignore[reportPrivateUsage]


class TestNotice:
    def _generate(self, use_notice: bool) -> str:
        ctx = _pipeline_state({"retrieval": "fused"}, records=[FUSED_ROW], context="ctx", use_notice=use_notice)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="LLM ANSWER")
        with (
            patch("src.rag_graph.generation.rag_pipeline.ChatOllama", return_value=llm),
            patch("src.rag_graph.generation.rag_pipeline.PromptManager"),
        ):
            return str(ctx["pipe"]._generate(ctx["state"])["answer"])  # pyright: ignore[reportPrivateUsage]

    def test_notice_is_prepended_when_the_use_is_unmatched(self) -> None:
        answer = self._generate(True)
        assert answer.startswith(USE_UNMATCHED_NOTICE.format(cou="degreasing"))
        assert answer.endswith("LLM ANSWER")

    def test_no_notice_when_the_use_matched(self) -> None:
        assert self._generate(False) == "LLM ANSWER"


class TestFormatting:
    def test_fused_context_shows_tier_and_channels(self) -> None:
        text = records_to_context([FUSED_ROW], "ClC=C(Cl)Cl")
        assert "Tier 1" in text and "Channels chemical,use,prior" in text and "SUBSTITUTED AN ORGANIC SOLVENT" in text

    def test_fused_table_has_tier_and_channels_columns(self) -> None:
        lines: List[str] = records_table_lines([FUSED_ROW])
        assert lines[1].endswith("Tier | Channels | Rank |") and lines[3].endswith("1 | chemical,use,prior | 1 |")

    def test_legacy_table_and_context_are_unchanged(self) -> None:
        lines = records_table_lines([LEGACY_ROW])
        assert lines[1] == "| Chemical | CAS | Similarity | Sector | Reduction Activity | Year | State |"
        assert lines[2] == "|---|---|---|---|---|---|---|"
        assert (
            lines[3]
            == "| Trichloroethylene | 79-01-6 | 1.0 | 332813 — Electroplating | SUBSTITUTED AN ORGANIC SOLVENT | 2020 | TX |"
        )
        assert "Tier" not in records_to_context([LEGACY_ROW], "ClC=C(Cl)Cl")
