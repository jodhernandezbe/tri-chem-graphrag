# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the fused sector notice: matched uses and sectors, the similarity threshold, and the notice text."""

from typing import Dict, FrozenSet, Iterable, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest

from src.rag_graph.generation.rag_pipeline import RAGPipeline
from src.rag_graph.retrieval.fused_retriever import (
    SECTOR_UNMATCHED_NOTICE,
    USE_UNMATCHED_NOTICE,
    FusedRetriever,
    ReductionRecord,
    match_entries,
    matches_header_lines,
    record_industries,
    row_sector_code,
    sector_is_unmatched,
)
from src.rag_graph.retrieval.switches import (
    DEFAULT_SECTOR_NOTICE_THRESHOLD,
    resolve_sector_notice_threshold,
)

TITLES = {"313310": "Textile and fabric finishing mills", "335224": "Household laundry equipment manufacturing"}


def rec(rid: str, cas: str, code: str, sector: str, title: str) -> ReductionRecord:
    return ReductionRecord(rid, rid, cas, f"chem-{cas}", code, f"name-{code}", ("solvent",), sector, title, 2020, "TX", None)


class Source:
    """In-memory FusedSource with sector titles."""

    def __init__(self, records: List[ReductionRecord], sectors: Dict[str, float], with_titles: bool = True) -> None:
        self.records, self.sectors = records, sectors
        if with_titles:
            self.sector_titles = self._titles  # type: ignore[method-assign]

    def _titles(self, codes: Iterable[str]) -> Dict[str, str]:
        return {c: TITLES[c] for c in codes if c in TITLES}

    def chemical_hits(self, smiles: str, exclude_cas: FrozenSet[str]) -> List[Tuple[str, float, str]]:
        return [("A", 1.0, "")]

    def activity_hits(self, text: str, k: int) -> Dict[str, float]:
        return {"solvent": 0.9, "reactant": 0.7}

    def sector_hits(self, text: str, k: int) -> Dict[str, float]:
        return self.sectors

    def reduction_records(self, report_split: Optional[str]) -> List[ReductionRecord]:
        return self.records


RECORDS = [rec("r1", "A", "S01", "313310", "Textile and fabric finishing mills")]


def _retrieve(sectors: Dict[str, float], sector_query: str = "Dry cleaning", with_titles: bool = True):  # type: ignore[no-untyped-def]
    return FusedRetriever(Source(RECORDS, sectors, with_titles)).retrieve("C", sector_query, "solvent")


class TestThreshold:
    def test_default_threshold_is_the_documented_value(self) -> None:
        assert DEFAULT_SECTOR_NOTICE_THRESHOLD == 0.85

    def test_config_value_is_read(self) -> None:
        assert resolve_sector_notice_threshold() == 0.85

    @pytest.mark.parametrize("bad", [0, 1.5, "0.8", True])
    def test_invalid_threshold_is_rejected(self, bad: object) -> None:
        cfg = MagicMock()
        cfg.config = {"RAG_GRAPH": {"RETRIEVAL": {"SECTOR_NOTICE_MIN_SIMILARITY": bad}}}
        with patch("src.rag_graph.retrieval.switches.ConfigManager", return_value=cfg), pytest.raises(ValueError):
            resolve_sector_notice_threshold()


class TestMatches:
    def test_matches_are_exposed_best_first_with_titles(self) -> None:
        res = _retrieve({"335224": 0.80, "313310": 0.86})
        assert [m["code"] for m in res.matched_sectors] == ["313310", "335224"]
        assert res.matched_sectors[0] == {"code": "313310", "title": TITLES["313310"], "score": 0.86, "qualifying": True}
        assert res.matched_sectors[1]["qualifying"] is False
        assert res.matched_uses == [{"name": "solvent", "score": 0.9}, {"name": "reactant", "score": 0.7}]

    def test_titles_fall_back_to_record_titles(self) -> None:
        res = _retrieve({"313310": 0.9}, with_titles=False)
        assert res.matched_sectors[0]["title"] == "Textile and fabric finishing mills"

    def test_skipped_channels_expose_nothing(self) -> None:
        res = FusedRetriever(Source(RECORDS, {"313310": 0.9})).retrieve("C")
        assert res.matched_uses == [] and res.matched_sectors == [] and not res.sector_supplied

    def test_match_entries_marks_the_threshold_inclusively(self) -> None:
        _, sectors = match_entries({}, {"a": 0.85, "b": 0.8499}, {}, 0.85)
        assert [s["qualifying"] for s in sectors] == [True, False]

    def test_header_lines_list_both_channels(self) -> None:
        uses, sectors = match_entries({"solvent": 0.9}, {"313310": 0.86}, TITLES, 0.85)
        use_line, sector_line = matches_header_lines(uses, sectors)
        assert "solvent (0.9)" in use_line and "313310 Textile and fabric finishing mills (0.86*)" in sector_line

    def test_header_lines_show_a_dash_when_a_channel_did_not_run(self) -> None:
        assert matches_header_lines([], []) == [
            "Matched uses     : —",
            "Matched sectors  : —  [* = at or above the sector threshold]",
        ]


class TestSectorNotice:
    def test_fires_when_no_record_is_in_a_qualifying_sector(self) -> None:
        res = _retrieve({"335224": 0.80, "313310": 0.78})
        assert res.sector_supplied and res.sector_unmatched
        assert res.record_industries == ["Textile and fabric finishing mills"]

    def test_silent_when_a_record_sits_in_a_qualifying_sector(self) -> None:
        assert not _retrieve({"313310": 0.86}).sector_unmatched

    def test_silent_when_the_only_qualifying_sector_has_no_returned_record(self) -> None:
        assert _retrieve({"335224": 0.9, "313310": 0.7}).sector_unmatched

    def test_a_sector_at_the_threshold_counts_as_present(self) -> None:
        assert not _retrieve({"313310": 0.85}).sector_unmatched

    def test_silent_when_no_sector_was_supplied(self) -> None:
        res = _retrieve({"313310": 0.1}, sector_query=" ")
        assert not res.sector_supplied and not res.sector_unmatched

    def test_silent_when_there_are_no_records(self) -> None:
        assert not sector_is_unmatched([], [{"code": "1", "qualifying": True}])

    def test_use_flag_is_independent_of_the_sector_flag(self) -> None:
        res = _retrieve({"313310": 0.1})
        assert res.sector_unmatched and not res.use_unmatched

    def test_row_helpers(self) -> None:
        rows = [{"Sector": "313310 — Textile"}, {"Sector": "N/A — N/A"}, {"Sector": "313310 — Textile"}]
        assert row_sector_code(rows[0]) == "313310" and record_industries(rows) == ["Textile"]

    def test_notice_text_names_the_sector_and_the_industries(self) -> None:
        text = SECTOR_UNMATCHED_NOTICE.format(sector="Dry cleaning", industries="Textile; Soaps")
        assert "('Dry cleaning')" in text and "(Textile; Soaps)" in text


class TestPipelineNotice:
    def _answer(self, **extra: object) -> str:
        pipe = RAGPipeline.__new__(RAGPipeline)
        state = pipe._initial_state(
            "q", "C", "Dry cleaning", "solvent", {"retrieval": "fused"}
        )  # pyright: ignore[reportPrivateUsage]
        state.update({"records": [{"Reduction Activity": "x"}], "context": "ctx", **extra})  # type: ignore[typeddict-item]
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content="LLM ANSWER")
        with (
            patch("src.rag_graph.generation.rag_pipeline.ChatOllama", return_value=llm),
            patch("src.rag_graph.generation.rag_pipeline.PromptManager"),
            patch("src.rag_graph.generation.rag_pipeline.fix_query_chemical_mislabel", side_effect=lambda a, r: a),
        ):
            return str(pipe._generate(state)["answer"])  # pyright: ignore[reportPrivateUsage]

    def test_sector_notice_is_prepended(self) -> None:
        answer = self._answer(sector_notice=True, record_industries=["Textile", "Soaps"])
        assert answer == SECTOR_UNMATCHED_NOTICE.format(sector="Dry cleaning", industries="Textile; Soaps") + "\n\nLLM ANSWER"

    def test_no_notice_when_the_sector_matched(self) -> None:
        assert self._answer(sector_notice=False) == "LLM ANSWER"

    def test_both_notices_put_the_use_notice_first(self) -> None:
        answer = self._answer(sector_notice=True, use_notice=True, record_industries=["Textile"])
        assert answer.startswith(USE_UNMATCHED_NOTICE.format(cou="solvent")) and "stated sector" in answer
        assert answer.endswith("LLM ANSWER")

    def test_retrieve_passes_the_matches_to_the_state(self) -> None:
        pipe = RAGPipeline.__new__(RAGPipeline)
        state = pipe._initial_state("q", "C", "Dry cleaning", "", {"retrieval": "fused"})  # pyright: ignore[reportPrivateUsage]
        factory = MagicMock()
        factory.retrieve_fused.return_value = _retrieve({"335224": 0.8})
        with patch("src.rag_graph.generation.rag_pipeline.RetrieverFactory", return_value=factory):
            out = pipe._retrieve(state)  # pyright: ignore[reportPrivateUsage]
        assert out["sector_notice"] is True and out["matched_sectors"][0]["code"] == "335224"
        assert out["record_industries"] == ["Textile and fabric finishing mills"]
