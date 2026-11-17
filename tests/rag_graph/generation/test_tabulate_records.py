# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the shared source-reduction record tabulation helper."""

from src.rag_graph.generation.rag_pipeline import records_to_context, tabulate_records


def _hit(cas: str, score: float) -> dict:
    return {"node": {"cas_number": cas}, "score": score}


def _chemical_subgraph(cas: str, name: str, reports: list) -> dict:
    return {"node_type": "Chemical", "cas_number": cas, "name": name, "reports": reports}


class TestTabulateRecords:

    def test_empty_inputs_return_empty(self):
        assert tabulate_records([], []) == []

    def test_extracts_reduction_code_and_ranks_by_similarity(self):
        nodes = [_hit("71-43-2", 0.9), _hit("50-00-0", 0.3)]
        subgraphs = [
            _chemical_subgraph(
                "50-00-0",
                "Formaldehyde",
                [{"reduction_code": "S01", "reduction_name": "Substitution", "year": 2021, "state": "TX"}],
            ),
            _chemical_subgraph(
                "71-43-2",
                "Benzene",
                [{"reduction_code": "W13", "reduction_name": "Recycling", "year": 2022, "state": "CA"}],
            ),
        ]
        records = tabulate_records(nodes, subgraphs)
        # Highest-similarity chemical (benzene, 0.9) ranks first.
        assert records[0]["CAS"] == "71-43-2"
        assert records[0]["ReductionCode"] == "W13"
        assert records[1]["ReductionCode"] == "S01"

    def test_deduplicates_identical_records(self):
        nodes = [_hit("71-43-2", 0.9)]
        rep = {"reduction_code": "S01", "reduction_name": "Substitution", "year": 2021, "state": "TX"}
        subgraphs = [_chemical_subgraph("71-43-2", "Benzene", [rep, dict(rep)])]
        records = tabulate_records(nodes, subgraphs)
        assert len(records) == 1

    def test_ignores_non_chemical_subgraphs(self):
        subgraphs = [{"node_type": "Sector", "naics_code": "3254", "context": []}]
        assert tabulate_records([], subgraphs) == []

    def test_respects_limit(self):
        nodes = [_hit("71-43-2", 0.9)]
        reps = [{"reduction_code": f"S{i:02d}", "reduction_name": f"R{i}", "year": 2020 + i, "state": "TX"} for i in range(5)]
        subgraphs = [_chemical_subgraph("71-43-2", "Benzene", reps)]
        records = tabulate_records(nodes, subgraphs, limit=2)
        assert len(records) == 2

    def test_skips_reports_without_reduction(self):
        nodes = [_hit("71-43-2", 0.9)]
        subgraphs = [_chemical_subgraph("71-43-2", "Benzene", [{"reduction_code": None, "reduction_name": None, "year": 2021}])]
        assert tabulate_records(nodes, subgraphs) == []

    def test_interleave_surfaces_neighbor_before_saturation(self):
        # Top chemical (0.9) has many activities; a similar neighbour (0.8) has one.
        nodes = [_hit("71-43-2", 0.9), _hit("108-88-3", 0.8)]
        top_reps = [
            {"reduction_code": f"S{i:02d}", "reduction_name": f"R{i}", "year": 2020 + i, "state": "TX"} for i in range(5)
        ]
        neighbor_reps = [{"reduction_code": "W13", "reduction_name": "Recycling", "year": 2022, "state": "CA"}]
        subgraphs = [
            _chemical_subgraph("71-43-2", "Benzene", top_reps),
            _chemical_subgraph("108-88-3", "Toluene", neighbor_reps),
        ]
        # Similarity ranking alone fills the top-2 with the saturating chemical only.
        ranked = tabulate_records(nodes, subgraphs, limit=2)
        assert {r["CAS"] for r in ranked} == {"71-43-2"}
        # Interleaving surfaces the neighbour in the second slot.
        woven = tabulate_records(nodes, subgraphs, limit=2, interleave=True)
        assert woven[0]["CAS"] == "71-43-2"
        assert woven[1]["CAS"] == "108-88-3"


class TestRecordsToContext:

    def _rec(self, chem: str, cas: str, sim: float, activity: str) -> dict:
        return {
            "Chemical": chem,
            "CAS": cas,
            "Similarity": sim,
            "Sector": "N/A — N/A",
            "Reduction Activity": activity,
            "Year": 2022,
            "State": "TX",
        }

    def test_empty_records_yield_empty_context(self):
        assert records_to_context([], "CCO") == ""

    def test_context_only_contains_supplied_chemicals(self):
        records = [
            self._rec("Benzene", "71-43-2", 1.0, "Recycling"),
            self._rec("Toluene", "108-88-3", 0.88, "Substitution"),
        ]
        ctx = records_to_context(records, "c1ccccc1")
        # Query header and both supplied chemicals appear; nothing else can leak in.
        assert "QUERY STRUCTURE (SMILES): c1ccccc1" in ctx
        assert "Benzene" in ctx and "Toluene" in ctx
        assert "IDENTICAL STRUCTURE TO THE QUERY" in ctx
        assert "ANALOGUE (not the query chemical)" in ctx
        assert "Recycling" in ctx and "Substitution" in ctx

    def test_chemicals_ordered_by_descending_similarity(self):
        records = [
            self._rec("Toluene", "108-88-3", 0.80, "Substitution"),
            self._rec("Benzene", "71-43-2", 1.0, "Recycling"),
        ]
        ctx = records_to_context(records, "c1ccccc1")
        assert ctx.index("Benzene") < ctx.index("Toluene")

    def test_unnamed_query_is_not_given_an_analogue_name(self):
        ctx = records_to_context([self._rec("Allyl chloride", "107-05-1", 0.98, "X")], "C=CCl")
        assert "QUERY CHEMICAL NAME: not provided" in ctx
        assert "[Allyl chloride | CAS 107-05-1 | Similarity 0.98 — ANALOGUE" in ctx


class TestSectorFromReport:

    def test_chemical_path_uses_report_sector(self):
        rep = {
            "reduction_code": "S01",
            "reduction_name": "Sub",
            "year": 2020,
            "state": "TX",
            "sector_code": "3254",
            "sector_title": "Pharma",
        }
        rec = tabulate_records([_hit("1-1-1", 0.5)], [_chemical_subgraph("1-1-1", "X", [rep])])[0]
        assert rec["Sector"] == "3254 — Pharma"

    def test_chemical_sector_path_uses_subgraph_sector(self):
        sg = _chemical_subgraph("1-1-1", "X", [{"reduction_code": "S01", "year": 2020, "state": "TX"}])
        sg.update(node_type="Chemical_Sector", sector_code="3311", sector_title="Steel")
        assert tabulate_records([_hit("1-1-1", 0.5)], [sg])[0]["Sector"] == "3311 — Steel"

    def test_missing_sector_stays_na_and_keys_unchanged(self):
        reps = [
            {"reduction_code": "S01", "year": 2020, "state": "TX", "sector_code": "1", "sector_title": "A"},
            {"reduction_code": "S01", "year": 2020, "state": "TX", "sector_code": "2", "sector_title": "B"},
        ]
        records = tabulate_records([_hit("1-1-1", 0.5)], [_chemical_subgraph("1-1-1", "X", reps)])
        # Same (chemical, activity, year, state) still collapses to one record; first wins.
        assert len(records) == 1 and records[0]["Sector"] == "1 — A"
        bare = tabulate_records([_hit("1-1-1", 0.5)], [_chemical_subgraph("1-1-1", "X", [{"reduction_code": "S01"}])])
        assert bare[0]["Sector"] == "N/A — N/A"
