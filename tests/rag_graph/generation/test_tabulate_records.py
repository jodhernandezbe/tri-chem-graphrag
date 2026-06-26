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
        assert "Query chemical SMILES: c1ccccc1" in ctx
        assert "Benzene" in ctx and "Toluene" in ctx
        assert "QUERY CHEMICAL" in ctx  # the similarity-1.0 chemical is marked
        assert "Recycling" in ctx and "Substitution" in ctx

    def test_chemicals_ordered_by_descending_similarity(self):
        records = [
            self._rec("Toluene", "108-88-3", 0.80, "Substitution"),
            self._rec("Benzene", "71-43-2", 1.0, "Recycling"),
        ]
        ctx = records_to_context(records, "c1ccccc1")
        assert ctx.index("Benzene") < ctx.index("Toluene")
