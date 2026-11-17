# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the ``ranking`` switch of ``tabulate_records``."""

from typing import Any, Dict, List, Optional

import pytest

from src.rag_graph.generation.rag_pipeline import tabulate_records

# S_COMMON is reported by everybody (IDF 0.1); S_RARE by few (IDF 3.0).
IDF: Dict[str, float] = {"S_COMMON": 0.1, "S_RARE": 3.0, "S_MID": 1.0}


def _hit(cas: str, score: float, context: Optional[float] = None) -> Dict[str, Any]:
    hit: Dict[str, Any] = {"node": {"cas_number": cas}, "content": "", "score": score}
    if context is not None:
        hit["context_similarity"] = context
    return hit


def _sg(cas: str, codes: List[str]) -> Dict[str, Any]:
    reports = [{"reduction_code": c, "reduction_name": c, "year": 2020 + i, "state": "OH"} for i, c in enumerate(codes)]
    return {"node_type": "Chemical", "cas_number": cas, "name": cas, "reports": reports}


def _codes(records: List[Dict[str, Any]]) -> List[str]:
    return [r["ReductionCode"] for r in records]


class TestSimilarityIsDefault:

    def test_default_orders_by_molecular_similarity(self):
        nodes = [_hit("a", 0.9), _hit("b", 0.5)]
        sgs = [_sg("b", ["S_RARE"]), _sg("a", ["S_COMMON"])]
        assert _codes(tabulate_records(nodes, sgs)) == ["S_COMMON", "S_RARE"]

    def test_explicit_similarity_equals_default(self):
        nodes = [_hit("a", 0.9), _hit("b", 0.5)]
        sgs = [_sg("b", ["S_RARE", "S_MID"]), _sg("a", ["S_COMMON"])]
        assert tabulate_records(nodes, sgs, interleave=True) == tabulate_records(
            nodes, sgs, interleave=True, ranking="similarity"
        )

    def test_default_adds_no_context_key(self):
        records = tabulate_records([_hit("a", 0.9)], [_sg("a", ["S_COMMON"])])
        assert "ContextSimilarity" not in records[0]


class TestSpecificity:

    def test_rare_code_beats_higher_similarity_common_code(self):
        nodes = [_hit("a", 0.9), _hit("b", 0.5)]
        sgs = [_sg("a", ["S_COMMON"]), _sg("b", ["S_RARE"])]
        # 0.9 * 0.1 = 0.09 < 0.5 * 3.0 = 1.5
        assert _codes(tabulate_records(nodes, sgs, ranking="specificity", code_idf=IDF)) == ["S_RARE", "S_COMMON"]

    def test_requires_idf(self):
        with pytest.raises(ValueError, match="code_idf"):
            tabulate_records([_hit("a", 0.9)], [_sg("a", ["S_COMMON"])], ranking="specificity")

    def test_unknown_ranking_rejected(self):
        with pytest.raises(ValueError):
            tabulate_records([_hit("a", 0.9)], [_sg("a", ["S_COMMON"])], ranking="bogus", code_idf=IDF)

    def test_unseen_code_gets_maximum_idf(self):
        nodes = [_hit("a", 0.5)]
        sgs = [_sg("a", ["S_NEW", "S_COMMON"])]
        assert _codes(tabulate_records(nodes, sgs, ranking="specificity", code_idf=IDF))[0] == "S_NEW"


class TestConsensusSpecificity:

    def test_code_reported_by_several_neighbours_rises(self):
        idf = {"S_A": 1.0, "S_B": 1.0}
        nodes = [_hit("a", 0.9), _hit("b", 0.5), _hit("c", 0.5)]
        sgs = [_sg("a", ["S_A"]), _sg("b", ["S_B"]), _sg("c", ["S_B"])]
        # consensus: S_B = (0.5 + 0.5) * 1 = 1.0 > S_A = 0.9 ; plain specificity keeps S_A first (0.9 > 0.5)
        assert _codes(tabulate_records(nodes, sgs, ranking="consensus_specificity", code_idf=idf))[0] == "S_B"
        assert _codes(tabulate_records(nodes, sgs, ranking="specificity", code_idf=idf))[0] == "S_A"

    def test_same_chemical_counts_once_per_code(self):
        nodes = [_hit("a", 0.5), _hit("b", 0.5)]
        sgs = [_sg("a", ["S_MID", "S_MID", "S_MID"]), _sg("b", ["S_RARE"])]
        # S_MID is one supporting chemical: 0.5 * 1.0 < S_RARE 0.5 * 3.0
        assert _codes(tabulate_records(nodes, sgs, ranking="consensus_specificity", code_idf=IDF))[0] == "S_RARE"

    def test_interleave_is_ignored(self):
        nodes = [_hit("a", 0.9), _hit("b", 0.5)]
        sgs = [_sg("a", ["S_COMMON", "S_MID"]), _sg("b", ["S_RARE"])]
        plain = tabulate_records(nodes, sgs, ranking="specificity", code_idf=IDF)
        assert tabulate_records(nodes, sgs, interleave=True, ranking="specificity", code_idf=IDF) == plain


class TestContextSimilarityDrivesRanking:

    def test_context_similarity_replaces_molecular_for_ordering_only(self):
        nodes = [_hit("a", 0.9, context=0.1), _hit("b", 0.5, context=0.8)]
        sgs = [_sg("a", ["S_COMMON"]), _sg("b", ["S_MID"])]
        records = tabulate_records(nodes, sgs)
        assert _codes(records) == ["S_MID", "S_COMMON"]
        assert [r["Similarity"] for r in records] == [0.5, 0.9]
        assert [r["ContextSimilarity"] for r in records] == [0.8, 0.1]

    def test_specificity_uses_context_similarity(self):
        nodes = [_hit("a", 0.9, context=0.01), _hit("b", 0.5, context=0.5)]
        sgs = [_sg("a", ["S_RARE"]), _sg("b", ["S_MID"])]
        # molecular: 0.9*3 = 2.7 > 0.5 ; context: 0.01*3 = 0.03 < 0.5*1
        assert _codes(tabulate_records(nodes, sgs, ranking="specificity", code_idf=IDF)) == ["S_MID", "S_RARE"]
