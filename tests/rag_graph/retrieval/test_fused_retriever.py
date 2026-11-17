# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for the fused retriever: rank fusion, channels, tiers, de-duplication, variants and the retrieval switch."""

from typing import Dict, FrozenSet, List, Optional, Tuple

import pytest

from src.rag_graph.retrieval.fused_retriever import (
    F1,
    F2,
    F3,
    FusedRetriever,
    ReductionRecord,
    assign_tier,
    code_popularity,
    prior_channel,
    rank_positions,
    reciprocal_rank_fusion,
    sector_channel,
    use_channel,
)
from src.rag_graph.retrieval.switches import (
    RETRIEVAL_FUSED,
    RETRIEVAL_LEGACY,
    resolve_switch,
)


def rec(
    rid: str, cas: str, code: str, uses: Tuple[str, ...] = (), sector: str = "", year: int = 2020, name: str = ""
) -> ReductionRecord:
    return ReductionRecord(rid, rid, cas, f"chem-{cas}", code, name or f"name-{code}", uses, sector, "", year, "TX", None)


class FakeSource:
    """In-memory FusedSource with call logging."""

    def __init__(
        self,
        records: List[ReductionRecord],
        chem: List[Tuple[str, float, str]],
        acts: Dict[str, float],
        sectors: Dict[str, float],
    ) -> None:
        self.records, self.chem, self.acts, self.sectors = records, chem, acts, sectors
        self.calls: List[str] = []
        self.splits: List[Optional[str]] = []

    def chemical_hits(self, smiles: str, exclude_cas: FrozenSet[str]) -> List[Tuple[str, float, str]]:
        self.calls.append("chemical")
        return [h for h in self.chem if h[0] not in exclude_cas]

    def activity_hits(self, text: str, k: int) -> Dict[str, float]:
        self.calls.append("use")
        return self.acts

    def sector_hits(self, text: str, k: int) -> Dict[str, float]:
        self.calls.append("sector")
        return self.sectors

    def reduction_records(self, report_split: Optional[str]) -> List[ReductionRecord]:
        self.splits.append(report_split)
        return self.records


RECORDS = [
    rec("r1", "A", "S01", ("solvent",), "111111"),
    rec("r2", "A", "S02", ("reactant",), "222222"),
    rec("r3", "B", "S03", ("solvent",), "111111"),
    rec("r4", "B", "S04", ("reactant",), "111111"),
    rec("r5", "C", "S05", ("solvent",), "333333"),
    rec("r6", "Z", "S01", (), "444444"),
]
CHEM = [("A", 1.0, ""), ("B", 0.9, "")]


def _source() -> FakeSource:
    return FakeSource(RECORDS, CHEM, {"solvent": 0.8}, {"111111": 0.7})


class TestRankFusion:
    def test_equal_scores_share_the_best_rank(self) -> None:
        assert rank_positions({0: 0.9, 1: 0.9, 2: 0.5}) == {0: 1, 1: 1, 2: 3}

    def test_rrf_is_the_sum_of_reciprocal_ranks_with_k_60(self) -> None:
        fused = reciprocal_rank_fusion({"a": {0: 2.0, 1: 1.0}, "b": {1: 5.0}})
        assert fused[0] == pytest.approx(1 / 61)
        assert fused[1] == pytest.approx(1 / 62 + 1 / 61)

    def test_record_in_no_channel_is_absent(self) -> None:
        assert 5 not in reciprocal_rank_fusion({"a": {0: 1.0}})

    def test_use_channel_takes_the_best_matched_use(self) -> None:
        r = rec("x", "A", "S01", ("a", "b"))
        assert use_channel([r], {"a": 0.4, "b": 0.9}) == {0: 0.9}

    def test_sector_channel_matches_exact_codes(self) -> None:
        assert sector_channel(RECORDS, {"111111": 0.7}) == {0: 0.7, 2: 0.7, 3: 0.7}

    def test_prior_counts_distinct_chemicals_per_code(self) -> None:
        assert code_popularity(RECORDS)["S01"] == 2
        assert prior_channel(RECORDS)[5] == 2.0


class TestTiers:
    @pytest.mark.parametrize(
        "similarity,use,sector,tier",
        [
            (1.0, True, False, 1),
            (0.9, True, False, 2),
            (1.0, False, False, 3),
            (0.9, False, True, 4),
            (0.9, False, False, 5),
            (None, True, True, 6),
            (None, False, False, 6),
        ],
    )
    def test_tier_assignment(self, similarity: Optional[float], use: bool, sector: bool, tier: int) -> None:
        assert assign_tier(similarity, use, sector) == tier

    def test_retrieved_rows_carry_tier_and_channels(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", "sector text", "solvent use", limit=50)
        by_id = {(r["CAS"], r["ReductionCode"]): r for r in out.records}
        assert by_id[("A", "S01")]["Tier"] == 1
        assert by_id[("A", "S02")]["Tier"] == 3
        assert by_id[("B", "S03")]["Tier"] == 2
        assert by_id[("B", "S04")]["Tier"] == 4
        assert by_id[("C", "S05")]["Tier"] == 6
        assert by_id[("A", "S01")]["Channels"] == ["chemical", "use", "sector", "prior"]


class TestRetrieval:
    def test_channels_for_missing_inputs_are_skipped(self) -> None:
        src = _source()
        FusedRetriever(src).retrieve("C", "", "", options=F2)
        assert src.calls == ["chemical"]

    def test_report_split_is_passed_to_the_source(self) -> None:
        src = _source()
        FusedRetriever(src).retrieve("C", report_split="index")
        assert src.splits == ["index"]

    def test_hidden_reports_never_appear(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", "", "", exclude_report_ids=frozenset({"r1", "r2"}), limit=50)
        assert {r["CAS"] for r in out.records if r["CAS"] == "A"} == set()

    def test_excluded_chemicals_leave_the_pool(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", "", "", options=F2, exclude_cas=frozenset({"A"}), limit=50)
        assert out.records[0]["CAS"] == "B"
        assert all(r["Similarity"] == 0.0 for r in out.records if r["CAS"] != "B")

    def test_dedup_matches_tabulate_key(self) -> None:
        dup = RECORDS + [rec("r7", "A", "S01", ("solvent",), "111111")]
        out = FusedRetriever(FakeSource(dup, CHEM, {}, {})).retrieve("C", options=F2, limit=50)
        keys = [(r["Chemical"], r["Reduction Activity"], r["Year"], r["State"]) for r in out.records]
        assert len(keys) == len(set(keys)) == len(RECORDS) - 2  # r6 is outside the pool, r7 duplicates r1

    def test_row_shape_extends_tabulate_records(self) -> None:
        row = FusedRetriever(_source()).retrieve("C", "", "").records[0]
        for key in (
            "Chemical",
            "CAS",
            "Similarity",
            "EntryPoint",
            "Sector",
            "Reduction Activity",
            "ReductionCode",
            "Year",
            "State",
        ):
            assert key in row
        assert {"Tier", "Channels"} <= set(row)

    def test_limit_applies_after_dedup(self) -> None:
        assert len(FusedRetriever(_source()).retrieve("C", limit=3).records) == 3


class TestVariants:
    def _catch_all_source(self) -> FakeSource:
        recs = [rec("a", "A", "W58", name="OTHER PROCESS MODIFICATIONS"), rec("b", "A", "S01", name="SUBSTITUTED SOLVENT")]
        return FakeSource(recs, [("A", 1.0, "")], {}, {})

    def test_f2_has_no_prior_channel(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", options=F2, limit=50)
        assert all("prior" not in r["Channels"] for r in out.records)

    def test_f1_has_the_prior_channel(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", options=F1, limit=50)
        assert all("prior" in r["Channels"] for r in out.records)

    def test_f3_moves_catch_all_codes_to_the_end(self) -> None:
        src = self._catch_all_source()
        f1 = [r["ReductionCode"] for r in FusedRetriever(src).retrieve("C", options=F1).records]
        f3 = [r["ReductionCode"] for r in FusedRetriever(src).retrieve("C", options=F3).records]
        assert f1 == ["S01", "W58"] or f1 == ["W58", "S01"]
        assert f3 == ["S01", "W58"]


class TestUseNotice:
    def test_flag_set_when_no_record_reaches_tier_1_or_2(self) -> None:
        src = FakeSource(RECORDS, CHEM, {"unrelated": 0.9}, {})
        out = FusedRetriever(src).retrieve("C", "", "an unmatched use")
        assert out.use_supplied and out.use_unmatched

    def test_flag_clear_when_a_use_matches(self) -> None:
        out = FusedRetriever(_source()).retrieve("C", "", "solvent use")
        assert out.use_supplied and not out.use_unmatched

    def test_flag_clear_when_no_use_is_supplied(self) -> None:
        out = FusedRetriever(_source()).retrieve("C")
        assert not out.use_supplied and not out.use_unmatched


class TestRetrievalSwitch:
    def test_default_is_fused(self) -> None:
        assert resolve_switch("retrieval") == RETRIEVAL_FUSED

    def test_legacy_stays_available(self) -> None:
        assert resolve_switch("retrieval", "legacy") == RETRIEVAL_LEGACY

    def test_override_to_fused(self) -> None:
        assert resolve_switch("retrieval", "fused") == RETRIEVAL_FUSED

    def test_invalid_value_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            resolve_switch("retrieval", "hybrid")
