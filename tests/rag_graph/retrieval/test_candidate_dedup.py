# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Unit tests for category deduplication of candidate-derived hits."""

from src.rag_graph.retrieval.retriever_factory import dedup_categories_by_max_score


class TestDedupCategoriesByMaxScore:

    def test_empty_returns_empty(self):
        assert dedup_categories_by_max_score([]) == []

    def test_collapses_same_category_to_max_score_keeping_entry_member(self):
        # Three members of N084 match; keep the category once, at the best member's score,
        # and retain that member as the entry point.
        hits = [
            ("N084", "2-chlorophenol (CAS 95-57-8)", 0.62),
            ("N084", "2,3-dichlorophenol (CAS 576-24-9)", 0.81),
            ("N084", "3-chlorophenol (CAS 108-43-0)", 0.55),
        ]
        assert dedup_categories_by_max_score(hits) == [("N084", "2,3-dichlorophenol (CAS 576-24-9)", 0.81)]

    def test_keeps_one_row_per_distinct_category(self):
        hits = [("N084", "m1", 0.62), ("N230", "m2", 0.40), ("N084", "m3", 0.90)]
        result = dedup_categories_by_max_score(hits)
        assert result == [("N084", "m3", 0.90), ("N230", "m2", 0.40)]

    def test_sorted_by_score_descending(self):
        hits = [("A", "ma", 0.2), ("B", "mb", 0.9), ("C", "mc", 0.5)]
        scores = [score for _, _, score in dedup_categories_by_max_score(hits)]
        assert scores == sorted(scores, reverse=True)

    def test_no_duplicate_categories_in_output(self):
        hits = [("A", "m1", 0.1), ("A", "m2", 0.2), ("B", "m3", 0.3), ("B", "m4", 0.3)]
        categories = [cat for cat, _, _ in dedup_categories_by_max_score(hits)]
        assert len(categories) == len(set(categories))
