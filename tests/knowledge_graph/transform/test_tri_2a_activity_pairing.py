# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Regression test: each activity must pair with its own estimated annual reduction code."""

from typing import Dict

import pandas as pd

from src.knowledge_graph.transform.tri_2a_transformer import Tri2aTransformer

ORDINALS = ["first", "second", "third", "fourth"]


def _build_row(estimates: Dict[str, str]) -> Dict[str, object]:
    row: Dict[str, object] = {"document_control_number": 1320219126947}
    for i, ordinal in enumerate(ORDINALS):
        prefix = f"{ordinal}_source_reduction_activity"
        row[f"{prefix}_code"] = f"W{i + 1}"
        row[f"{prefix}_description"] = f"desc {i + 1}"
        row[f"{prefix}_code_est_annual_reduction_code"] = estimates[ordinal]
        row[f"{prefix}_code_est_annual_reduction_code_description"] = "greater than 0%, but less than 5%"
    return row


class TestActivityPairing:

    def test_mapping_excludes_description_columns(self):
        columns = list(_build_row({o: "R1" for o in ORDINALS}))
        mapping = Tri2aTransformer()._get_reduction_columns_mapping(columns)
        assert len(mapping["annual_reductions"]) == 4
        assert all(not col.endswith("_description") for col in mapping["annual_reductions"])

    def test_four_activities_pair_with_own_estimate(self):
        estimates = {"first": "R3", "second": "R1", "third": "R5", "fourth": "R2"}
        df = pd.DataFrame([_build_row(estimates)])
        result = Tri2aTransformer().transform(df)
        got = dict(zip(result["source_reduction_code"], result["annual_reduction"]))
        assert got == {"W1": 25.0, "W2": 100.0, "W4": 50.0}

    def test_filter_columns_drops_estimate_descriptions(self):
        df = pd.DataFrame([_build_row({o: "R1" for o in ORDINALS})])
        kept = Tri2aTransformer()._filter_columns(df).columns
        assert not any(col.endswith("est_annual_reduction_code_description") for col in kept)
