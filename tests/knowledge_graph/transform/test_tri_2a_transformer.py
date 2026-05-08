# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import patch

import pandas as pd

from src.knowledge_graph.transform.tri_2a_transformer import Tri2aTransformer


class TestTri2aTransformer:

    def test_transformer_initialization(self):
        transformer = Tri2aTransformer()
        assert transformer.file_type == "2a"
        assert hasattr(transformer, "_reduction_codes_considered")
        assert isinstance(transformer._reduction_codes_considered, set)
        assert "R1" in transformer._reduction_codes_considered
        assert "R2" in transformer._reduction_codes_considered
        assert "R3" in transformer._reduction_codes_considered

    def test_get_reduction_value_from_config(self):
        transformer = Tri2aTransformer()
        assert transformer._get_reduction_value_from_config("R1") == 100.0
        assert transformer._get_reduction_value_from_config("R2") == 50.0
        assert transformer._get_reduction_value_from_config("R3") == 25.0
        assert transformer._get_reduction_value_from_config("INVALID") is None

    def test_is_valid_scalar(self):
        transformer = Tri2aTransformer()
        assert not transformer._is_valid_scalar(None)
        assert not transformer._is_valid_scalar(pd.NA)
        assert not transformer._is_valid_scalar(pd.Series([1, 2, 3]))
        assert not transformer._is_valid_scalar("")
        assert transformer._is_valid_scalar("S01")
        assert transformer._is_valid_scalar("R1")

    def test_is_significant_annual_reduction_accepts_configured_codes(self):
        transformer = Tri2aTransformer()
        assert transformer._is_significant_annual_reduction("R1")
        assert transformer._is_significant_annual_reduction("R2")
        assert transformer._is_significant_annual_reduction("R3")

    def test_is_significant_annual_reduction_rejects_unconfigured_codes(self):
        transformer = Tri2aTransformer()
        assert not transformer._is_significant_annual_reduction("R4")
        assert not transformer._is_significant_annual_reduction("R5")
        assert not transformer._is_significant_annual_reduction("R6")
        assert not transformer._is_significant_annual_reduction(None)
        assert not transformer._is_significant_annual_reduction("")

    def test_get_column_value_safely(self):
        transformer = Tri2aTransformer()
        row = pd.Series({"col_a": "value", "col_b": None, "col_c": pd.NA})
        assert transformer._get_column_value_safely(row, "col_a") == "value"
        assert transformer._get_column_value_safely(row, "col_b") is None
        assert transformer._get_column_value_safely(row, "col_c") is None
        assert transformer._get_column_value_safely(row, "missing") is None
        assert transformer._get_column_value_safely(row, None) is None

    def test_create_reduction_row_uses_config_value(self):
        transformer = Tri2aTransformer()
        base_row = pd.Series({"document_control_number": 123})
        reduction_data = {"code": "S01", "description": "Process improvement", "annual_reduction": "R1"}
        result = transformer._create_reduction_row(base_row, reduction_data)
        assert result["source_reduction_code"] == "S01"
        assert result["source_reduction_description"] == "Process improvement"
        assert result["annual_reduction"] == 100.0

    def test_create_reduction_row_none_annual_reduction(self):
        transformer = Tri2aTransformer()
        base_row = pd.Series({"document_control_number": 123})
        reduction_data = {"code": "S01", "description": "Desc", "annual_reduction": None}
        result = transformer._create_reduction_row(base_row, reduction_data)
        assert result["annual_reduction"] is None

    def test_get_reduction_columns_mapping(self):
        transformer = Tri2aTransformer()
        columns = [
            "document_control_number",
            "first_source_reduction_activity_code",
            "first_source_reduction_activity_description",
            "first_source_reduction_activity_code_est_annual_reduction_code",
            "second_source_reduction_activity_code",
        ]
        mapping = transformer._get_reduction_columns_mapping(columns)
        assert "first_source_reduction_activity_code" in mapping["codes"]
        assert "second_source_reduction_activity_code" in mapping["codes"]
        assert "first_source_reduction_activity_description" in mapping["descriptions"]
        assert "first_source_reduction_activity_code_est_annual_reduction_code" in mapping["annual_reductions"]
        assert "document_control_number" not in mapping["codes"]

    def test_extract_reduction_activities_filters_by_annual_code(self):
        transformer = Tri2aTransformer()
        row = pd.Series(
            {
                "document_control_number": 123,
                "first_source_reduction_activity_code": "S01",
                "first_source_reduction_activity_description": "Desc",
                "first_source_reduction_activity_code_est_annual_reduction_code": "R1",
                "second_source_reduction_activity_code": "S02",
                "second_source_reduction_activity_description": "Desc2",
                "second_source_reduction_activity_code_est_annual_reduction_code": "R5",
            }
        )
        columns_mapping = transformer._get_reduction_columns_mapping(list(row.index))
        activities = transformer._extract_reduction_activities_from_row(row, columns_mapping)
        assert len(activities) == 1
        assert activities[0]["code"] == "S01"
        assert activities[0]["annual_reduction"] == "R1"

    def test_expand_reduction_activities_output_columns(self):
        transformer = Tri2aTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "first_source_reduction_activity_code": ["S01"],
                "first_source_reduction_activity_description": ["Process improvement"],
                "first_source_reduction_activity_code_est_annual_reduction_code": ["R1"],
            }
        )
        result = transformer._expand_reduction_activities(test_df)
        assert set(result.columns) == {
            "document_control_number",
            "source_reduction_code",
            "source_reduction_description",
            "annual_reduction",
        }

    def test_expand_reduction_activities_skips_insignificant(self):
        transformer = Tri2aTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "first_source_reduction_activity_code": ["S01"],
                "first_source_reduction_activity_description": ["Desc"],
                "first_source_reduction_activity_code_est_annual_reduction_code": ["R6"],
            }
        )
        result = transformer._expand_reduction_activities(test_df)
        assert len(result) == 0

    def test_transform_returns_numeric_annual_reduction(self):
        transformer = Tri2aTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "first_source_reduction_activity_code": ["S01"],
                "first_source_reduction_activity_description": ["Desc"],
                "first_source_reduction_activity_code_est_annual_reduction_code": ["R2"],
            }
        )
        result = transformer.transform(test_df)
        assert len(result) == 1
        assert result["annual_reduction"].iloc[0] == 50.0

    def test_transform_document_control_number_is_integer(self):
        transformer = Tri2aTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "first_source_reduction_activity_code": ["S01"],
                "first_source_reduction_activity_description": ["Desc"],
                "first_source_reduction_activity_code_est_annual_reduction_code": ["R1"],
            }
        )
        result = transformer.transform(test_df)
        assert pd.api.types.is_integer_dtype(result["document_control_number"])
