# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import patch

import pandas as pd
import pytest

from src.knowledge_graph.transform.tri_1b_transformer import Tri1bTransformer


class TestTri1bTransformer:

    def test_transformer_initialization(self):
        transformer = Tri1bTransformer()
        assert transformer.file_type == "1b"
        assert hasattr(transformer, "_activity_columns")
        assert hasattr(transformer, "_yes_values")
        assert isinstance(transformer._yes_values, set)

    def test_get_activity_columns_excludes_document_control_number(self):
        transformer = Tri1bTransformer()
        columns = transformer._get_activity_columns_from_config()
        assert isinstance(columns, list)
        assert "document_control_number" not in columns

    def test_is_yes_value_true_cases(self):
        transformer = Tri1bTransformer()
        for val in ["yes", "Yes", "YES", "y", "Y", "1", "true", "True"]:
            assert transformer._is_yes_value(val), f"Expected True for '{val}'"

    def test_is_yes_value_false_cases(self):
        transformer = Tri1bTransformer()
        for val in ["no", "No", "0", "false", "", None, pd.NA]:
            assert not transformer._is_yes_value(val), f"Expected False for '{val}'"

    def test_format_activity_label(self):
        transformer = Tri1bTransformer()
        assert transformer._format_activity_label("used_as_an_article_component") == "Used as an article component"
        assert transformer._format_activity_label("produce_the_chemical") == "Produce the chemical"
        assert transformer._format_activity_label("z299_other") == "Z299 other"

    def test_standardize_activity_columns_converts_to_bool(self):
        transformer = Tri1bTransformer()
        test_df = pd.DataFrame(
            {
                "produce_the_chemical": ["yes", "no", "Yes"],
                "import_the_chemical": ["1", "0", "true"],
                "document_control_number": ["123", "456", "789"],
            }
        )
        result = transformer._standardize_activity_columns(test_df)
        assert result["produce_the_chemical"].tolist() == [True, False, True]
        assert result["import_the_chemical"].tolist() == [True, False, True]
        assert result["document_control_number"].tolist() == ["123", "456", "789"]

    def test_get_active_activities_returns_only_true(self):
        transformer = Tri1bTransformer()
        row = pd.Series(
            {
                "produce_the_chemical": True,
                "import_the_chemical": False,
                "on_site_use_of_the_chemical": True,
            }
        )
        with patch.object(
            transformer, "_activity_columns", ["produce_the_chemical", "import_the_chemical", "on_site_use_of_the_chemical"]
        ):
            result = transformer._get_active_activities(row)
        assert set(result) == {"produce_the_chemical", "on_site_use_of_the_chemical"}

    def test_expand_row_produces_human_readable_labels(self):
        transformer = Tri1bTransformer()
        row = pd.Series(
            {
                "document_control_number": 123,
                "produce_the_chemical": True,
                "import_the_chemical": False,
            }
        )
        with patch.object(transformer, "_activity_columns", ["produce_the_chemical", "import_the_chemical"]):
            result = transformer._expand_row(row)
        assert len(result) == 1
        assert result[0]["chemical_activity"] == "Produce the chemical"
        assert result[0]["document_control_number"] == 123

    def test_expand_row_drops_row_with_no_activities(self):
        transformer = Tri1bTransformer()
        row = pd.Series(
            {
                "document_control_number": 456,
                "produce_the_chemical": False,
            }
        )
        with patch.object(transformer, "_activity_columns", ["produce_the_chemical"]):
            result = transformer._expand_row(row)
        assert result == []

    def test_expand_activities_output_columns(self):
        transformer = Tri1bTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123, 456],
                "produce_the_chemical": ["yes", "no"],
                "import_the_chemical": ["no", "yes"],
            }
        )
        result = transformer._expand_activities(test_df)
        assert "document_control_number" in result.columns
        assert "chemical_activity" in result.columns
        assert len(result) == 2

    def test_transform_returns_only_relevant_columns(self):
        transformer = Tri1bTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123, 456],
                "produce_the_chemical": ["yes", "no"],
                "import_the_chemical": ["no", "yes"],
            }
        )
        result = transformer.transform(test_df)
        assert set(result.columns) == {"document_control_number", "chemical_activity"}

    def test_transform_drops_inactive_rows(self):
        transformer = Tri1bTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "produce_the_chemical": ["no"],
                "import_the_chemical": ["no"],
            }
        )
        result = transformer.transform(test_df)
        assert len(result) == 0

    def test_transform_expands_multiple_activities(self):
        transformer = Tri1bTransformer()
        test_df = pd.DataFrame(
            {
                "document_control_number": [123],
                "produce_the_chemical": ["yes"],
                "import_the_chemical": ["yes"],
            }
        )
        result = transformer.transform(test_df)
        assert len(result) == 2
        labels = set(result["chemical_activity"].tolist())
        assert labels == {"Produce the chemical", "Import the chemical"}
