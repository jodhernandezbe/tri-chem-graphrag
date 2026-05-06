# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import patch

import pandas as pd

from src.knowledge_graph.transform.tri_2a_transformer import Tri2aTransformer


class TestTri2aTransformer:

    def test_transformer_initialization(self):
        """Test that transformer initializes correctly."""
        transformer = Tri2aTransformer()

        assert transformer.file_type == "2a"
        assert hasattr(transformer, "_reduction_codes_considered")
        assert isinstance(transformer._reduction_codes_considered, set)

    def test_get_reduction_value_from_config(self):
        """Test reduction value lookup from config."""
        transformer = Tri2aTransformer()

        # Test with a valid code (assuming these exist in config)
        result = transformer._get_reduction_value_from_config("A")
        assert isinstance(result, (float, int, type(None)))

        # Test with invalid code
        result_invalid = transformer._get_reduction_value_from_config("INVALID_CODE")
        assert result_invalid is None

    def test_is_considered_reduction_code(self):
        """Test validation of reduction codes."""
        transformer = Tri2aTransformer()

        # Test with None/NaN values
        assert not transformer._is_considered_reduction_code(None)
        assert not transformer._is_considered_reduction_code(pd.NA)

        # Test with non-scalar values
        assert not transformer._is_considered_reduction_code(pd.Series([1, 2, 3]))

        # Test with valid string codes (if any exist in config)
        valid_codes = list(transformer._reduction_codes_considered)
        if valid_codes:
            assert transformer._is_considered_reduction_code(valid_codes[0])

        # Test with invalid code
        assert not transformer._is_considered_reduction_code("INVALID_CODE")

    def test_get_reduction_code_from_row(self):
        """Test extraction of reduction code from row."""
        transformer = Tri2aTransformer()

        # Create test row
        test_row = pd.Series(
            {"source_reduction_activity_code_a": "A", "source_reduction_activity_code_b": "INVALID", "other_column": "value"}
        )

        # Test with existing column
        result = transformer._get_reduction_code_from_row(test_row, "source_reduction_activity_code_a")
        # Result depends on whether 'A' is in the considered codes
        assert isinstance(result, (str, type(None)))

        # Test with non-existing column
        result_none = transformer._get_reduction_code_from_row(test_row, "nonexistent_column")
        assert result_none is None

    def test_get_column_value_safely(self):
        """Test safe column value extraction."""
        transformer = Tri2aTransformer()

        # Create test row
        test_row = pd.Series({"column1": "value1", "column2": None, "column3": pd.NA})

        # Test with existing column
        result = transformer._get_column_value_safely(test_row, "column1")
        assert result == "value1"

        # Test with None value
        result_none = transformer._get_column_value_safely(test_row, "column2")
        assert result_none is None

        # Test with NA value
        result_na = transformer._get_column_value_safely(test_row, "column3")
        assert result_na is None

        # Test with non-existing column
        result_missing = transformer._get_column_value_safely(test_row, "nonexistent")
        assert result_missing is None

        # Test with None column name
        result_none_col = transformer._get_column_value_safely(test_row, None)
        assert result_none_col is None

    def test_create_reduction_row(self):
        """Test creation of expanded reduction row."""
        transformer = Tri2aTransformer()

        # Create base row
        base_row = pd.Series({"facility_name": "Test Facility", "year": 2020})

        # Create reduction data
        reduction_data = {"code": "A", "description": "Test reduction", "annual_reduction": "B"}

        result_row = transformer._create_reduction_row(base_row, reduction_data)

        # Should have base columns plus new reduction columns
        assert result_row["facility_name"] == "Test Facility"
        assert result_row["source_reduction_code"] == "A"
        assert result_row["source_reduction_description"] == "Test reduction"
        assert "annual_reduction" in result_row.index

    def test_create_empty_reduction_row(self):
        """Test creation of empty reduction row."""
        transformer = Tri2aTransformer()

        # Create base row
        base_row = pd.Series({"facility_name": "Test Facility", "year": 2020})

        result_row = transformer._create_empty_reduction_row(base_row)

        # Should have base columns plus None reduction columns
        assert result_row["facility_name"] == "Test Facility"
        assert result_row["source_reduction_code"] is None
        assert result_row["source_reduction_description"] is None
        assert result_row["annual_reduction"] is None

    def test_extract_reduction_activities_from_row(self):
        """Test extraction of reduction activities from a row."""
        transformer = Tri2aTransformer()

        # Create test row with some reduction codes
        test_row = pd.Series(
            {
                "facility_name": "Test Facility",
                "source_reduction_activity_code_a": "A",
                "source_reduction_activity_code_b": "B",
                "source_reduction_activity_description_a": "Description A",
                "est_annual_reduction_code_a": "C",
            }
        )

        activities = transformer._extract_reduction_activities_from_row(test_row)

        # Should return list of activities
        assert isinstance(activities, list)
        # Number of activities depends on valid codes in config

    def test_expand_single_row_with_activities(self):
        """Test expansion of a single row with reduction activities."""
        transformer = Tri2aTransformer()

        # Create test row
        test_row = pd.Series({"facility_name": "Test Facility", "year": 2020})

        # Mock the activity extraction to return some activities
        mock_activities = [{"code": "A", "description": "Activity A", "annual_reduction": "B"}]

        with patch.object(transformer, "_extract_reduction_activities_from_row", return_value=mock_activities):
            expanded_rows = transformer._expand_single_row(test_row)

            assert isinstance(expanded_rows, list)
            assert len(expanded_rows) == 1
            assert expanded_rows[0]["source_reduction_code"] == "A"

    def test_expand_single_row_without_activities(self):
        """Test expansion of a single row without reduction activities."""
        transformer = Tri2aTransformer()

        # Create test row
        test_row = pd.Series({"facility_name": "Test Facility", "year": 2020})

        # Mock the activity extraction to return no activities
        with patch.object(transformer, "_extract_reduction_activities_from_row", return_value=[]):
            expanded_rows = transformer._expand_single_row(test_row)

            assert isinstance(expanded_rows, list)
            assert len(expanded_rows) == 1
            assert expanded_rows[0]["source_reduction_code"] is None

    def test_expand_reduction_activities(self):
        """Test expansion of entire DataFrame."""
        transformer = Tri2aTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame({"facility_name": ["Facility A", "Facility B"], "year": [2020, 2021]})

        # Mock single row expansion
        mock_expanded_row = pd.Series(
            {
                "facility_name": "Test",
                "year": 2020,
                "source_reduction_code": "A",
                "source_reduction_description": "Test",
                "annual_reduction": None,
            }
        )

        with patch.object(transformer, "_expand_single_row", return_value=[mock_expanded_row]):
            result_df = transformer._expand_reduction_activities(test_df)

            assert isinstance(result_df, pd.DataFrame)
            # Should have same number of rows as mock returns (2 original rows * 1 expanded each)
            assert len(result_df) == 2

    def test_add_naics_alias(self):
        """Test adding NAICS alias column."""
        transformer = Tri2aTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame({"primary_naics_code": ["123456", "789012"], "facility_name": ["Facility A", "Facility B"]})

        result_df = transformer._add_naics_alias(test_df)

        # Should add naics_code column
        assert "naics_code" in result_df.columns
        assert result_df["naics_code"].equals(result_df["primary_naics_code"])

    def test_clean_numeric_columns(self):
        """Test cleaning of numeric columns."""
        transformer = Tri2aTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame({"annual_reduction": ["100.5", "200", "invalid", None], "facility_name": ["A", "B", "C", "D"]})

        result_df = transformer._clean_numeric_columns(test_df)

        # Should convert to numeric, invalid values become NaN
        assert pd.api.types.is_numeric_dtype(result_df["annual_reduction"])

    def test_transform_basic_functionality(self):
        """Test basic transform functionality."""
        transformer = Tri2aTransformer()

        # Create minimal test DataFrame
        test_df = pd.DataFrame({"primary_naics_code": ["123456"], "facility_name": ["Test Facility"]})

        result_df = transformer.transform(test_df)

        # Should return a DataFrame with expanded rows
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= len(test_df)  # Should have at least same number of rows (possibly more due to expansion)
        assert "naics_code" in result_df.columns
