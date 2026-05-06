# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import patch

import pandas as pd

from src.knowledge_graph.transform.tri_1b_transformer import Tri1bTransformer


class TestTri1bTransformer:

    def test_transformer_initialization(self):
        """Test that transformer initializes correctly."""
        transformer = Tri1bTransformer()

        assert transformer.file_type == "1b"
        assert hasattr(transformer, "_activity_columns")
        assert hasattr(transformer, "_yes_values")
        assert isinstance(transformer._yes_values, set)

    def test_get_activity_columns_from_config(self):
        """Test getting activity columns from config."""
        transformer = Tri1bTransformer()

        activity_columns = transformer._get_activity_columns_from_config()

        # Should return a list of column names
        assert isinstance(activity_columns, list)
        # Should exclude document_control_number
        assert "document_control_number" not in activity_columns

    def test_is_yes_value(self):
        """Test validation of 'yes' values."""
        transformer = Tri1bTransformer()

        # Test valid yes values
        assert transformer._is_yes_value("yes")
        assert transformer._is_yes_value("Yes")
        assert transformer._is_yes_value("YES")
        assert transformer._is_yes_value("y")
        assert transformer._is_yes_value("Y")
        assert transformer._is_yes_value("1")
        assert transformer._is_yes_value("true")
        assert transformer._is_yes_value("True")

        # Test invalid values
        assert not transformer._is_yes_value("no")
        assert not transformer._is_yes_value("No")
        assert not transformer._is_yes_value("0")
        assert not transformer._is_yes_value("false")
        assert not transformer._is_yes_value("")
        assert not transformer._is_yes_value(None)
        assert not transformer._is_yes_value(pd.NA)

    def test_standardize_single_column(self):
        """Test standardization of a single column."""
        transformer = Tri1bTransformer()

        # Create test series
        test_series = pd.Series(["yes", "no", "Yes", "1", "0", None, "true", "false"])

        result = transformer._standardize_single_column(test_series)

        # Should convert to boolean values
        expected = [True, False, True, True, False, False, True, False]
        assert result.tolist() == expected

    def test_standardize_activity_columns(self):
        """Test standardization of activity columns in DataFrame."""
        transformer = Tri1bTransformer()

        # Create test DataFrame with activity columns
        test_df = pd.DataFrame(
            {
                "produce_the_chemical": ["yes", "no", "Yes"],
                "import_the_chemical": ["1", "0", "true"],
                "facility_name": ["A", "B", "C"],  # Non-activity column
                "document_control_number": ["123", "456", "789"],  # Should be excluded
            }
        )

        result_df = transformer._standardize_activity_columns(test_df)

        # Activity columns should be converted to boolean
        assert result_df["produce_the_chemical"].dtype == bool
        assert result_df["import_the_chemical"].dtype == bool

        # Non-activity columns should remain unchanged (string type)
        assert pd.api.types.is_string_dtype(result_df["facility_name"])
        assert pd.api.types.is_string_dtype(result_df["document_control_number"])

    def test_get_active_activities_for_row(self):
        """Test extraction of active activities from a row."""
        transformer = Tri1bTransformer()

        # Create test row with some True activities
        test_row = pd.Series(
            {
                "produce_the_chemical": True,
                "import_the_chemical": False,
                "on_site_use_of_the_chemical": True,
                "facility_name": "Test Facility",
            }
        )

        # Mock activity columns to match test data
        with patch.object(
            transformer, "_activity_columns", ["produce_the_chemical", "import_the_chemical", "on_site_use_of_the_chemical"]
        ):
            active_activities = transformer._get_active_activities_for_row(test_row)

            # Should return activities that are True
            expected_activities = ["produce_the_chemical", "on_site_use_of_the_chemical"]
            assert set(active_activities) == set(expected_activities)

    def test_create_expanded_row(self):
        """Test creation of expanded row for a specific activity."""
        transformer = Tri1bTransformer()

        # Create base row
        base_row = pd.Series(
            {"facility_name": "Test Facility", "year": 2020, "produce_the_chemical": True, "import_the_chemical": False}
        )

        result_row = transformer._create_expanded_row(base_row, "produce_the_chemical")

        # Should have base columns plus activity column
        assert result_row["facility_name"] == "Test Facility"
        assert result_row["year"] == 2020
        assert result_row["chemical_activity"] == "produce_the_chemical"

    def test_create_empty_activity_row(self):
        """Test creation of empty activity row."""
        transformer = Tri1bTransformer()

        # Create base row
        base_row = pd.Series({"facility_name": "Test Facility", "year": 2020})

        result_row = transformer._create_empty_activity_row(base_row)

        # Should have base columns plus None activity
        assert result_row["facility_name"] == "Test Facility"
        assert result_row["year"] == 2020
        assert result_row["chemical_activity"] is None

    def test_expand_single_row_with_activities(self):
        """Test expansion of a single row with active activities."""
        transformer = Tri1bTransformer()

        # Create test row
        test_row = pd.Series({"facility_name": "Test Facility", "produce_the_chemical": True, "import_the_chemical": False})

        # Mock the active activities method
        with patch.object(transformer, "_get_active_activities_for_row", return_value=["produce_the_chemical"]):
            expanded_rows = transformer._expand_single_row(test_row)

            assert isinstance(expanded_rows, list)
            assert len(expanded_rows) == 1
            assert expanded_rows[0]["chemical_activity"] == "produce_the_chemical"

    def test_expand_single_row_without_activities(self):
        """Test expansion of a single row without active activities."""
        transformer = Tri1bTransformer()

        # Create test row
        test_row = pd.Series({"facility_name": "Test Facility", "produce_the_chemical": False, "import_the_chemical": False})

        # Mock the active activities method to return empty list
        with patch.object(transformer, "_get_active_activities_for_row", return_value=[]):
            expanded_rows = transformer._expand_single_row(test_row)

            assert isinstance(expanded_rows, list)
            assert len(expanded_rows) == 1
            assert expanded_rows[0]["chemical_activity"] is None

    def test_expansion_workflow_in_transform(self):
        """Test that the expansion workflow works in the transform method."""
        transformer = Tri1bTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "primary_naics_code": ["123456", "789012"],
                "facility_name": ["Facility A", "Facility B"],
                "produce_the_chemical": ["yes", "no"],
                "import_the_chemical": ["no", "yes"],
            }
        )

        result_df = transformer.transform(test_df)

        # Should return a DataFrame with expanded rows
        assert isinstance(result_df, pd.DataFrame)
        # Should have at least the same number of rows (could be more due to expansion)
        assert len(result_df) >= len(test_df)
        assert "naics_code" in result_df.columns
        assert "chemical_activity" in result_df.columns

    def test_add_naics_alias(self):
        """Test adding NAICS alias column."""
        transformer = Tri1bTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame({"primary_naics_code": ["123456", "789012"], "facility_name": ["Facility A", "Facility B"]})

        result_df = transformer._add_naics_alias(test_df)

        # Should add naics_code column
        assert "naics_code" in result_df.columns
        assert result_df["naics_code"].equals(result_df["primary_naics_code"])

    def test_transform_basic_functionality(self):
        """Test basic transform functionality."""
        transformer = Tri1bTransformer()

        # Create minimal test DataFrame
        test_df = pd.DataFrame(
            {"primary_naics_code": ["123456"], "facility_name": ["Test Facility"], "produce_the_chemical": ["yes"]}
        )

        result_df = transformer.transform(test_df)

        # Should return a DataFrame with expanded rows
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) >= len(test_df)  # Should have at least same number of rows
        assert "naics_code" in result_df.columns
        assert "chemical_activity" in result_df.columns
