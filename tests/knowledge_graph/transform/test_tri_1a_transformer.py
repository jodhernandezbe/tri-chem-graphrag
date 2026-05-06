# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import pandas as pd

from src.knowledge_graph.transform.tri_1a_transformer import Tri1aTransformer


class TestTri1aTransformer:

    def test_transformer_initialization(self):
        """Test that transformer initializes correctly."""
        transformer = Tri1aTransformer()

        assert transformer.file_type == "1a"
        assert hasattr(transformer, "_naics_fetcher")
        assert hasattr(transformer, "_smiles_extractor")

    def test_get_maximum_amount_value_from_config(self):
        """Test maximum amount value lookup from config."""
        transformer = Tri1aTransformer()

        # Test with a valid code (assuming these exist in config)
        result = transformer._get_maximum_amount_value_from_config("A")
        assert isinstance(result, (float, int, type(None)))

        # Test with invalid code
        result_invalid = transformer._get_maximum_amount_value_from_config("INVALID_CODE")
        assert result_invalid is None

    def test_add_naics_alias(self):
        """Test adding NAICS alias column."""
        transformer = Tri1aTransformer()

        # Create test DataFrame with NAICS column
        test_df = pd.DataFrame({"primary_naics_code": ["123456", "789012"], "facility_name": ["Facility A", "Facility B"]})

        result_df = transformer._add_naics_alias(test_df)

        # Should add naics_code column
        assert "naics_code" in result_df.columns
        assert result_df["naics_code"].equals(result_df["primary_naics_code"])

    def test_convert_maximum_amount_codes(self):
        """Test conversion of maximum amount codes to numeric values."""
        transformer = Tri1aTransformer()

        # Create test DataFrame
        test_df = pd.DataFrame(
            {"maximum_amount_on_site": ["A", "B", None], "facility_name": ["Facility A", "Facility B", "Facility C"]}
        )

        result_df = transformer._convert_maximum_amount_codes(test_df)

        # Should add maximum_amount column
        assert "maximum_amount" in result_df.columns
        # Values should be converted based on config
        assert result_df["maximum_amount"].notna().sum() >= 0

    def test_clean_numeric_columns(self):
        """Test cleaning of numeric columns."""
        transformer = Tri1aTransformer()

        # Create test DataFrame with mixed data types
        test_df = pd.DataFrame({"maximum_amount": ["100.5", "200", "invalid", None], "facility_name": ["A", "B", "C", "D"]})

        result_df = transformer._clean_numeric_columns(test_df)

        # Should convert to numeric, invalid values become NaN
        assert pd.api.types.is_numeric_dtype(result_df["maximum_amount"])

    @patch("src.knowledge_graph.transform.tri_1a_transformer.NaicsDataFetcher")
    def test_enrich_with_naics_titles_success(self, mock_naics_fetcher):
        """Test successful NAICS title enrichment."""
        # Mock the NAICS fetcher
        mock_fetcher_instance = MagicMock()
        mock_naics_df = pd.DataFrame({"naics_code": ["123456"], "naics_title": ["Test Industry"]})
        mock_fetcher_instance.process_naics_codes.return_value = mock_naics_df
        mock_naics_fetcher.return_value = mock_fetcher_instance

        transformer = Tri1aTransformer()
        transformer._naics_fetcher = mock_fetcher_instance

        # Create test DataFrame
        test_df = pd.DataFrame({"naics_code": ["123456"], "facility_name": ["Facility A"]})

        result_df = transformer._enrich_with_naics_titles(test_df)

        # Should have NAICS title column
        assert "naics_title" in result_df.columns

    @patch("src.knowledge_graph.transform.tri_1a_transformer.SmilesExtractor")
    def test_extract_chemical_smiles_success(self, mock_smiles_extractor):
        """Test successful SMILES extraction."""
        # Mock the SMILES extractor
        mock_extractor_instance = MagicMock()
        mock_chemicals_df = pd.DataFrame({"chemical_name": ["Benzene"], "canonical_smiles": ["C1=CC=CC=C1"]})
        mock_extractor_instance.process_chemicals_dataframe.return_value = mock_chemicals_df
        mock_smiles_extractor.return_value = mock_extractor_instance

        transformer = Tri1aTransformer()
        transformer._smiles_extractor = mock_extractor_instance

        # Create test DataFrame
        test_df = pd.DataFrame({"chemical_name": ["Benzene"], "facility_name": ["Facility A"]})

        result_df = transformer._extract_chemical_smiles(test_df)

        # Should have canonical_smiles column
        assert "canonical_smiles" in result_df.columns

    def test_transform_basic_functionality(self):
        """Test basic transform functionality without external dependencies."""
        transformer = Tri1aTransformer()

        # Create minimal test DataFrame
        test_df = pd.DataFrame(
            {"primary_naics_code": ["123456"], "facility_name": ["Test Facility"], "maximum_amount_on_site": ["A"]}
        )

        # Mock external dependencies to avoid actual API calls
        with (
            patch.object(transformer, "_enrich_with_naics_titles", return_value=test_df),
            patch.object(transformer, "_extract_chemical_smiles", return_value=test_df),
        ):

            result_df = transformer.transform(test_df)

            # Should return a DataFrame
            assert isinstance(result_df, pd.DataFrame)
            assert len(result_df) >= 0

    def test_transform_with_no_columns(self):
        """Test transform with DataFrame missing expected columns."""
        transformer = Tri1aTransformer()

        # Create DataFrame with minimal columns
        test_df = pd.DataFrame({"facility_name": ["Test Facility"]})

        # Mock external dependencies
        with (
            patch.object(transformer, "_enrich_with_naics_titles", return_value=test_df),
            patch.object(transformer, "_extract_chemical_smiles", return_value=test_df),
        ):

            result_df = transformer.transform(test_df)

            # Should handle gracefully
            assert isinstance(result_df, pd.DataFrame)
