# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import pandas as pd

from src.knowledge_graph.extract.smiles_extractor import SmilesExtractor


class TestSmilesExtractor:

    def test_smiles_extractor_singleton(self):
        """Test that SmilesExtractor follows singleton pattern."""
        extractor1 = SmilesExtractor()
        extractor2 = SmilesExtractor()

        assert extractor1 is extractor2

    def test_normalize_chemical_name(self):
        """Test chemical name normalization."""
        extractor = SmilesExtractor()

        assert extractor._normalize_chemical_name("  Benzene  ") == "benzene"
        assert extractor._normalize_chemical_name("TOLUENE") == "toluene"

    def test_normalize_cas_number(self):
        """Test CAS number normalization."""
        extractor = SmilesExtractor()

        assert extractor._normalize_cas_number("71-43-2") == "71432"
        assert extractor._normalize_cas_number(" 108-88-3 ") == "108883"

    @patch("src.knowledge_graph.extract.smiles_extractor.Chem.MolFromSmiles")
    @patch("src.knowledge_graph.extract.smiles_extractor.Chem.MolToSmiles")
    def test_smiles_from_name_success(self, mock_mol_to_smiles, mock_mol_from_smiles):
        """Test successful SMILES extraction from chemical name."""
        # Mock RDKit behavior
        mock_mol = MagicMock()
        mock_mol_from_smiles.return_value = mock_mol
        mock_mol_to_smiles.return_value = "C1=CC=CC=C1"

        extractor = SmilesExtractor()
        result = extractor._smiles_from_name("benzene")

        assert result == "C1=CC=CC=C1"
        assert extractor._cache_hits == 0
        assert extractor._cache_misses == 1

    def test_smiles_from_name_caching(self):
        """Test that SMILES extraction uses caching."""
        extractor = SmilesExtractor()

        # First call should be a cache miss
        result1 = extractor._smiles_from_name("test_chemical")

        # Second call should be a cache hit
        result2 = extractor._smiles_from_name("test_chemical")

        assert result1 == result2
        assert extractor._cache_hits == 1
        assert extractor._cache_misses == 1

    def test_smiles_from_cas_placeholder(self):
        """Test that CAS to SMILES conversion returns None (placeholder)."""
        extractor = SmilesExtractor()

        result = extractor._smiles_from_cas("71-43-2")
        assert result is None

    def test_extract_smiles_invalid_type(self):
        """Test extract_smiles with invalid identifier type."""
        extractor = SmilesExtractor()

        result = extractor.extract_smiles("benzene", "invalid_type")
        assert result is None

    def test_process_chemicals_dataframe(self):
        """Test processing a DataFrame with chemical data."""
        extractor = SmilesExtractor()

        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "chemical_name": ["benzene", "toluene"],
                "cas_number": ["71-43-2", "108-88-3"],
                "other_column": ["value1", "value2"],
            }
        )

        result_df = extractor.process_chemicals_dataframe(test_df)

        # Check that canonical_smiles column was added
        assert "canonical_smiles" in result_df.columns
        assert len(result_df) == len(test_df)
        assert list(result_df.columns) == list(test_df.columns) + ["canonical_smiles"]

    def test_process_chemicals_dataframe_no_chemical_columns(self):
        """Test processing DataFrame with no chemical columns."""
        extractor = SmilesExtractor()

        # Create test DataFrame without chemical columns
        test_df = pd.DataFrame({"facility_name": ["Facility A", "Facility B"], "year": [2020, 2021]})

        result_df = extractor.process_chemicals_dataframe(test_df)

        # Should still add canonical_smiles column with None values
        assert "canonical_smiles" in result_df.columns
        assert bool(result_df["canonical_smiles"].isna().all())
