# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import pandas as pd

from src.knowledge_graph.extract.smiles_extractor import SmilesExtractor


class TestSmilesExtractor:

    def test_singleton_pattern(self):
        extractor1 = SmilesExtractor()
        extractor2 = SmilesExtractor()
        assert extractor1 is extractor2

    def test_normalize_cas_number(self):
        extractor = SmilesExtractor()
        assert extractor._normalize_cas_number("71-43-2") == "71432"
        assert extractor._normalize_cas_number(" 108-88-3 ") == "108883"
        assert extractor._normalize_cas_number("7439-92-1") == "7439921"

    def test_smiles_from_cas_returns_none_for_empty(self):
        extractor = SmilesExtractor()
        assert extractor._smiles_from_cas("") is None
        assert extractor._smiles_from_cas(None) is None  # type: ignore[arg-type]

    @patch("src.knowledge_graph.extract.smiles_extractor.requests.get")
    def test_smiles_from_cas_uses_pubchem(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"PropertyTable": {"Properties": [{"CID": 5352425, "ConnectivitySMILES": "[Pb]"}]}}
        mock_get.return_value = mock_response

        extractor = SmilesExtractor()
        extractor._cas_cache.clear()
        result = extractor._smiles_from_cas("7439-92-1")

        assert result == "[Pb]"
        mock_get.assert_called_once()

    @patch("src.knowledge_graph.extract.smiles_extractor.requests.get")
    def test_smiles_from_cas_caches_result(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"PropertyTable": {"Properties": [{"CID": 887, "ConnectivitySMILES": "CO"}]}}
        mock_get.return_value = mock_response

        extractor = SmilesExtractor()
        extractor._cas_cache.clear()
        extractor._cache_hits = 0
        extractor._cache_misses = 0

        extractor._smiles_from_cas("67-56-1")
        extractor._smiles_from_cas("67-56-1")

        assert extractor._cache_hits == 1
        assert extractor._cache_misses == 1
        assert mock_get.call_count == 1

    @patch("src.knowledge_graph.extract.smiles_extractor.requests.get")
    def test_smiles_from_cas_returns_none_on_404(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response

        extractor = SmilesExtractor()
        extractor._cas_cache.clear()
        result = extractor._smiles_from_cas("INVALID-CAS")
        assert result is None

    @patch("src.knowledge_graph.extract.smiles_extractor.requests.get")
    def test_extract_smiles_delegates_to_smiles_from_cas(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"PropertyTable": {"Properties": [{"CID": 887, "ConnectivitySMILES": "CO"}]}}
        mock_get.return_value = mock_response

        extractor = SmilesExtractor()
        extractor._cas_cache.clear()
        result = extractor.extract_smiles("67-56-1")
        assert result == "CO"

    @patch("src.knowledge_graph.extract.smiles_extractor.requests.get")
    def test_process_chemicals_dataframe_adds_smiles_column(self, mock_get):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"PropertyTable": {"Properties": [{"CID": 887, "ConnectivitySMILES": "CO"}]}}
        mock_get.return_value = mock_response

        extractor = SmilesExtractor()
        extractor._cas_cache.clear()

        test_df = pd.DataFrame({"cas_number": ["67-56-1", "71-43-2"]})
        result = extractor.process_chemicals_dataframe(test_df)

        assert "canonical_smiles" in result.columns
        assert len(result) == len(test_df)

    def test_process_chemicals_dataframe_without_cas_column(self):
        extractor = SmilesExtractor()
        test_df = pd.DataFrame({"facility_name": ["A", "B"]})
        result = extractor.process_chemicals_dataframe(test_df)
        assert "canonical_smiles" in result.columns
        assert result["canonical_smiles"].isna().all()
