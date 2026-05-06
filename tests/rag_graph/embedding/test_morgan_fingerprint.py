# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import pytest

from src.rag_graph.embedding.morgan_fingerprint import MorganFingerprintEmbeddings


class TestMorganFingerprintEmbeddings:

    def test_initialization_default_params(self):
        """Test MorganFingerprintEmbeddings initializes with default parameters."""
        embeddings = MorganFingerprintEmbeddings()

        assert embeddings.radius == 2
        assert embeddings.n_bits == 2048
        assert embeddings.model_name == "morgan_fingerprint"
        assert embeddings.device == "cpu"

    def test_initialization_custom_params(self):
        """Test MorganFingerprintEmbeddings initializes with custom parameters."""
        embeddings = MorganFingerprintEmbeddings(radius=3, n_bits=1024)

        assert embeddings.radius == 3
        assert embeddings.n_bits == 1024
        assert embeddings.model_name == "morgan_fingerprint"
        assert embeddings.device == "cpu"

    @patch("src.rag_graph.embedding.morgan_fingerprint.Chem.MolFromSmiles")
    @patch("src.rag_graph.embedding.morgan_fingerprint.rdMolDescriptors.GetMorganFingerprintAsBitVect")
    def test_get_fp_valid_smiles(self, mock_fingerprint, mock_mol_from_smiles):
        """Test fingerprint generation for valid SMILES."""
        embeddings = MorganFingerprintEmbeddings()

        # Mock RDKit components
        mock_mol = MagicMock()
        mock_mol_from_smiles.return_value = mock_mol

        mock_bitvect = MagicMock()
        mock_bitvect.__iter__ = lambda x: iter([1, 0, 1, 0])
        mock_fingerprint.return_value = mock_bitvect

        result = embeddings._get_fp("CCO")

        assert result == [1.0, 0.0, 1.0, 0.0]
        mock_mol_from_smiles.assert_called_once_with("CCO")
        mock_fingerprint.assert_called_once_with(mock_mol, 2, nBits=2048)

    @patch("src.rag_graph.embedding.morgan_fingerprint.Chem.MolFromSmiles")
    def test_get_fp_invalid_smiles(self, mock_mol_from_smiles):
        """Test fingerprint generation for invalid SMILES."""
        embeddings = MorganFingerprintEmbeddings()

        mock_mol_from_smiles.return_value = None

        result = embeddings._get_fp("INVALID")

        assert result == [0.0] * 2048
        mock_mol_from_smiles.assert_called_once_with("INVALID")

    def test_embed_documents(self):
        """Test embed_documents method."""
        embeddings = MorganFingerprintEmbeddings()

        with patch.object(embeddings, "_get_fp") as mock_get_fp:
            mock_get_fp.side_effect = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

            result = embeddings.embed_documents(["CCO", "CC"])

            assert len(result) == 2
            assert result[0] == [1.0, 0.0, 1.0]
            assert result[1] == [0.0, 1.0, 0.0]

    def test_embed_documents_empty_list(self):
        """Test embed_documents with empty list."""
        embeddings = MorganFingerprintEmbeddings()

        result = embeddings.embed_documents([])

        assert result == []

    def test_embed_query(self):
        """Test embed_query method."""
        embeddings = MorganFingerprintEmbeddings()

        with patch.object(embeddings, "_get_fp") as mock_get_fp:
            mock_get_fp.return_value = [1.0, 0.0, 1.0]

            result = embeddings.embed_query("CCO")

            assert result == [1.0, 0.0, 1.0]
            mock_get_fp.assert_called_once_with("CCO")

    def test_get_embedding(self):
        """Test _get_embedding method."""
        embeddings = MorganFingerprintEmbeddings()

        with patch.object(embeddings, "_get_fp") as mock_get_fp:
            mock_get_fp.side_effect = [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

            result = embeddings._get_embedding(["CCO", "CC"])

            assert len(result) == 2
            assert result[0] == [1.0, 0.0, 1.0]
            assert result[1] == [0.0, 1.0, 0.0]

    @patch("src.rag_graph.embedding.morgan_fingerprint.Chem.MolFromSmiles")
    @patch("src.rag_graph.embedding.morgan_fingerprint.rdMolDescriptors.GetMorganFingerprintAsBitVect")
    def test_custom_radius_and_bits(self, mock_fingerprint, mock_mol_from_smiles):
        """Test that custom radius and n_bits are used correctly."""
        embeddings = MorganFingerprintEmbeddings(radius=3, n_bits=1024)

        mock_mol = MagicMock()
        mock_mol_from_smiles.return_value = mock_mol

        mock_bitvect = MagicMock()
        mock_bitvect.__iter__ = lambda x: iter([1] * 1024)
        mock_fingerprint.return_value = mock_bitvect

        result = embeddings._get_fp("CCO")

        assert len(result) == 1024
        mock_fingerprint.assert_called_once_with(mock_mol, 3, nBits=1024)

    def test_chemical_specific_smiles(self):
        """Test embedding generation with chemical-specific SMILES strings."""
        embeddings = MorganFingerprintEmbeddings()

        chemical_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"]  # ethanol  # acetic acid  # benzene  # triethylamine

        with patch.object(embeddings, "_get_fp") as mock_get_fp:
            mock_get_fp.return_value = [0.1] * 2048

            result = embeddings.embed_documents(chemical_smiles)

            assert len(result) == len(chemical_smiles)
            assert all(len(emb) == 2048 for emb in result)
            assert mock_get_fp.call_count == len(chemical_smiles)
