# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.rag_graph.embedding.hybrid import HybridMolecularEmbeddings


class TestHybridMolecularEmbeddings:

    def test_initialization_default_params(self):
        """Test HybridMolecularEmbeddings initializes with default parameters."""
        embeddings = HybridMolecularEmbeddings()

        assert embeddings.radius == 2
        assert embeddings.n_bits == 1024
        assert hasattr(embeddings, "tokenizer")
        assert hasattr(embeddings, "model")

    def test_initialization_custom_params(self):
        """Test HybridMolecularEmbeddings initializes with custom parameters."""
        custom_model = "custom/model"
        custom_radius = 3
        custom_bits = 2048

        with (
            patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer"),
            patch("src.rag_graph.embedding.base_embeddings.AutoModel"),
        ):

            embeddings = HybridMolecularEmbeddings(model_name=custom_model, radius=custom_radius, n_bits=custom_bits)

            assert embeddings.radius == custom_radius
            assert embeddings.n_bits == custom_bits

    @patch("src.rag_graph.embedding.hybrid.Chem.MolFromSmiles")
    @patch("src.rag_graph.embedding.hybrid.rdMolDescriptors.GetMorganFingerprintAsBitVect")
    @patch("src.rag_graph.embedding.hybrid.DataStructs.ConvertToNumpyArray")
    def test_get_morgan_valid_smiles(self, mock_convert, mock_fingerprint, mock_mol_from_smiles):
        """Test Morgan fingerprint generation for valid SMILES."""
        embeddings = HybridMolecularEmbeddings()

        # Mock RDKit components
        mock_mol = MagicMock()
        mock_mol_from_smiles.return_value = mock_mol

        mock_fp = MagicMock()
        mock_fingerprint.return_value = mock_fp

        def mock_convert_side_effect(fp, arr):
            # Simulate converting fingerprint to numpy array
            # The actual RDKit function modifies arr in place by extending it
            test_data = np.ones(embeddings.n_bits, dtype=np.int8)
            arr.resize(embeddings.n_bits, refcheck=False)
            arr[:] = test_data

        mock_convert.side_effect = mock_convert_side_effect

        result = embeddings._get_morgan("CCO")  # ethanol

        assert isinstance(result, np.ndarray)
        assert len(result) == embeddings.n_bits
        mock_mol_from_smiles.assert_called_once_with("CCO")

    @patch("src.rag_graph.embedding.hybrid.Chem.MolFromSmiles")
    def test_get_morgan_invalid_smiles(self, mock_mol_from_smiles):
        """Test Morgan fingerprint generation for invalid SMILES."""
        embeddings = HybridMolecularEmbeddings()

        mock_mol_from_smiles.return_value = None  # Invalid SMILES

        result = embeddings._get_morgan("invalid_smiles")

        assert isinstance(result, np.ndarray)
        assert len(result) == embeddings.n_bits
        assert np.all(result == 0)  # Should be zero array

    def test_get_deberta(self):
        """Test DeBERTa embedding generation."""
        embeddings = HybridMolecularEmbeddings()

        # Mock tokenizer and model
        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(1, 3, 768)  # (batch, seq, hidden)
                mock_model.return_value = mock_output

                result = embeddings._get_deberta("CCO")

                assert isinstance(result, np.ndarray)
                assert len(result) == 768  # DeBERTa hidden size

    def test_embed_documents(self):
        """Test embedding multiple SMILES documents."""
        embeddings = HybridMolecularEmbeddings()

        smiles_list = ["CCO", "CC", "CCC"]

        # Mock both Morgan and DeBERTa methods
        with patch.object(embeddings, "_get_morgan") as mock_morgan:
            mock_morgan.return_value = np.ones(1024)  # Mock Morgan fingerprint

            with patch.object(embeddings, "_get_deberta") as mock_deberta:
                mock_deberta.return_value = np.ones(768)  # Mock DeBERTa embedding

                result = embeddings.embed_documents(smiles_list)

                assert len(result) == len(smiles_list)
                assert len(result[0]) == 1024 + 768  # Combined length
                assert all(isinstance(emb, list) for emb in result)

    def test_embed_query(self):
        """Test embedding a single SMILES query."""
        embeddings = HybridMolecularEmbeddings()

        # Mock _get_embedding to return predictable result
        with patch.object(embeddings, "_get_embedding") as mock_get_embedding:
            mock_get_embedding.return_value = [[0.1] * 1792]  # 1024 + 768

            result = embeddings.embed_query("CCO")

            assert result == [0.1] * 1792
            mock_get_embedding.assert_called_once_with(["CCO"])

    def test_embed_empty_list(self):
        """Test embedding empty SMILES list."""
        embeddings = HybridMolecularEmbeddings()

        result = embeddings.embed_documents([])

        assert result == []

    def test_hybrid_concatenation(self):
        """Test that Morgan and DeBERTa embeddings are properly concatenated."""
        embeddings = HybridMolecularEmbeddings()

        morgan_size = embeddings.n_bits
        deberta_size = 768  # Typical DeBERTa size

        with patch.object(embeddings, "_get_morgan") as mock_morgan:
            mock_morgan.return_value = np.ones(morgan_size) * 0.5

            with patch.object(embeddings, "_get_deberta") as mock_deberta:
                mock_deberta.return_value = np.ones(deberta_size) * 0.8

                result = embeddings.embed_documents(["CCO"])

                # Check concatenation
                assert len(result[0]) == morgan_size + deberta_size

                # Check that first part is Morgan (0.5) and second part is DeBERTa (0.8)
                morgan_part = result[0][:morgan_size]
                deberta_part = result[0][morgan_size:]

                assert all(abs(x - 0.5) < 1e-6 for x in morgan_part)
                assert all(abs(x - 0.8) < 1e-6 for x in deberta_part)

    def test_model_eval_mode(self):
        """Test that model is set to evaluation mode during initialization."""
        with patch("src.rag_graph.embedding.base_embeddings.AutoModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            HybridMolecularEmbeddings()

            mock_model.eval.assert_called_once()

    def test_tokenizer_parameters(self):
        """Test that tokenizer is called with correct parameters."""
        embeddings = HybridMolecularEmbeddings()

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(1, 3, 768)
                mock_model.return_value = mock_output

                embeddings._get_deberta("CCO")

                mock_tokenizer.assert_called_once_with(
                    "CCO",
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

    def test_no_grad_context(self):
        """Test that model inference runs in no_grad context."""
        embeddings = HybridMolecularEmbeddings()

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(1, 3, 768)
                mock_model.return_value = mock_output

                with patch("torch.no_grad") as mock_no_grad:
                    mock_no_grad.return_value.__enter__ = MagicMock()
                    mock_no_grad.return_value.__exit__ = MagicMock()

                    embeddings._get_deberta("CCO")

                    mock_no_grad.assert_called_once()
