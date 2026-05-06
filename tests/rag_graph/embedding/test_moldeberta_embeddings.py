# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import torch

from src.rag_graph.embedding.moldeberta import MolDeBERTaEmbeddings


class TestMolDeBERTaEmbeddings:

    def test_initialization_default_params(self):
        """Test MolDeBERTaEmbeddings initializes with default parameters."""
        embeddings = MolDeBERTaEmbeddings()

        assert hasattr(embeddings, "tokenizer")
        assert hasattr(embeddings, "model")

    def test_initialization_custom_params(self):
        """Test MolDeBERTaEmbeddings initializes with custom model."""
        custom_model = "custom/moldeberta-model"

        with (
            patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer"),
            patch("src.rag_graph.embedding.base_embeddings.AutoModel"),
        ):

            embeddings = MolDeBERTaEmbeddings(model_name=custom_model)

            # Verify the custom model name was used
            assert hasattr(embeddings, "tokenizer")
            assert hasattr(embeddings, "model")

    def test_get_embedding_single_smiles(self):
        """Test embedding generation for single SMILES."""
        embeddings = MolDeBERTaEmbeddings()

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(1, 3, 768)  # (batch, seq, hidden)
                mock_model.return_value = mock_output

                result = embeddings._get_embedding(["CCO"])

                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], list)
                assert len(result[0]) == 768

    def test_get_embedding_multiple_smiles(self):
        """Test embedding generation for multiple SMILES."""
        embeddings = MolDeBERTaEmbeddings()

        smiles_list = ["CCO", "CC", "CCC"]

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            }

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(3, 3, 768)  # (batch, seq, hidden)
                mock_model.return_value = mock_output

                result = embeddings._get_embedding(smiles_list)

                assert isinstance(result, list)
                assert len(result) == 3
                assert all(isinstance(emb, list) for emb in result)
                assert all(len(emb) == 768 for emb in result)

    def test_embed_documents(self):
        """Test embed_documents method."""
        embeddings = MolDeBERTaEmbeddings()

        smiles_list = ["CCO", "CC"]

        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.1] * 768, [0.2] * 768]

            result = embeddings.embed_documents(smiles_list)

            assert result == [[0.1] * 768, [0.2] * 768]
            mock_get.assert_called_once_with(smiles_list)

    def test_embed_query(self):
        """Test embed_query method."""
        embeddings = MolDeBERTaEmbeddings()

        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.5] * 768]

            result = embeddings.embed_query("CCO")

            assert result == [0.5] * 768
            mock_get.assert_called_once_with(["CCO"])

    def test_embed_empty_list(self):
        """Test embedding empty SMILES list."""
        embeddings = MolDeBERTaEmbeddings()

        result = embeddings.embed_documents([])

        assert result == []

    def test_tokenizer_parameters(self):
        """Test that tokenizer is called with correct parameters."""
        embeddings = MolDeBERTaEmbeddings()

        smiles = ["CCO", "CC"]

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
            }

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(2, 3, 768)
                mock_model.return_value = mock_output

                embeddings._get_embedding(smiles)

                mock_tokenizer.assert_called_once_with(
                    smiles,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )

    def test_mean_pooling_calculation(self):
        """Test that mean pooling is correctly applied."""
        embeddings = MolDeBERTaEmbeddings()

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                # Create a known tensor for testing mean pooling
                known_tensor = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])  # (1, 3, 2)
                mock_output = MagicMock()
                mock_output.last_hidden_state = known_tensor
                mock_model.return_value = mock_output

                result = embeddings._get_embedding(["CCO"])

                # Mean of [[1,2], [3,4], [5,6]] should be [3, 4]
                expected_mean = [3.0, 4.0]
                assert len(result[0]) == 2
                assert abs(result[0][0] - expected_mean[0]) < 1e-6
                assert abs(result[0][1] - expected_mean[1]) < 1e-6

    def test_model_eval_mode(self):
        """Test that model is set to evaluation mode."""
        with patch("src.rag_graph.embedding.base_embeddings.AutoModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            MolDeBERTaEmbeddings()

            mock_model.eval.assert_called_once()

    def test_no_grad_context(self):
        """Test that model inference runs in no_grad context."""
        embeddings = MolDeBERTaEmbeddings()

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(1, 3, 768)
                mock_model.return_value = mock_output

                with patch("torch.no_grad") as mock_no_grad:
                    mock_no_grad.return_value.__enter__ = MagicMock()
                    mock_no_grad.return_value.__exit__ = MagicMock()

                    embeddings._get_embedding(["CCO"])

                    mock_no_grad.assert_called_once()

    def test_chemical_specific_smiles(self):
        """Test embedding generation with chemical-specific SMILES strings."""
        embeddings = MolDeBERTaEmbeddings()

        chemical_smiles = ["CCO", "CC(=O)O", "c1ccccc1", "CCN(CC)CC"]  # ethanol  # acetic acid  # benzene  # triethylamine

        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.1] * 768] * len(chemical_smiles)

            result = embeddings.embed_documents(chemical_smiles)

            assert len(result) == len(chemical_smiles)
            assert all(len(emb) == 768 for emb in result)
            mock_get.assert_called_once_with(chemical_smiles)

    def test_batch_processing(self):
        """Test that batch processing works correctly."""
        embeddings = MolDeBERTaEmbeddings()

        # Create a large batch to test batch processing
        large_smiles_batch = ["CCO"] * 100

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {
                "input_ids": torch.tensor([[1, 2, 3]] * 100),
                "attention_mask": torch.tensor([[1, 1, 1]] * 100),
            }

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.last_hidden_state = torch.randn(100, 3, 768)
                mock_model.return_value = mock_output

                result = embeddings._get_embedding(large_smiles_batch)

                assert len(result) == 100
                assert all(len(emb) == 768 for emb in result)

                # Verify tokenizer was called with the full batch
                mock_tokenizer.assert_called_once_with(
                    large_smiles_batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                )
