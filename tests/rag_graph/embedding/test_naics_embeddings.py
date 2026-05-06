# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import torch

from src.rag_graph.embedding.naics_embeddings import NAICSEmbeddings


class TestNAICSEmbeddings:

    def test_initialization_default_params(self):
        """Test NAICSEmbeddings initializes with default parameters."""
        embeddings = NAICSEmbeddings()

        assert embeddings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert embeddings.max_length == 512
        assert hasattr(embeddings, "tokenizer")
        assert hasattr(embeddings, "model")

    def test_initialization_custom_params(self):
        """Test NAICSEmbeddings initializes with custom parameters."""
        custom_model = "sentence-transformers/all-mpnet-base-v2"
        custom_length = 256

        embeddings = NAICSEmbeddings(model_name=custom_model, max_length=custom_length)

        assert embeddings.model_name == custom_model
        assert embeddings.max_length == custom_length

    def test_mean_pooling(self):
        """Test mean pooling calculation."""
        embeddings = NAICSEmbeddings()

        # Create mock model output and attention mask
        batch_size, seq_len, hidden_size = 2, 5, 4
        token_embeddings = torch.randn(batch_size, seq_len, hidden_size)
        model_output = (token_embeddings,)
        attention_mask = torch.ones(batch_size, seq_len)

        result = embeddings._mean_pooling(model_output, attention_mask)

        assert result.shape == (batch_size, hidden_size)
        assert isinstance(result, torch.Tensor)

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_get_embedding_basic(self, mock_tokenizer_class, mock_model_class):
        """Test basic embedding generation."""
        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Mock model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.__getitem__.return_value = torch.randn(1, 3, 4)  # (batch, seq, hidden)
        mock_model.return_value = mock_output
        mock_model_class.from_pretrained.return_value = mock_model

        embeddings = NAICSEmbeddings()

        texts = ["Chemical Manufacturing"]
        result = embeddings._get_embedding(texts)

        assert isinstance(result, list)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 4  # hidden_size

    def test_embed_query(self):
        """Test embedding a single query."""
        embeddings = NAICSEmbeddings()

        # Mock _get_embedding to return predictable result
        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.1, 0.2, 0.3, 0.4]]

            result = embeddings.embed_query("Chemical Manufacturing")

            assert result == [0.1, 0.2, 0.3, 0.4]
            mock_get.assert_called_once_with(["Chemical Manufacturing"])

    def test_embed_documents(self):
        """Test embedding multiple documents."""
        embeddings = NAICSEmbeddings()

        # Mock _get_embedding to return predictable result
        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]

            texts = ["Chemical Manufacturing", "Petroleum Products"]
            result = embeddings.embed_documents(texts)

            assert len(result) == 2
            assert result[0] == [0.1, 0.2, 0.3, 0.4]
            assert result[1] == [0.5, 0.6, 0.7, 0.8]
            mock_get.assert_called_once_with(texts)

    def test_embed_documents_empty_list(self):
        """Test embedding empty document list."""
        embeddings = NAICSEmbeddings()

        result = embeddings.embed_documents([])

        assert result == []

    def test_embed_naics_specific_content(self):
        """Test embedding NAICS-specific content."""
        embeddings = NAICSEmbeddings()

        # Test with NAICS industry titles
        naics_titles = [
            "Chemical Manufacturing",
            "Petroleum and Coal Products Manufacturing",
            "Food Manufacturing",
            "Transportation Equipment Manufacturing",
        ]

        with patch.object(embeddings, "_get_embedding") as mock_get:
            mock_get.return_value = [[0.1] * 384] * len(naics_titles)

            result = embeddings.embed_documents(naics_titles)

            assert len(result) == len(naics_titles)
            assert all(len(emb) == 384 for emb in result)

    def test_tokenizer_parameters(self):
        """Test that tokenizer is called with correct parameters."""
        embeddings = NAICSEmbeddings(max_length=256)

        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                mock_output = MagicMock()
                mock_output.__getitem__.return_value = torch.randn(1, 3, 4)
                mock_model.return_value = mock_output

                embeddings._get_embedding(["test text"])

                mock_tokenizer.assert_called_once_with(
                    ["test text"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )

    def test_model_eval_mode(self):
        """Test that model is set to evaluation mode."""
        with patch("src.rag_graph.embedding.base_embeddings.AutoModel") as mock_model_class:
            mock_model = MagicMock()
            mock_model_class.from_pretrained.return_value = mock_model

            NAICSEmbeddings()

            mock_model.eval.assert_called_once()

    def test_mean_pooling(self):
        """Test mean pooling calculation."""
        embeddings = NAICSEmbeddings()

        # Create mock model output and attention mask
        batch_size, hidden_size = 2, 4
        token_embeddings = torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]],
                [[2.0, 4.0, 6.0, 8.0], [10.0, 12.0, 14.0, 16.0], [18.0, 20.0, 22.0, 24.0]],
            ]
        )
        model_output = (token_embeddings,)
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])  # Second sequence has padding

        result = embeddings._mean_pooling(model_output, attention_mask)

        assert result.shape == (batch_size, hidden_size)
        # First sequence: mean of all 3 tokens
        expected_first = torch.tensor([5.0, 6.0, 7.0, 8.0])
        # Second sequence: mean of first 2 tokens (third is masked)
        expected_second = torch.tensor([6.0, 8.0, 10.0, 12.0])

        assert torch.allclose(result[0], expected_first, atol=1e-6)
        assert torch.allclose(result[1], expected_second, atol=1e-6)

    def test_normalize_embeddings(self):
        """Test embedding normalization."""
        embeddings = NAICSEmbeddings()

        # Create unnormalized embeddings
        unnormalized = torch.tensor([[3.0, 4.0], [6.0, 8.0]])  # norms: 5.0, 10.0

        result = embeddings._normalize_embeddings(unnormalized)

        # Check that result has unit norm
        norms = torch.norm(result, dim=1)
        expected_norms = torch.tensor([1.0, 1.0])
        assert torch.allclose(norms, expected_norms, atol=1e-6)

        # Check that direction is preserved
        expected_normalized = torch.tensor([[0.6, 0.8], [0.6, 0.8]])
        assert torch.allclose(result, expected_normalized, atol=1e-6)

    def test_embedding_normalization(self):
        """Test that embeddings are properly normalized."""
        embeddings = NAICSEmbeddings()

        # Test with real tensors to verify normalization
        with patch.object(embeddings, "tokenizer") as mock_tokenizer:
            mock_tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}

            with patch.object(embeddings, "model") as mock_model:
                # Create non-normalized embeddings
                unnormalized = torch.tensor([[3.0, 4.0]])  # norm = 5.0
                mock_output = MagicMock()
                mock_output.__getitem__.return_value = unnormalized.unsqueeze(0)  # Add seq dimension
                mock_model.return_value = mock_output

                with patch.object(embeddings, "_mean_pooling") as mock_pooling:
                    mock_pooling.return_value = unnormalized

                    result = embeddings._get_embedding(["test"])

                    # Check if result is normalized (should have norm ~1.0)
                    result_tensor = torch.tensor(result)
                    norm = torch.norm(result_tensor, dim=1)
                    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6)
