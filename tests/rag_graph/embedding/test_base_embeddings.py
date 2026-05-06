# -*- coding: utf-8 -*-
# !/usr/bin/env python

from unittest.mock import MagicMock, patch

import torch

from src.rag_graph.embedding.base_embeddings import BaseEmbeddings


class ConcreteEmbeddings(BaseEmbeddings):
    """Concrete implementation for testing the abstract base class."""

    def _get_embedding(self, texts):
        return [[0.1, 0.2, 0.3]] * len(texts)


class TestBaseEmbeddings:

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_initialization_default_params(self, mock_tokenizer_class, mock_model_class):
        """Test BaseEmbeddings initializes with default parameters."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("test-model")

        assert embeddings.model_name == "test-model"
        assert embeddings.max_length == 512
        assert embeddings.device in ["cpu", "cuda"]

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_initialization_custom_params(self, mock_tokenizer_class, mock_model_class):
        """Test BaseEmbeddings initializes with custom parameters."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("custom-model", max_length=256, device="cpu")

        assert embeddings.model_name == "custom-model"
        assert embeddings.max_length == 256
        assert embeddings.device == "cpu"

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_tokenize_texts(self, mock_tokenizer_class, mock_model_class):
        """Test text tokenization."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("test-model")

        # Mock tokenizer
        mock_tokenizer_output = {"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.tensor([[1, 1, 1]])}
        embeddings.tokenizer = MagicMock(return_value=mock_tokenizer_output)

        result = embeddings._tokenize_texts(["test text"])

        assert "input_ids" in result
        assert "attention_mask" in result
        embeddings.tokenizer.assert_called_once_with(
            ["test text"],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_embed_documents(self, mock_tokenizer_class, mock_model_class):
        """Test embed_documents method."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("test-model")

        texts = ["text1", "text2", "text3"]
        result = embeddings.embed_documents(texts)

        assert len(result) == 3
        assert all(emb == [0.1, 0.2, 0.3] for emb in result)

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_embed_documents_empty_list(self, mock_tokenizer_class, mock_model_class):
        """Test embed_documents with empty list."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("test-model")

        result = embeddings.embed_documents([])

        assert result == []

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_embed_query(self, mock_tokenizer_class, mock_model_class):
        """Test embed_query method."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        embeddings = ConcreteEmbeddings("test-model")

        result = embeddings.embed_query("test text")

        assert result == [0.1, 0.2, 0.3]

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_device_auto_detection(self, mock_tokenizer_class, mock_model_class):
        """Test automatic device detection."""
        mock_tokenizer_class.from_pretrained.return_value = MagicMock()
        mock_model_class.from_pretrained.return_value = MagicMock()

        with patch("torch.cuda.is_available", return_value=True):
            embeddings = ConcreteEmbeddings("test-model", device="auto")
            assert embeddings.device == "cuda"

        with patch("torch.cuda.is_available", return_value=False):
            embeddings = ConcreteEmbeddings("test-model", device="auto")
            assert embeddings.device == "cpu"

    @patch("src.rag_graph.embedding.base_embeddings.AutoModel")
    @patch("src.rag_graph.embedding.base_embeddings.AutoTokenizer")
    def test_model_initialization(self, mock_tokenizer_class, mock_model_class):
        """Test that model and tokenizer are properly initialized."""
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        mock_model_class.from_pretrained.return_value = mock_model

        embeddings = ConcreteEmbeddings("test-model")

        mock_tokenizer_class.from_pretrained.assert_called_once_with("test-model")
        mock_model_class.from_pretrained.assert_called_once_with("test-model")
        mock_model.to.assert_called_once_with(embeddings.device)
        mock_model.eval.assert_called_once()

    def test_abstract_method_enforcement(self):
        """Test that abstract method _get_embedding must be implemented."""
        from abc import ABC

        # BaseEmbeddings should be abstract
        assert issubclass(BaseEmbeddings, ABC)

        # Should not be able to instantiate BaseEmbeddings directly
        try:
            BaseEmbeddings("test-model")
            assert False, "Should not be able to instantiate abstract class"
        except TypeError:
            pass  # Expected behavior
