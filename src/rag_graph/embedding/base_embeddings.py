# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Base embedding class with common functionality for all embedding types.

This module provides a base class that handles common embedding operations
like model initialization, tokenization, and tensor operations.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-06
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import torch
from langchain_core.embeddings import Embeddings
from transformers import AutoModel, AutoTokenizer


class BaseEmbeddings(Embeddings, ABC):
    """Base class for all embedding implementations."""

    def __init__(
        self,
        model_name: str,
        max_length: int = 512,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self._initialize_model()

    def _initialize_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModel.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

    def _tokenize_texts(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        # Move to device
        return {k: v.to(self.device) for k, v in inputs.items()}

    @abstractmethod
    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        pass

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of text documents to embed.

        Returns:
            List of embeddings for the documents.
        """
        if not texts:
            return []
        return self._get_embedding(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query.

        Args:
            text: Text query to embed.

        Returns:
            Embedding for the query.
        """
        return self._get_embedding([text])[0]
