# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Embeddings for NAICS industry titles using open-source text models.

This module provides functionality to embed NAICS industry titles using
lightweight open-source models that can run locally.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-06
"""

from typing import List

import torch

from .base_embeddings import BaseEmbeddings


class NAICSEmbeddings(BaseEmbeddings):
    """Embeddings for NAICS industry titles using text models."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_length: int = 512,
    ):
        super().__init__(model_name, max_length)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        inputs = self._tokenize_texts(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use mean pooling for better sentence representation
        embeddings = self._mean_pooling(outputs, inputs["attention_mask"])

        # Normalize embeddings
        embeddings = self._normalize_embeddings(embeddings)

        return embeddings.tolist()
