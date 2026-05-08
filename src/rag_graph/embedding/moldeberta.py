"""Embeddings for SMILES strings using the MolDeBERTa model.

This module provides functionality to embed SMILES strings using the MolDeBERTa model.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-06
"""

from typing import List

import torch

from .base_embeddings import BaseEmbeddings


class MolDeBERTaEmbeddings(BaseEmbeddings):
    """Embeddings for SMILES strings using the MolDeBERTa model."""

    def __init__(
        self,
        model_name: str = "SaeedLab/MolDeBERTa-base-123M-mtr",
        max_length: int = 128,
    ):
        super().__init__(model_name, max_length=max_length)

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        inputs = self._tokenize_texts(texts)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # For MolDeBERTa, mean pooling is often more stable for similarity
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings.tolist()
