"""Embeddings for SMILES strings using a hybrid of Morgan fingerprints and MolDeBERTa.

This module provides functionality to embed SMILES strings using a hybrid of Morgan fingerprints and MolDeBERTa.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-06

"""

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

from .base_embeddings import BaseEmbeddings


class HybridMolecularEmbeddings(BaseEmbeddings):
    """Embeddings for SMILES strings using a hybrid of Morgan fingerprints and MolDeBERTa."""

    def __init__(
        self,
        model_name: str = "SaeedLab/MolDeBERTa-base-123M-mtr",
        max_length: int = 128,
        radius: int = 2,
        n_bits: int = 1024,
    ):
        super().__init__(model_name, max_length=max_length)
        self.radius = radius
        self.n_bits = n_bits

    def _get_morgan(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(self.n_bits)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, self.radius, nBits=self.n_bits)
        arr = np.zeros((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr

    def _get_deberta(self, smiles: str):
        inputs = self.tokenizer(
            smiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        combined_vectors = []
        for smiles in texts:
            vector = np.concatenate([self._get_morgan(smiles), self._get_deberta(smiles)])
            combined_vectors.append(vector.tolist())
        return combined_vectors
