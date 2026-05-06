"""Embeddings for SMILES strings using Morgan fingerprints.

This module provides functionality to embed SMILES strings using Morgan fingerprints.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-06

"""

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from .base_embeddings import BaseEmbeddings


class MorganFingerprintEmbeddings(BaseEmbeddings):
    """Embeddings for SMILES strings using Morgan fingerprints."""

    def __init__(
        self,
        radius: int = 2,
        n_bits: int = 2048,
    ):
        self.radius = radius
        self.n_bits = n_bits
        # Override base class initialization since we don't use transformers
        # Set required attributes manually
        self.model_name = "morgan_fingerprint"
        self.max_length = None
        self.device = "cpu"

    def _get_fp(self, smiles: str):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [0.0] * self.n_bits

        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
            mol,
            self.radius,
            nBits=self.n_bits,
        )
        return [float(i) for i in fp]

    def _get_embedding(self, texts: list[str]) -> list[list[float]]:
        return [self._get_fp(t) for t in texts]
