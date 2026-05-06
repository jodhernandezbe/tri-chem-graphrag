# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""RDKit-based canonical SMILES extractor for chemical identification.

This module provides functionality to generate canonical SMILES from CAS numbers
and chemical names using the RDKit library.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
import pathlib
from typing import Dict, List, Optional, Union

import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. SMILES extraction will be disabled.")

CURRENT_DIR = pathlib.Path(__file__).parent
WORKING_DIR = CURRENT_DIR.parent.parent.parent
INTERMEDIARY_DATA_DIR = WORKING_DIR / "data" / "processed"

logger = logging.getLogger(__name__)


class SmilesExtractor:
    """Singleton class for extracting canonical SMILES from chemical identifiers."""

    def __new__(cls):
        """Create or return existing singleton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(SmilesExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES extraction. Please install rdkit-pypi.")

        self._name_cache: Dict[str, Optional[str]] = {}
        self._cas_cache: Dict[str, Optional[str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _normalize_chemical_name(self, chemical_name: str) -> str:
        return chemical_name.strip().lower()

    def _normalize_cas_number(self, cas_number: str) -> str:
        return cas_number.strip().replace("-", "").replace(" ", "")

    def _smiles_from_name(self, chemical_name: str) -> Optional[str]:
        if pd.isna(chemical_name) or not chemical_name.strip():
            return None

        normalized_name = self._normalize_chemical_name(chemical_name)

        if normalized_name in self._name_cache:
            self._cache_hits += 1
            return self._name_cache[normalized_name]

        self._cache_misses += 1

        try:
            mol = Chem.MolFromSmiles(chemical_name)  # type: ignore [reportPossiblyUnboundVariable]
            if mol is not None:
                canonical_smiles = Chem.MolToSmiles(mol, canonical=True)  # type: ignore [reportPossiblyUnboundVariable]
                self._name_cache[normalized_name] = canonical_smiles
                return canonical_smiles

            self._name_cache[normalized_name] = None
            return None

        except Exception as e:
            logger.debug(f"Failed to extract SMILES from name '{chemical_name}': {e}")
            self._name_cache[normalized_name] = None
            return None

    def _smiles_from_cas(self, cas_number: str) -> Optional[str]:
        if pd.isna(cas_number) or not cas_number.strip():
            return None

        normalized_cas = self._normalize_cas_number(cas_number)

        if normalized_cas in self._cas_cache:
            self._cache_hits += 1
            return self._cas_cache[normalized_cas]

        self._cache_misses += 1

        # CAS number to SMILES conversion requires external databases
        # This would typically integrate with PubChem, ChemSpider, or other chemical databases
        # For now, return None because this is a placeholder for future implementation
        logger.debug(f"CAS to SMILES conversion not yet implemented for: {cas_number}")
        self._cas_cache[normalized_cas] = None
        return None

    def extract_smiles(
        self,
        identifier: str,
        identifier_type: str = "name",
    ) -> Optional[str]:
        """Extract canonical SMILES from a chemical identifier.

        Args:
            identifier (str): Chemical name or CAS number
            identifier_type (str): Type of identifier ("name" or "cas")

        Returns:
            Optional[str]: Canonical SMILES string if successful, None otherwise
        """
        if identifier_type == "name":
            return self._smiles_from_name(identifier)
        elif identifier_type == "cas":
            return self._smiles_from_cas(identifier)
        else:
            logger.warning(f"Unknown identifier type: {identifier_type}")
            return None

    def _initialize_smiles_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["canonical_smiles"] = None
        return df_copy

    def _extract_smiles_from_chemical_names(self, df: pd.DataFrame) -> pd.DataFrame:
        if "chemical_name" not in df.columns:
            return df

        logger.info("Extracting SMILES from chemical names")
        df_copy = df.copy()

        for idx in df_copy.index:
            chemical_name = df_copy.at[idx, "chemical_name"]
            if pd.notna(chemical_name):
                smiles = self.extract_smiles(str(chemical_name), "name")
                if smiles:
                    df_copy.at[idx, "canonical_smiles"] = smiles

        return df_copy

    def _extract_smiles_from_cas_numbers(self, df: pd.DataFrame) -> pd.DataFrame:
        if "cas_number" not in df.columns:
            return df

        logger.info("Extracting SMILES from CAS numbers for remaining records")
        df_copy = df.copy()

        # Only process records that don't have SMILES yet
        mask = df_copy["canonical_smiles"].isna()
        records_to_process = df_copy[mask]

        for idx in records_to_process.index:
            cas_number = df_copy.at[idx, "cas_number"]
            if pd.notna(cas_number):
                smiles = self.extract_smiles(str(cas_number), "cas")
                if smiles:
                    df_copy.at[idx, "canonical_smiles"] = smiles

        return df_copy

    def _log_extraction_results(self, df: pd.DataFrame) -> None:
        successful_extractions = df["canonical_smiles"].notna().sum()
        total_records = len(df)
        cache_hit_rate = (
            self._cache_hits / (self._cache_hits + self._cache_misses) * 100
            if (self._cache_hits + self._cache_misses) > 0
            else 0
        )

        logger.info(f"Successfully extracted SMILES for {successful_extractions}/{total_records} chemicals")
        logger.info(f"Cache statistics: {self._cache_hits} hits, {self._cache_misses} misses, {cache_hit_rate:.1f}% hit rate")

    def process_chemicals_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame to add canonical SMILES column.

        Args:
            df (pd.DataFrame): DataFrame containing chemical_name and/or cas_number columns

        Returns:
            pd.DataFrame: DataFrame with added canonical_smiles column
        """
        logger.info("Processing chemicals DataFrame for SMILES extraction")

        # Initialize SMILES column
        df_with_smiles_col = self._initialize_smiles_column(df)

        # Extract SMILES from chemical names first
        df_with_name_smiles = self._extract_smiles_from_chemical_names(df_with_smiles_col)

        # Extract SMILES from CAS numbers for remaining records
        df_final = self._extract_smiles_from_cas_numbers(df_with_name_smiles)

        # Log results
        self._log_extraction_results(df_final)

        return df_final
