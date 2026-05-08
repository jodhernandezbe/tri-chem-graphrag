# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""RDKit-based canonical SMILES extractor for chemical identification.

This module provides functionality to generate canonical SMILES from CAS numbers
and chemical names using the PubChem REST API and RDKit for canonicalization.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
import time
from typing import Dict, Optional
from urllib.parse import quote

import pandas as pd
import requests

try:
    from rdkit import Chem

    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False
    logging.warning("RDKit not available. SMILES extraction will be disabled.")

logger = logging.getLogger(__name__)

_PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{identifier}/property/CanonicalSMILES/JSON"
_PUBCHEM_DELAY = 0.2


class SmilesExtractor:
    """Singleton class for extracting canonical SMILES from chemical identifiers via PubChem."""

    def __new__(cls):
        """Create or return existing singleton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(SmilesExtractor, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        if not RDKIT_AVAILABLE:
            raise ImportError("RDKit is required for SMILES canonicalization. Please install rdkit-pypi.")

        self._cas_cache: Dict[str, Optional[str]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _normalize_cas_number(self, cas_number: str) -> str:
        return cas_number.strip().replace("-", "").replace(" ", "")

    def _canonicalize(self, smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles)  # type: ignore[reportPossiblyUnboundVariable]
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)  # type: ignore[reportPossiblyUnboundVariable]

    def _fetch_from_pubchem(self, identifier: str) -> Optional[str]:
        url = _PUBCHEM_URL.format(identifier=quote(identifier))
        try:
            time.sleep(_PUBCHEM_DELAY)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                props = response.json()["PropertyTable"]["Properties"][0]
                raw_smiles = props.get("CanonicalSMILES") or props.get("ConnectivitySMILES") or props.get("SMILES")
                if raw_smiles:
                    return self._canonicalize(raw_smiles) or raw_smiles
            logger.debug(f"PubChem returned {response.status_code} for '{identifier}'")
        except Exception as e:
            logger.debug(f"PubChem lookup failed for '{identifier}': {e}")
        return None

    def _smiles_from_cas(self, cas_number: str) -> Optional[str]:
        if pd.isna(cas_number) or not cas_number.strip():
            return None

        normalized = self._normalize_cas_number(cas_number)
        if normalized in self._cas_cache:
            self._cache_hits += 1
            return self._cas_cache[normalized]

        self._cache_misses += 1
        smiles = self._fetch_from_pubchem(cas_number.strip())
        self._cas_cache[normalized] = smiles
        return smiles

    def extract_smiles(self, cas_number: str) -> Optional[str]:
        """Extract canonical SMILES for a chemical from its CAS number via PubChem.

        Args:
            cas_number (str): CAS registry number (e.g., "7439-92-1").

        Returns:
            Optional[str]: Canonical SMILES string if found, None otherwise.
        """
        return self._smiles_from_cas(cas_number)

    def _initialize_smiles_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        df_copy["canonical_smiles"] = None
        return df_copy

    def _fill_smiles_from_cas(self, df: pd.DataFrame) -> pd.DataFrame:
        if "cas_number" not in df.columns:
            return df

        df_copy = df.copy()
        for idx in df_copy.index:
            cas_number = df_copy.at[idx, "cas_number"]
            if pd.notna(cas_number):
                smiles = self._smiles_from_cas(str(cas_number))
                if smiles:
                    df_copy.at[idx, "canonical_smiles"] = smiles

        return df_copy

    def _log_extraction_results(self, df: pd.DataFrame) -> None:
        successful = df["canonical_smiles"].notna().sum()
        total = len(df)
        hits_misses = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / hits_misses * 100 if hits_misses > 0 else 0
        logger.info(f"Successfully extracted SMILES for {successful}/{total} chemicals")
        logger.info(f"Cache: {self._cache_hits} hits, {self._cache_misses} misses, {hit_rate:.1f}% hit rate")

    def process_chemicals_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a DataFrame to add canonical SMILES via PubChem CAS lookup.

        Args:
            df (pd.DataFrame): DataFrame containing a cas_number column.

        Returns:
            pd.DataFrame: DataFrame with added canonical_smiles column.
        """
        logger.info("Processing chemicals DataFrame for SMILES extraction")
        df_init = self._initialize_smiles_column(df)
        df_final = self._fill_smiles_from_cas(df_init)
        self._log_extraction_results(df_final)
        return df_final
