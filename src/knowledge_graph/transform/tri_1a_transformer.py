# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI 1a file transformer for Releases and Waste Management data.

This module processes TRI 1a files which contain release and waste management data.
No row expansion is needed for this file type.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
from typing import Optional, cast

import pandas as pd

from ..extract.naics_api_queries import NaicsDataFetcher
from ..extract.smiles_extractor import SmilesExtractor
from .base_transformer import BaseTriTransformer

logger = logging.getLogger(__name__)


class Tri1aTransformer(BaseTriTransformer):
    """Transformer for TRI 1a files (Releases and Waste Management)."""

    def __init__(self):
        super().__init__(file_type="1a")
        self._naics_fetcher = NaicsDataFetcher()
        self._smiles_extractor = SmilesExtractor()

    def _get_maximum_amount_value_from_config(self, code: str) -> Optional[float]:
        for amount_info in self.config["TRI_FILE_TYPES"]["MAXIMUM_AMOUNT_CODE_NOMENCLATURE"]:
            if amount_info["CODE"] == code:
                return amount_info["VALUE"]
        return None

    def _add_naics_alias(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        naics_column = self.config["TRI_FILE_TYPES"]["NAICS_COLUMN"]
        if naics_column in df_copy.columns:
            df_copy["naics_code"] = df_copy[naics_column]
        return df_copy

    def _convert_maximum_amount_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "maximum_amount_on_site" not in df_copy.columns:
            return df_copy

        # Convert codes to numeric values
        df_copy["maximum_amount"] = df_copy["maximum_amount_on_site"].apply(
            lambda code: self._get_maximum_amount_value_from_config(str(code).strip()) if pd.notna(code) else None
        )

        return df_copy

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        numeric_columns = ["maximum_amount"]

        for col in numeric_columns:
            if col in df_copy.columns:
                df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")

        return df_copy

    def _enrich_with_naics_titles(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "naics_code" not in df_copy.columns:
            logger.warning("No naics_code column found for NAICS title enrichment")
            return df_copy

        try:
            naics_df = self._naics_fetcher.process_naics_codes(df_copy, "naics_code")
            df_enriched = df_copy.merge(naics_df, on="naics_code", how="left")
            logger.info("Successfully enriched data with NAICS titles")
            return df_enriched
        except Exception as e:
            logger.error(f"Error enriching NAICS data: {e}")
            return df_copy

    def _extract_chemical_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        chemical_columns = ["chemical_name", "cas_number", "tri_chem_id"]
        available_chem_cols = [col for col in chemical_columns if col in df_copy.columns]

        if not available_chem_cols:
            logger.warning("No chemical columns found for SMILES extraction")
            return df_copy

        try:
            unique_chemicals = cast(
                pd.DataFrame,
                df_copy.loc[:, available_chem_cols].drop_duplicates(),
            )
            chemicals_with_smiles = self._smiles_extractor.process_chemicals_dataframe(unique_chemicals)

            # Merge SMILES back to the main dataframe
            merge_columns = [col for col in available_chem_cols if col in chemicals_with_smiles.columns]
            if merge_columns:
                df_with_smiles = df_copy.merge(
                    chemicals_with_smiles[merge_columns + ["canonical_smiles"]], on=merge_columns, how="left"
                )
                logger.info("Successfully extracted and merged SMILES data")
                return df_with_smiles
            else:
                logger.warning("No suitable columns for merging SMILES data")
                return df_copy

        except Exception as e:
            logger.error(f"Error extracting SMILES: {e}")
            return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 1a data according to knowledge graph requirements.

        Args:
            df: Input DataFrame with filtered 1a data

        Returns:
            Transformed DataFrame ready for knowledge graph loading
        """
        logger.info("Transforming TRI 1a data")

        # Add naics_code alias for knowledge graph
        df_with_naics = self._add_naics_alias(df)

        # Convert maximum amount codes to numeric values
        df_with_amounts = self._convert_maximum_amount_codes(df_with_naics)

        # Clean numeric columns
        df_cleaned = self._clean_numeric_columns(df_with_amounts)

        # Enrich with NAICS titles
        df_with_naics_titles = self._enrich_with_naics_titles(df_cleaned)

        # Extract and add canonical SMILES
        df_final = self._extract_chemical_smiles(df_with_naics_titles)

        logger.info(f"Transformed 1a data: {len(df_final)} rows with enrichments")
        return df_final
