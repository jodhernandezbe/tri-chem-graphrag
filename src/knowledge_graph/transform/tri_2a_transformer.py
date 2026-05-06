# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI 2a file transformer for Source Reduction data.

This module processes TRI 2a files which contain source reduction activity data.
Requires row expansion for each reduction activity.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
from typing import Dict, List, Optional

import pandas as pd

from .base_transformer import BaseTriTransformer

logger = logging.getLogger(__name__)


class Tri2aTransformer(BaseTriTransformer):
    """Transformer for TRI 2a files (Source Reduction)."""

    def __init__(self):
        super().__init__(file_type="2a")
        self._reduction_codes_considered = set(self.config["TRI_FILE_TYPES"]["SOURCE_REDUCTION_CODES_CONSIDERED"])

    def _get_reduction_value_from_config(self, code: str) -> Optional[float]:
        for reduction_info in self.config["TRI_FILE_TYPES"]["SOURCE_REDUCTION_CODE_NOMENCLATURE"]:
            if reduction_info["CODE"] == code:
                return reduction_info["VALUE"]
        return None

    def _get_reduction_columns_mapping(self) -> Dict[str, List[str]]:
        all_columns = self._get_columns_to_use()

        codes = [col for col in all_columns if "source_reduction_activity_code" in col and "est_annual" not in col]
        descriptions = [col for col in all_columns if "source_reduction_activity_description" in col]
        annual_reductions = [col for col in all_columns if "est_annual_reduction_code" in col]

        return {"codes": codes, "descriptions": descriptions, "annual_reductions": annual_reductions}

    def _is_considered_reduction_code(self, code_value: object) -> bool:
        if not pd.api.types.is_scalar(code_value) or bool(pd.isna(code_value)):
            return False
        return str(code_value).strip() in self._reduction_codes_considered

    def _get_reduction_code_from_row(self, row: pd.Series, code_column: str) -> Optional[str]:
        if code_column not in row.index:
            return None

        code_value = row.at[code_column]
        if self._is_considered_reduction_code(code_value):
            return str(code_value).strip()
        return None

    def _get_column_value_safely(self, row: pd.Series, column: Optional[str]) -> Optional[str]:
        if column is not None and column in row.index:
            value = row.at[column]
            if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
                return None
            return str(value)
        return None

    def _create_reduction_row(self, base_row: pd.Series, reduction_data: Dict) -> pd.Series:
        expanded_row = base_row.copy()
        expanded_row["source_reduction_code"] = reduction_data["code"]
        expanded_row["source_reduction_description"] = reduction_data.get("description")

        # Convert annual reduction code to numeric value
        annual_reduction_code = reduction_data.get("annual_reduction")
        if annual_reduction_code:
            expanded_row["annual_reduction"] = self._get_reduction_value_from_config(annual_reduction_code)
        else:
            expanded_row["annual_reduction"] = None

        return expanded_row

    def _create_empty_reduction_row(self, base_row: pd.Series) -> pd.Series:
        row = base_row.copy()
        row["source_reduction_code"] = None
        row["source_reduction_description"] = None
        row["annual_reduction"] = None
        return row

    def _extract_reduction_activities_from_row(self, row: pd.Series) -> List[Dict]:
        columns_mapping = self._get_reduction_columns_mapping()
        codes = columns_mapping["codes"]
        descriptions = columns_mapping["descriptions"]
        annual_reductions = columns_mapping["annual_reductions"]

        activities = []

        for i, code_col in enumerate(codes):
            reduction_code = self._get_reduction_code_from_row(row, code_col)

            if reduction_code:
                desc_col = descriptions[i] if i < len(descriptions) else None
                annual_col = annual_reductions[i] if i < len(annual_reductions) else None

                activity = {
                    "code": reduction_code,
                    "description": self._get_column_value_safely(row, desc_col),
                    "annual_reduction": self._get_column_value_safely(row, annual_col),
                }
                activities.append(activity)

        return activities

    def _expand_single_row(self, row: pd.Series) -> List[pd.Series]:
        reduction_activities = self._extract_reduction_activities_from_row(row)

        if reduction_activities:
            return [self._create_reduction_row(row, activity) for activity in reduction_activities]
        else:
            return [self._create_empty_reduction_row(row)]

    def _expand_reduction_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        expanded_rows = []

        for _, row in df.iterrows():
            expanded_rows.extend(self._expand_single_row(row))

        return pd.DataFrame(expanded_rows)

    def _add_naics_alias(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        naics_column = self.config["TRI_FILE_TYPES"]["NAICS_COLUMN"]
        if naics_column in df_copy.columns:
            df_copy["naics_code"] = df_copy[naics_column]
        return df_copy

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "annual_reduction" in df_copy.columns:
            df_copy["annual_reduction"] = pd.to_numeric(df_copy["annual_reduction"], errors="coerce")
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 2a data according to knowledge graph requirements.

        Args:
            df (pd.DataFrame): Input DataFrame with filtered 2a data

        Returns:
            pd.DataFrame: Transformed DataFrame ready for knowledge graph loading
        """
        logger.info("Transforming TRI 2a data")

        # Add naics_code alias for knowledge graph
        df_with_naics = self._add_naics_alias(df)

        # Expand rows based on reduction activities
        df_expanded = self._expand_reduction_activities(df_with_naics)

        # Clean numeric columns
        df_cleaned = self._clean_numeric_columns(df_expanded)

        logger.info(f"Transformed 2a data: {len(df)} -> {len(df_cleaned)} rows after reduction activity expansion")
        return df_cleaned
