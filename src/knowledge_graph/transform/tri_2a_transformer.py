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

ANNUAL_REDUCTION_SUFFIX = "_est_annual_reduction_code"


class Tri2aTransformer(BaseTriTransformer):
    """Transformer for TRI 2a files (Source Reduction)."""

    def __init__(self):
        super().__init__(file_type="2a")
        self._reduction_codes_considered = set(self.config["TRI_FILE_TYPES"]["SOURCE_REDUCTION_CODES_CONSIDERED"])

    def _filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        base_columns = [col for col in self._get_columns_to_use() if col in df.columns]
        reduction_columns = [
            col
            for col in df.columns
            if ("source_reduction_activity" in col or col.endswith(ANNUAL_REDUCTION_SUFFIX))
            and not col.endswith(ANNUAL_REDUCTION_SUFFIX + "_description")
        ]
        return df[base_columns + reduction_columns].copy()  # type: ignore[return-value]

    def _get_reduction_value_from_config(self, code: str) -> Optional[float]:
        for reduction_info in self.config["TRI_FILE_TYPES"]["SOURCE_REDUCTION_CODE_NOMENCLATURE"]:
            if reduction_info["CODE"] == code:
                return reduction_info["VALUE"]
        return None

    def _get_reduction_columns_mapping(self, all_columns: List[str]) -> Dict[str, List[str]]:
        codes = [col for col in all_columns if "source_reduction_activity_code" in col and "est_annual" not in col]
        descriptions = [col for col in all_columns if "source_reduction_activity_description" in col]
        annual_reductions = [col for col in all_columns if col.endswith(ANNUAL_REDUCTION_SUFFIX)]
        return {"codes": codes, "descriptions": descriptions, "annual_reductions": annual_reductions}

    def _is_valid_scalar(self, value: object) -> bool:
        if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
            return False
        return bool(str(value).strip())

    def _is_significant_annual_reduction(self, code_value: object) -> bool:
        if not self._is_valid_scalar(code_value):
            return False
        return str(code_value).strip() in self._reduction_codes_considered

    def _get_column_value_safely(self, row: pd.Series, column: Optional[str]) -> Optional[str]:
        if column is not None and column in row.index:
            value = row.at[column]
            if not pd.api.types.is_scalar(value) or bool(pd.isna(value)):
                return None
            return str(value)
        return None

    def _create_reduction_row(self, base_row: pd.Series, reduction_data: Dict) -> pd.Series:
        annual_code = reduction_data.get("annual_reduction")
        return pd.Series(
            {
                "document_control_number": base_row["document_control_number"],
                "source_reduction_code": reduction_data["code"],
                "source_reduction_description": reduction_data.get("description"),
                "annual_reduction": self._get_reduction_value_from_config(annual_code) if annual_code else None,
            }
        )

    def _extract_reduction_activities_from_row(
        self,
        row: pd.Series,
        columns_mapping: Dict[str, List[str]],
    ) -> List[Dict]:
        codes = columns_mapping["codes"]
        descriptions = columns_mapping["descriptions"]
        annual_reductions = columns_mapping["annual_reductions"]
        activities = []

        for i, code_col in enumerate(codes):
            activity_code = self._get_column_value_safely(row, code_col)
            if not activity_code:
                continue
            annual_col = annual_reductions[i] if i < len(annual_reductions) else None
            annual_code = self._get_column_value_safely(row, annual_col)
            if not self._is_significant_annual_reduction(annual_code):
                continue
            activities.append(
                {
                    "code": activity_code,
                    "description": self._get_column_value_safely(row, descriptions[i] if i < len(descriptions) else None),
                    "annual_reduction": annual_code,
                }
            )

        return activities

    def _expand_single_row(self, row: pd.Series, columns_mapping: Dict[str, List[str]]) -> List[pd.Series]:
        activities = self._extract_reduction_activities_from_row(row, columns_mapping)
        return [self._create_reduction_row(row, activity) for activity in activities]

    def _expand_reduction_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_mapping = self._get_reduction_columns_mapping(list(df.columns))
        expanded_rows: List[pd.Series] = []
        for _, row in df.iterrows():
            expanded_rows.extend(self._expand_single_row(row, columns_mapping))
        if not expanded_rows:
            return pd.DataFrame(
                columns=pd.Index(
                    ["document_control_number", "source_reduction_code", "source_reduction_description", "annual_reduction"]
                )
            )
        result = pd.DataFrame(expanded_rows)
        if "document_control_number" in result.columns:
            numeric: pd.Series = pd.to_numeric(result["document_control_number"], errors="coerce")  # type: ignore[assignment]
            result["document_control_number"] = numeric.astype("Int64")
        return result

    def _clean_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "annual_reduction" in df_copy.columns:
            df_copy["annual_reduction"] = pd.to_numeric(df_copy["annual_reduction"], errors="coerce")
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 2a data according to knowledge graph requirements.

        Args:
            df (pd.DataFrame): Input DataFrame with filtered 2a data.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for knowledge graph loading.
        """
        logger.info("Transforming TRI 2a data")
        df_expanded = self._expand_reduction_activities(df)
        df_cleaned = self._clean_numeric_columns(df_expanded)
        logger.info(f"Transformed 2a data: {len(df)} rows → {len(df_cleaned)} reduction activity rows")
        return df_cleaned
