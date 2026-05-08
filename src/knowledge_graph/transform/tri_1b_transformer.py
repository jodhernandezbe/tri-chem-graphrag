# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI 1b file transformer for Chemical Activity and Use data.

This module processes TRI 1b files which contain chemical activity and use data.
Requires row expansion for each activity marked as True, one row per active activity.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
from typing import List

import pandas as pd

from .base_transformer import BaseTriTransformer

logger = logging.getLogger(__name__)


class Tri1bTransformer(BaseTriTransformer):
    """Transformer for TRI 1b files (Chemical Activity and Use)."""

    def __init__(self):
        super().__init__(file_type="1b")
        self._activity_columns = self._get_activity_columns_from_config()
        self._yes_values = set(str(v).lower() for v in ["yes", "y", "1", "true"])

    def _get_activity_columns_from_config(self) -> List[str]:
        return [col for col in self._get_columns_to_use() if col not in ["document_control_number"]]

    def _is_yes_value(self, value: object) -> bool:
        if pd.isna(value):  # type: ignore[arg-type]
            return False
        return str(value).lower().strip() in self._yes_values

    def _standardize_activity_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        available = [col for col in self._activity_columns if col in df_copy.columns]
        for col in available:
            series = df_copy[col]
            if isinstance(series, pd.Series):
                df_copy[col] = series.map(self._is_yes_value)
        return df_copy

    def _format_activity_label(self, column_name: str) -> str:
        return column_name.replace("_", " ").capitalize()

    def _get_active_activities(self, row: pd.Series) -> List[str]:
        return [col for col in self._activity_columns if col in row.index and bool(row.at[col]) is True]

    def _expand_row(self, row: pd.Series) -> List[pd.Series]:
        active = self._get_active_activities(row)
        result = []
        for activity in active:
            new_row = pd.Series({"document_control_number": row["document_control_number"]})
            new_row["chemical_activity"] = self._format_activity_label(activity)
            result.append(new_row)
        return result

    def _expand_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._activity_columns:
            logger.warning("No activity columns found for expansion")
            return df

        df_standardized = self._standardize_activity_columns(df)
        expanded_rows: List[pd.Series] = []
        for _, row in df_standardized.iterrows():
            expanded_rows.extend(self._expand_row(row))

        result = pd.DataFrame(expanded_rows)
        if "document_control_number" in result.columns:
            numeric: pd.Series = pd.to_numeric(result["document_control_number"], errors="coerce")  # type: ignore[assignment]
            result["document_control_number"] = numeric.astype("Int64")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 1b data according to knowledge graph requirements.

        Args:
            df (pd.DataFrame): Input DataFrame with filtered 1b data.

        Returns:
            pd.DataFrame: Transformed DataFrame ready for knowledge graph loading.
        """
        logger.info("Transforming TRI 1b data")
        df_expanded = self._expand_activities(df)
        logger.info(f"Transformed 1b data: {len(df)} rows → {len(df_expanded)} activity rows")
        return df_expanded
