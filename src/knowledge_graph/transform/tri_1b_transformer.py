# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI 1b file transformer for Chemical Activity and Use data.

This module processes TRI 1b files which contain chemical activity and use data.
Requires row expansion for each activity marked as "Yes".

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

    def _is_yes_value(self, value) -> bool:
        if pd.isna(value):
            return False
        return str(value).lower().strip() in self._yes_values

    def _standardize_single_column(self, series: pd.Series) -> pd.Series:
        return series.map(self._is_yes_value)

    def _standardize_activity_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        available_activities = [col for col in self._activity_columns if col in df_copy.columns]

        for col in available_activities:
            column_series = df_copy[col]
            if isinstance(column_series, pd.Series):
                df_copy[col] = self._standardize_single_column(column_series)

        return df_copy

    def _get_active_activities_for_row(self, row: pd.Series) -> List[str]:
        available_activities = [col for col in self._activity_columns if col in row.index]
        return [col for col in available_activities if row[col] is True]

    def _create_expanded_row(self, base_row: pd.Series, activity: str) -> pd.Series:
        expanded_row = base_row.copy()
        expanded_row["chemical_activity"] = activity
        return expanded_row

    def _create_empty_activity_row(self, base_row: pd.Series) -> pd.Series:
        row = base_row.copy()
        row["chemical_activity"] = None
        return row

    def _expand_single_row(self, row: pd.Series) -> List[pd.Series]:
        active_activities = self._get_active_activities_for_row(row)

        if active_activities:
            return [self._create_expanded_row(row, activity) for activity in active_activities]
        else:
            return [self._create_empty_activity_row(row)]

    def _expand_activities(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._activity_columns:
            logger.warning("No activity columns found for expansion")
            return df

        df_standardized = self._standardize_activity_columns(df)

        expanded_rows = []
        for _, row in df_standardized.iterrows():
            expanded_rows.extend(self._expand_single_row(row))

        return pd.DataFrame(expanded_rows)

    def _add_naics_alias(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        naics_column = self.config["TRI_FILE_TYPES"]["NAICS_COLUMN"]
        if naics_column in df_copy.columns:
            df_copy["naics_code"] = df_copy[naics_column]
        return df_copy

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform 1b data according to knowledge graph requirements.

        Args:
            df (pd.DataFrame): Input DataFrame with filtered 1b data

        Returns:
            pd.DataFrame: Transformed DataFrame ready for knowledge graph loading
        """
        logger.info("Transforming TRI 1b data")

        # Add naics_code alias for knowledge graph
        df_with_naics = self._add_naics_alias(df)

        # Expand rows based on activities
        df_expanded = self._expand_activities(df_with_naics)

        logger.info(f"Transformed 1b data: {len(df)} -> {len(df_expanded)} rows after activity expansion")
        return df_expanded
