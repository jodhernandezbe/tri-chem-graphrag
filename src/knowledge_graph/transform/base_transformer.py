# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Base transformation class for TRI data files.

This module provides the abstract base class for all TRI file transformers,
defining common interfaces and utilities for data processing.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
import pathlib
from abc import ABC, abstractmethod
from typing import Dict, List, cast

import pandas as pd

from src.config_manager import ConfigManager

CURRENT_DIR = pathlib.Path(__file__).parent
WORKING_DIR = CURRENT_DIR.parent.parent.parent
RAW_DATA_DIR = WORKING_DIR / "data" / "raw"
INTERMEDIARY_DATA_DIR = WORKING_DIR / "data" / "processed"

logger = logging.getLogger(__name__)


class BaseTriTransformer(ABC):
    """Abstract base class for TRI file transformers."""

    def __init__(self, file_type: str):
        """Initialize the transformer with file type configuration.

        Args:
            file_type: The TRI file type (e.g., '1a', '1b', '2a')
        """
        self.file_type = file_type
        self.config = ConfigManager().config
        self.file_config = self._get_file_config()
        self.kg_config = self.config["KNOWLEDGE_GRAPH"]

    def _get_file_config(self) -> Dict:
        for file_cfg in self.config["TRI_FILE_TYPES"]["FILE_TYPES"]:
            if file_cfg["FILE_TYPE"] == self.file_type:
                return file_cfg
        raise ValueError(f"File type {self.file_type} not found in configuration")

    def _get_columns_to_use(self) -> List[str]:
        return self.file_config["COLUMNS_TO_USE"]

    def _filter_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_to_use = self._get_columns_to_use()
        available_columns = [col for col in columns_to_use if col in df.columns]

        if len(available_columns) != len(columns_to_use):
            missing_columns = set(columns_to_use) - set(available_columns)
            logger.warning(f"Missing columns for file type {self.file_type}: {missing_columns}")

        return cast(pd.DataFrame, df.loc[:, available_columns].copy())

    def load_raw_data(self, year: str) -> pd.DataFrame:
        """Load raw TRI data for specified year and file type.

        Args:
            year: The reporting year

        Returns:
            Raw DataFrame from CSV file

        Raises:
            FileNotFoundError: If the raw data file doesn't exist
        """
        file_path = RAW_DATA_DIR / f"tri_{self.file_type}_{year}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"Raw data file not found: {file_path}")

        logger.info(f"Loading raw data from {file_path}")
        return pd.read_csv(file_path, low_memory=False)

    def save_intermediary_data(self, df: pd.DataFrame, year: str, suffix: str = "") -> pathlib.Path:
        """Save intermediary transformed data.

        Args:
            df: DataFrame to save
            year: The reporting year
            suffix: Optional suffix for the filename

        Returns:
            Path to saved file
        """
        INTERMEDIARY_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filename = f"tri_{self.file_type}_{year}_transformed"
        if suffix:
            filename += f"_{suffix}"
        filename += ".csv"

        file_path = INTERMEDIARY_DATA_DIR / filename
        df.to_csv(file_path, index=False, encoding="utf-8")
        logger.info(f"Saved intermediary data to {file_path}")
        return file_path

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame according to file type rules.

        Args:
            df: Input DataFrame with raw TRI data

        Returns:
            Transformed DataFrame ready for knowledge graph loading
        """
        pass

    def validate_input(self, df: pd.DataFrame) -> bool:
        """Validate that input DataFrame has required columns.

        Args:
            df: Input DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        required_columns = self._get_columns_to_use()
        missing_columns = set(required_columns) - set(df.columns)

        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return False

        return True

    def process_year(self, year: str) -> pathlib.Path:
        """Complete transformation pipeline for a given year.

        Args:
            year: The reporting year to process

        Returns:
            Path to the saved transformed data
        """
        logger.info(f"Processing {self.file_type} data for year {year}")

        # Load raw data
        raw_df = self.load_raw_data(year)

        # Validate input
        if not self.validate_input(raw_df):
            raise ValueError(f"Invalid input data for {self.file_type} year {year}")

        # Filter columns
        filtered_df = self._filter_columns(raw_df)

        # Apply transformation
        transformed_df = self.transform(filtered_df)

        # Save intermediary data
        output_path = self.save_intermediary_data(transformed_df, year)

        logger.info(f"Completed processing {self.file_type} for year {year}. Output: {output_path}")
        return output_path
