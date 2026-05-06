# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""TRI Scrapper Module.

This module contains the TRIScrapper class, which is used to scrape the TRI data from the EPA's TRI database.

Note: This code is a refactoring of the code used in https://github.com/jodhernandezbe/EoL4Chem (2018-2020 project)

Author: Jose D. Hernandez-Betancur
Date: 2026-04-23

"""

import io
import logging
import pathlib
import zipfile
from collections import defaultdict
from typing import IO, Any, Dict, List, Optional
from urllib.parse import urljoin

import chardet
import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config_manager import ConfigManager

CURRENT_DIR = pathlib.Path(__file__).parent
WORKING_DIR = CURRENT_DIR.parent.parent.parent
ANCILLARY_DIR = WORKING_DIR / "ancillary"
RAW_DATA_DIR = WORKING_DIR / "data" / "raw"
EPA_BASE_URL = "https://www.epa.gov"

TRI_FILE_TYPES = [
    "1a",  # Releases and Waste Management
    "1b",  # Chemical Activity and Use
    "2a",  # Source Reduction
]


logger = logging.getLogger(__name__)


class TRIScrapper:
    """Singleton class for scraping TRI data from EPA's TRI database."""

    def __new__(cls):
        """Create or return existing singleton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(TRIScrapper, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self._config: Dict[str, Any] = ConfigManager().config["WEB_SITES"]["TRI"]
        self._queries: Dict[str, str] = self._config["QUERIES"]
        self._url: str = self._config["URL"]
        self._tri_zip_options: Optional[Dict[str, str]] = None
        self._calling_tri_columns()

    def _visit(self) -> None:
        logger.info(f"Visiting {self._url}")
        html = requests.get(self._url, timeout=30).text
        self._soup = BeautifulSoup(html, "html.parser")

    @property
    def tri_zip_options(self) -> Dict[str, str]:
        """Get available TRI zip file options from EPA website.

        Returns:
            Dict[str, str]: Mapping of years to zip file URLs.
        """
        if self._tri_zip_options is None:
            logger.info("Getting TRI zip options")

            self._visit()
            self._tri_zip_options = {}
            for option in self._soup.select(self._queries["CSS_TO_OPTIONS"]):
                logger.info(f"Processing option: {option.get_text(strip=True)}")

                raw_ref = option.get("value")
                if isinstance(raw_ref, str):
                    year = option.get_text(strip=True)
                    self._tri_zip_options[year] = raw_ref
                else:
                    continue

        return self._tri_zip_options

    def _calling_tri_columns(self) -> None:
        self._tri_file_columns_dictionary: Dict[str, List[str]] = defaultdict(list)

        for file_type in TRI_FILE_TYPES:
            file_path = ANCILLARY_DIR / "tri" / "columns" / f"tri_file_{file_type}_columns.txt"
            if not file_path.exists():
                raise FileNotFoundError(f"File not found at {file_path}")

            with open(file_path, "rb") as f:
                inf_chardet = chardet.detect(f.read())
            inf_encoding = inf_chardet.get("encoding", "utf-8")
            with open(file_path, "r", encoding=inf_encoding) as file:
                self._tri_file_columns_dictionary[file_type] = [line.rstrip().replace("\n", "") for line in file.readlines()]

    def extract_tri_data_by_year(self, year: str) -> None:
        """Extract the TRI data for a given year from the TRI zip file.

        Args:
            year (str): The year to extract the TRI data for.

        Returns:
            None

        """
        logger.info(f"Extracting TRI data for year {year}")

        zip_url = self.tri_zip_options.get(year)
        if zip_url is None:
            raise ValueError(f"Year {year} not found in TRI zip options")

        zip_file = self._get_zip_file(zip_url)
        with zipfile.ZipFile(io.BytesIO(zip_file)) as zip_file_object:
            self._extract_file_types_from_zip(zip_file_object, year)

    def extract_tri_data_all_years(self, years: List[str]) -> None:
        """Extract the TRI data for all years from the TRI zip file.

        Args:
            years (List[str]): The years to extract the TRI data for.

        Returns:
            None
        """
        for year in years:
            self.extract_tri_data_by_year(year)

    def _get_zip_file(self, zip_url: str) -> bytes:
        # Convert relative URLs to absolute URLs
        if zip_url.startswith("/"):
            base_url = EPA_BASE_URL
            zip_url = urljoin(base_url, zip_url)

        response = requests.get(zip_url, timeout=120)

        if not response.ok:
            raise ValueError(f"Failed to download zip file from {zip_url}")

        return response.content

    def _extract_file_types_from_zip(
        self,
        zip_file: zipfile.ZipFile,
        year: str,
    ) -> None:
        for file_type in TRI_FILE_TYPES:
            logger.info(f"Extracting and saving file type {file_type} for year {year}")

            if not zip_file.getinfo(f"US_{file_type}_{year}.txt"):
                raise FileNotFoundError(f"File not found in zip file: US_{file_type}_{year}.txt")

            if pathlib.Path(RAW_DATA_DIR / f"tri_{file_type}_{year}.csv").exists():
                logger.info(f"File already exists: US_{file_type}_{year}.csv")
                continue

            with zip_file.open(f"US_{file_type}_{year}.txt") as txt_file:
                df_file_type = self._convert_txt_to_csv(txt_file, file_type)
                self._save_csv_file(df_file_type, file_type, year)

    def _save_csv_file(
        self,
        df: pd.DataFrame,
        file_type: str,
        year: str,
    ) -> None:
        csv_file_path = RAW_DATA_DIR / f"tri_{file_type}_{year}.csv"
        df.to_csv(
            csv_file_path,
            index=False,
            sep=",",
            encoding="utf-8",
        )

    def _convert_txt_to_csv(
        self,
        txt_file: IO[bytes],
        file_type: str,
    ) -> pd.DataFrame:
        columns = self._tri_file_columns_dictionary[file_type]
        df = pd.read_csv(
            txt_file,
            header=None,
            encoding="ISO-8859-1",
            on_bad_lines="skip",
            sep="\t",
            low_memory=False,
            skiprows=[0],
            lineterminator="\n",
            usecols=range(len(columns)),
        )
        df.columns = columns
        return df
