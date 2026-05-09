# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Module for fetching NAICS code descriptions from the U.S. Census Bureau API.

Notes:
    - The API key must be stored in an `.env` file with the key `CENSUS_DATA_API_KEY`.

Author: Jose D. Hernandez-Betancur
Date: 2026-04-24

"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Union

import aiohttp
import pandas as pd
from dotenv import load_dotenv

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)

_TRANSIENT_ERRORS = (aiohttp.ServerDisconnectedError, aiohttp.ClientConnectionError, asyncio.TimeoutError)


class NaicsDataFetcher:
    """Singleton class for fetching NAICS code descriptions from the Census API with caching."""

    def __new__(cls):
        """Create or return existing singleton instance."""
        if not hasattr(cls, "instance"):
            cls.instance = super(NaicsDataFetcher, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        self.census_cfg = ConfigManager().config["WEB_SITES"]["CENSUS_API"]
        self.us_spending_cfg = ConfigManager().config["WEB_SITES"]["US_SPENDING_API"]
        self.census_api_key = self._load_api_key()
        self.base_url = f"{self.census_cfg['BASE_URL']}/{self.census_cfg['DATASET']}"
        self.max_retries: int = self.census_cfg["MAX_RETRIES"]
        self.time = f"{datetime.now().year}-01"
        self._naics_cache: Dict[str, Optional[str]] = {}

    def _load_api_key(self) -> str:
        load_dotenv()
        api_key = os.getenv("CENSUS_DATA_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "CENSUS_DATA_API_KEY not found. Please ensure the .env file is present and contains the API key."
            )
        return api_key

    async def _fetch_from_auxiliar_endpoint(
        self,
        naics_code: str,
        session: aiohttp.ClientSession,
    ) -> Union[str, None]:
        query_url = self.us_spending_cfg["BASE_URL"].format(naics_code=naics_code)
        async with session.get(query_url) as response:
            if response.status == 200:
                data = await response.json()
                return data["results"][0]["naics_description"].capitalize()
            else:
                logger.error(f"Failed to fetch data for NAICS code {naics_code}: {response.status}")
                return None

    async def _fetch_with_retry(
        self,
        naics_code: str,
        full_url: str,
        session: aiohttp.ClientSession,
    ) -> Optional[str]:
        for attempt in range(self.max_retries):
            try:
                async with session.get(full_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data[1][0].capitalize() if len(data) > 1 else None
                    return await self._fetch_from_auxiliar_endpoint(naics_code, session)
            except _TRANSIENT_ERRORS as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    logger.warning(f"Retry {attempt + 2}/{self.max_retries} for NAICS {naics_code}: {e}")
                else:
                    logger.error(f"NAICS {naics_code} failed after {self.max_retries} attempts: {e}")
                    raise

    async def _fetch_single_naics_data(
        self,
        naics_code: str,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Optional[str]]:
        if naics_title := self._naics_cache.get(naics_code):
            return {"naics_code": naics_code, "naics_title": naics_title}

        parameters = self.census_cfg["PARAMETERS"]
        full_url = (
            f"{self.base_url}"
            f"?get={"&".join(parameters['GET']).format(time=self.time)}"
            f"&{parameters['NAICS_CODE'].format(naics_code=naics_code)}"
            f"&key={self.census_api_key}"
        )
        async with semaphore:
            naics_title = await self._fetch_with_retry(naics_code, full_url, session)
        self._naics_cache[naics_code] = naics_title
        return {"naics_code": naics_code, "naics_title": naics_title}

    async def _fetch_all_naics_data(
        self,
        naics_codes: List[str],
    ) -> List[Dict[str, Optional[str]]]:
        semaphore = asyncio.Semaphore(self.census_cfg["MAX_CONCURRENT_REQUESTS"])
        async with aiohttp.ClientSession() as session:
            tasks = [self._fetch_single_naics_data(code, session, semaphore) for code in naics_codes]
            return await asyncio.gather(*tasks)

    def process_naics_codes(
        self,
        df: pd.DataFrame,
        code_column: str,
    ) -> pd.DataFrame:
        """Process NAICS codes in a DataFrame and return a DataFrame with fetched data.

        Args:
            df (pd.DataFrame): The DataFrame containing NAICS codes.
            code_column (str): The name of the column containing the NAICS codes.

        Returns:
            pd.DataFrame: A DataFrame containing `naics_code` and `naics_title`.
        """
        unique_naics_codes = df[code_column].unique().tolist()
        data = asyncio.run(self._fetch_all_naics_data(unique_naics_codes))
        result_df = pd.DataFrame(data)
        result_df["naics_title"] = result_df["naics_title"].str.capitalize()
        return result_df
