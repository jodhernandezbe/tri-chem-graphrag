# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Knowledge Graph Pipeline Orchestrator for TRI data processing.

This module orchestrates the complete extract-transform-load pipeline
for generating the TRI knowledge graph.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import logging
import pathlib
from typing import List

from .extract.tri_scrapper import TRIScrapper
from .transform.tri_1a_transformer import Tri1aTransformer
from .transform.tri_1b_transformer import Tri1bTransformer
from .transform.tri_2a_transformer import Tri2aTransformer

CURRENT_DIR = pathlib.Path(__file__).parent
WORKING_DIR = CURRENT_DIR.parent.parent
INTERMEDIARY_DATA_DIR = WORKING_DIR / "data" / "processed"
FILE_TYPES = ["1a", "1b", "2a"]

logger = logging.getLogger(__name__)


class KnowledgeGraphPipeline:
    """Orchestrates the complete TRI knowledge graph generation pipeline."""

    def __init__(self, years: List[str]):
        self.years = years
        self.scrapper = TRIScrapper()
        self.transformers = {"1a": Tri1aTransformer(), "1b": Tri1bTransformer(), "2a": Tri2aTransformer()}

    def extract(self) -> None:
        """Extract raw TRI data for specified years.

        Args:
            years: List of years to extract data for
        """
        logger.info("Starting raw data extraction")

        for year in self.years:
            logger.info(f"Extracting TRI data for year {year}")
            self.scrapper.extract_tri_data_by_year(year)

        logger.info(f"Completed raw data extraction for {len(self.years)} years")

    def transform(self) -> None:
        """Transform raw TRI data for specified years.

        Args:
            years (List[str]): List of years to transform data for
        """
        logger.info("Starting data transformation")

        for year in self.years:
            logger.info(f"Transforming data for year {year}")

            for file_type, transformer in self.transformers.items():
                try:
                    transformer.process_year(year)
                    logger.info(f"Successfully transformed {file_type} for year {year}")

                except FileNotFoundError as e:
                    logger.warning(f"Raw data file not found for {file_type} year {year}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error transforming {file_type} for year {year}: {e}")
                    continue

        logger.info(f"Completed data transformation for {len(self.years)} years")

    def run(self) -> None:
        """Run the complete extract-transform pipeline.

        Args:
            years (List[str]): List of years to process
        """
        logger.info("Starting complete TRI knowledge graph pipeline")

        try:
            # Step 1: Extract raw data
            self.extract()

            # Step 2: Transform data (includes enrichment)
            self.transform()

            logger.info("Successfully completed pipeline")

        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
