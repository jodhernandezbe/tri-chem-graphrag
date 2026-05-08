# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Knowledge Graph Pipeline Orchestrator for TRI data processing.

This module orchestrates the complete extract-transform-load pipeline
for generating the TRI knowledge graph.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-02
"""

import argparse
import logging
import sys
from typing import List

from .extract.tri_scrapper import TRIScrapper
from .load import GraphLoader
from .transform.tri_1a_transformer import Tri1aTransformer
from .transform.tri_1b_transformer import Tri1bTransformer
from .transform.tri_2a_transformer import Tri2aTransformer
from .year_tracker import YearTracker

PIPELINE_STEPS: List[str] = ["extract", "transform", "load"]

logger = logging.getLogger(__name__)


class KnowledgeGraphPipeline:
    """Orchestrates the complete TRI knowledge graph generation pipeline."""

    def __init__(self, years: List[str]) -> None:
        self.years = years
        self.scrapper = TRIScrapper()
        self.transformers = {
            "1a": Tri1aTransformer(),
            "1b": Tri1bTransformer(),
            "2a": Tri2aTransformer(),
        }
        self._graph_loader = GraphLoader()
        self._tracker = YearTracker()

    def extract(self) -> None:
        """Extract raw TRI data files for all configured years."""
        logger.info("Starting raw data extraction")
        for year in self.years:
            if self._tracker.is_step_done(year, "extract"):
                logger.info(f"Year {year} extract already completed — skipping")
                continue
            try:
                self.scrapper.extract_tri_data_by_year(year)
                self._tracker.mark_step_done(year, "extract")
                logger.info(f"Successfully extracted TRI data for year {year}")
            except Exception as e:
                logger.error(f"Error extracting year {year}: {e}")
        logger.info(f"Completed raw data extraction for {len(self.years)} years")

    def transform(self) -> None:
        """Transform raw TRI data for all configured years."""
        logger.info("Starting data transformation")
        for year in self.years:
            if self._tracker.is_step_done(year, "transform"):
                logger.info(f"Year {year} transform already completed — skipping")
                continue
            for file_type, transformer in self.transformers.items():
                try:
                    transformer.process_year(year)
                    logger.info(f"Successfully transformed {file_type} for year {year}")
                except FileNotFoundError as e:
                    logger.warning(f"Raw data file not found for {file_type} year {year}: {e}")
                except Exception as e:
                    logger.error(f"Error transforming {file_type} for year {year}: {e}")
            self._tracker.mark_step_done(year, "transform")
        logger.info(f"Completed data transformation for {len(self.years)} years")

    def load(self) -> None:
        """Bulk-load transformed TRI data into Neo4j for all configured years."""
        logger.info("Starting data loading into Neo4j")
        for year in self.years:
            if self._tracker.is_step_done(year, "load"):
                logger.info(f"Year {year} load already completed — skipping")
                continue
            try:
                self._graph_loader.load(year)
                self._tracker.mark_step_done(year, "load")
                logger.info(f"Successfully loaded data for year {year}")
            except Exception as e:
                logger.error(f"Error loading year {year}: {e}")
        logger.info(f"Completed data loading for {len(self.years)} years")

    def run(self) -> None:
        """Run the complete extract-transform-load pipeline."""
        logger.info("Starting complete TRI knowledge graph pipeline")
        try:
            self.extract()
            self.transform()
            self.load()
            logger.info("Successfully completed pipeline")
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )  # type: ignore[reportUnknownArgumentType]

    parser = argparse.ArgumentParser(description="TRI Chemical GraphRAG ETL pipeline")
    parser.add_argument(
        "--years",
        nargs="+",
        required=True,
        metavar="YEAR",
        help="Reporting years to process, e.g. 2022 2023",
    )
    args = parser.parse_args()

    pipeline = KnowledgeGraphPipeline(args.years)
    pipeline.run()
