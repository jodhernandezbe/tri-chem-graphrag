# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Graph loading orchestrator for the TRI Knowledge Graph.

Reads transformed CSVs from data/processed/ and drives the full loading
sequence: schema setup → nodes → edges (dependency-ordered) → embeddings.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-07
"""

import logging
import pathlib
from typing import List, Optional

import pandas as pd
from pandas import DataFrame

from .edge_loader import EdgeLoader
from .node_loader import NodeLoader
from .vector_index_manager import VectorIndexManager

CURRENT_DIR = pathlib.Path(__file__).parent
PROCESSED_DATA_DIR = CURRENT_DIR.parent.parent.parent / "data" / "processed"

logger = logging.getLogger(__name__)


class GraphLoader:
    """Orchestrates bulk-loading of transformed TRI data into the Neo4j knowledge graph."""

    def __init__(self) -> None:
        self._nodes = NodeLoader()
        self._edges = EdgeLoader()
        self._index_manager = VectorIndexManager()

    def _read_processed(self, year: str, file_type: str) -> Optional[pd.DataFrame]:
        path = PROCESSED_DATA_DIR / f"tri_{file_type}_{year}_transformed.csv"
        if not path.exists():
            logger.warning(f"Processed file not found, skipping: {path}")
            return None
        return pd.read_csv(path, low_memory=False)

    def _get_contributing_dcns(self, df_2a: pd.DataFrame) -> List:
        """Return the list of document control numbers that have at least one significant source reduction."""
        valid = df_2a.dropna(subset=["source_reduction_code"])
        return list(valid["document_control_number"].dropna().unique())

    def _filter_by_dcn(self, df: pd.DataFrame, dcns: List) -> DataFrame:
        mask = df["document_control_number"].isin(dcns)
        return df.loc[mask].copy()  # type: ignore[return-value]

    def _load_1a(self, df: pd.DataFrame) -> None:
        # Nodes must be created before the edges that reference them.
        # FILED_BY must be written before RECORDS (RECORDS traverses it).
        self._nodes.load_chemicals(df)
        self._nodes.load_facilities(df)
        self._nodes.load_sectors(df)
        self._nodes.load_reports(df)
        self._edges.load_filed_by(df)
        self._edges.load_classified_as(df)
        self._edges.load_reports_on(df)

    def _load_1b(self, df: pd.DataFrame) -> None:
        self._nodes.load_activities(df)
        self._edges.load_records(df)

    def _load_2a(self, df: pd.DataFrame) -> None:
        valid = df.dropna(subset=["source_reduction_code"])
        self._nodes.load_reductions(valid)
        self._edges.load_documents(valid)

    def _load_year(self, year: str) -> Optional[pd.DataFrame]:
        df_1a = self._read_processed(year, "1a")
        df_1b = self._read_processed(year, "1b")
        df_2a = self._read_processed(year, "2a")

        if df_2a is None:
            logger.warning(f"No 2a data for {year} — nothing to load (no contributing documents)")
            return None

        contributing_dcns = self._get_contributing_dcns(df_2a)
        if not contributing_dcns:
            logger.warning(f"No significant source reductions found for {year} — skipping all loads")
            return None

        logger.info(f"Found {len(contributing_dcns)} documents with significant source reductions for {year}")

        if df_1a is not None:
            self._load_1a(self._filter_by_dcn(df_1a, contributing_dcns))
        if df_1b is not None:
            self._load_1b(self._filter_by_dcn(df_1b, contributing_dcns))
        self._load_2a(df_2a)

        return df_1a if df_1a is not None else None

    def _embed_all(self, frames_1a: List[pd.DataFrame]) -> None:
        if not frames_1a:
            logger.warning("No 1a data available — skipping embedding computation")
            return
        combined = pd.concat(frames_1a, ignore_index=True)
        self._index_manager.embed_chemicals(combined)
        self._index_manager.embed_sectors(combined)

    def load(self, year: str) -> None:
        """Bulk-load all transformed TRI data for the given year into Neo4j.

        Creates schema (constraints + vector indices) if not yet present, loads
        nodes and edges in dependency order, then computes vector embeddings.

        Args:
            year (str): Reporting year to load, e.g. "2022".
        """
        logger.info(f"Starting graph load for year {year}")
        self._index_manager.setup_schema()
        df_1a = self._load_year(year)
        self._embed_all([df_1a] if df_1a is not None else [])
        logger.info(f"Graph load complete for year {year}")
