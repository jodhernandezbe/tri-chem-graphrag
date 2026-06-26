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
from typing import List, Optional, Tuple, cast

import pandas as pd
from pandas import DataFrame

from src.neo4j_connector import Neo4jConnector

from .edge_loader import EdgeLoader
from .node_loader import NodeLoader
from .vector_index_manager import VectorIndexManager

CURRENT_DIR = pathlib.Path(__file__).parent
PROCESSED_DATA_DIR = CURRENT_DIR.parent.parent.parent / "data" / "processed"
ANCILLARY_DIR = CURRENT_DIR.parent.parent.parent / "ancillary"

# chemicals_in_categories.csv → snake_case columns for the Candidate / CONTAINS loaders.
_CATEGORY_COLUMNS = {
    "CAS NUMBER": "cas_number",
    "CATEGORY CODE": "category_code",
    "CATEGORY NAME": "category_name",
    "CHEMICAL NAME": "chemical_name",
    "CHEMICAL TYPE": "chemical_type",
    "SOURCE": "source",
}

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
        # Reports (1a) must exist before RECORDS (1b), which attaches to them.
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

    def _load_year(self, year: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        df_1a = self._read_processed(year, "1a")
        df_1b = self._read_processed(year, "1b")
        df_2a = self._read_processed(year, "2a")

        if df_2a is None:
            logger.warning(f"No 2a data for {year} — nothing to load (no contributing documents)")
            return None, None

        contributing_dcns = self._get_contributing_dcns(df_2a)
        if not contributing_dcns:
            logger.warning(f"No significant source reductions found for {year} — skipping all loads")
            return None, None

        logger.info(f"Found {len(contributing_dcns)} documents with significant source reductions for {year}")

        df_1b_filtered: Optional[pd.DataFrame] = None
        if df_1a is not None:
            self._load_1a(self._filter_by_dcn(df_1a, contributing_dcns))
        if df_1b is not None:
            df_1b_filtered = self._filter_by_dcn(df_1b, contributing_dcns)
            self._load_1b(df_1b_filtered)
        self._load_2a(df_2a)

        return df_1a, df_1b_filtered

    def _embed_all(self, frames_1a: List[pd.DataFrame], frames_1b: List[pd.DataFrame]) -> None:
        if frames_1a:
            combined_1a = pd.concat(frames_1a, ignore_index=True)
            self._index_manager.embed_chemicals(combined_1a)
            self._index_manager.embed_sectors(combined_1a)
        else:
            logger.warning("No 1a data available — skipping chemical/sector embedding computation")
        if frames_1b:
            combined_1b = pd.concat(frames_1b, ignore_index=True)
            self._index_manager.embed_activities(combined_1b)
        else:
            logger.warning("No 1b data available — skipping Activity embedding computation")

    def load(self, year: str) -> None:
        """Bulk-load all transformed TRI data for the given year into Neo4j.

        Creates schema (constraints + vector indices) if not yet present, loads
        nodes and edges in dependency order, then computes vector embeddings.

        Args:
            year (str): Reporting year to load, e.g. "2022".
        """
        logger.info(f"Starting graph load for year {year}")
        self._index_manager.setup_schema()
        df_1a, df_1b = self._load_year(year)
        self._embed_all(
            [df_1a] if df_1a is not None else [],
            [df_1b] if df_1b is not None else [],
        )
        logger.info(f"Graph load complete for year {year}")

    def _existing_chemical_cas(self) -> List[str]:
        with Neo4jConnector().session() as session:
            return [r["cas"] for r in session.run("MATCH (c:Chemical) RETURN c.cas_number AS cas")]

    def load_candidates(self, csv_path: Optional[pathlib.Path] = None) -> None:
        """Load category-member Candidate nodes and CONTAINS edges from the ancillary CSV.

        Excludes members already present as ``:Chemical`` (no duplicates / no
        ``Chemical↔Chemical`` loops) and links only categories present in the graph, then
        creates the Candidate vector indices. Run once after the per-year loads — the
        category ``:Chemical`` nodes must already exist. SMILES resolution and the three
        embeddings are separate post-load steps (``scripts/resolve_candidate_smiles.py``,
        ``scripts/embed_candidates.py``), kept out of the ETL because they call PubChem.

        Args:
            csv_path (Optional[pathlib.Path]): Override the default
                ``ancillary/chemicals_in_categories.csv`` path.
        """
        path = csv_path or ANCILLARY_DIR / "chemicals_in_categories.csv"
        df = pd.read_csv(path, encoding="latin-1").rename(columns=_CATEGORY_COLUMNS)
        df = df.dropna(subset=["cas_number", "category_code"]).drop_duplicates()
        existing = self._existing_chemical_cas()

        present = cast(pd.DataFrame, df[df["category_code"].isin(existing)])  # category exists in the graph
        members = cast(pd.DataFrame, present[~present["cas_number"].isin(existing)])  # member is not reported
        logger.info(f"Loading {len(members)} category members across present categories (TRI-reported excluded)")

        self._nodes.load_candidates(members.drop_duplicates(subset=["cas_number"]))
        self._edges.load_contains(members)
        self._index_manager.create_candidate_indices()
