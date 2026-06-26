# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""One-off migration: re-point RECORDS edges from Facility to Report.

The canonical ontology (graph_model.svg) attaches the condition-of-use Activity to
the originating Report (``Report-[:RECORDS]->Activity``). Earlier loads incorrectly
attached it to the Facility (``Facility-[:RECORDS]->Activity``), discarding the
report association. The facility-level edge cannot be re-pointed in-graph because the
report link was lost at load time, so this migration rebuilds the edge from the 1b
source data (which carries ``document_control_number`` + ``chemical_activity``) and
then removes the stale facility-level edges.

Idempotent: re-running creates no duplicate edges (MERGE) and the cleanup is a no-op
once the facility-level edges are gone.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-24
"""

import glob
import logging
import os
from typing import Dict, List

import pandas as pd
from langchain_neo4j import Neo4jGraph

from src.config_manager import ConfigManager

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# Years actually loaded into the graph; do not introduce activities for unloaded reports.
LOADED_YEARS: range = range(2014, 2025)
BATCH_SIZE: int = 1000

_CREATE_REPORT_RECORDS = f"""
UNWIND $rows AS row
CALL {{
    WITH row
    MATCH (r:Report {{document_control_number: row.document_control_number}})
    MATCH (a:Activity {{name: row.chemical_activity}})
    MERGE (r)-[:RECORDS]->(a)
}} IN TRANSACTIONS OF {BATCH_SIZE} ROWS
"""

_DROP_FACILITY_RECORDS = """
MATCH (f:Facility)-[rel:RECORDS]->(:Activity)
CALL {
    WITH rel
    DELETE rel
} IN TRANSACTIONS OF 5000 ROWS
"""

_COUNT_BY_ENDPOINT = """
MATCH (x)-[:RECORDS]->(:Activity)
RETURN labels(x)[0] AS from_label, count(*) AS n
ORDER BY n DESC
"""


def _graph() -> Neo4jGraph:
    cfg = ConfigManager().config["DUMMY_NEO4J_CONFIG"]
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI", cfg["NEO4J_URI"]),
        username=os.getenv("NEO4J_USER", cfg["NEO4J_USER"]),
        password=os.getenv("NEO4J_PASSWORD", cfg["NEO4J_PASSWORD"]),
    )


def _load_1b_rows() -> List[Dict[str, str]]:
    """Read the 1b transformed rows for the loaded years.

    Returns:
        List[Dict[str, str]]: Deduplicated (document_control_number, chemical_activity) rows.
    """
    frames: List[pd.DataFrame] = []
    for year in LOADED_YEARS:
        matches = glob.glob(f"data/processed/tri_1b_{year}_transformed.csv")
        for path in matches:
            df = pd.read_csv(path)
            frames.append(df)
            logger.info(f"Loaded {len(df):>8} rows from {os.path.basename(path)}")
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["document_control_number", "chemical_activity"])
    combined = combined.drop_duplicates(subset=["document_control_number", "chemical_activity"])
    # Report.document_control_number is stored as a float in the graph; Neo4j indexes
    # int and float keys separately, so the lookup value must be cast to float to match.
    combined["document_control_number"] = combined["document_control_number"].astype(float)
    combined["chemical_activity"] = combined["chemical_activity"].astype(str)
    logger.info(f"Total unique (report, activity) pairs to write: {len(combined)}")
    return combined.to_dict("records")  # type: ignore[return-value]


def main() -> None:
    graph = _graph()
    logger.info(f"Before: {graph.query(_COUNT_BY_ENDPOINT)}")

    rows = _load_1b_rows()
    graph.query(_CREATE_REPORT_RECORDS, {"rows": rows})
    after_create = graph.query(_COUNT_BY_ENDPOINT)
    logger.info(f"After create: {after_create}")

    report_edges = next((c["n"] for c in after_create if c["from_label"] == "Report"), 0)
    if report_edges == 0:
        raise RuntimeError("No Report-[:RECORDS]->Activity edges created; aborting before cleanup.")

    graph.query(_DROP_FACILITY_RECORDS)
    logger.info(f"After cleanup: {graph.query(_COUNT_BY_ENDPOINT)}")


if __name__ == "__main__":
    main()
