# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Compute the three chemical embeddings for Candidate nodes and index them.

Reads the SMILES-bearing `:Candidate` nodes (populated by resolve_candidate_smiles),
computes Morgan / MolDeBERTa / Hybrid embeddings via the same pipeline as `:Chemical`,
stores them on the nodes, and creates the three Candidate vector indexes + the
uniqueness constraint. Additive and downstream-safe: it only writes new `:Candidate`
properties/indexes — existing `:Chemical` queries and the scored metrics are untouched.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-24
"""

import logging

import pandas as pd

from src.knowledge_graph.load.vector_index_manager import VectorIndexManager
from src.neo4j_connector import Neo4jConnector

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

_FETCH = (
    "MATCH (c:Candidate) WHERE c.canonical_smiles IS NOT NULL "
    "RETURN c.cas_number AS cas_number, c.canonical_smiles AS canonical_smiles"
)


def main() -> None:
    with Neo4jConnector().session() as session:
        df = pd.DataFrame([dict(r) for r in session.run(_FETCH)])
    logger.info(f"{len(df)} candidates with SMILES to embed")
    if df.empty:
        logger.warning("No SMILES-bearing candidates yet — run resolve_candidate_smiles first")
        return

    vim = VectorIndexManager()
    vim.create_constraints()  # idempotent; adds candidate_cas_unique
    vim.create_candidate_indices()
    vim.embed_candidates(df)


if __name__ == "__main__":
    main()
