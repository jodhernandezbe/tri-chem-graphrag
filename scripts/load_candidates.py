# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Load category → Candidate expansion into Neo4j from chemicals_in_categories.csv.

Thin wrapper around ``GraphLoader.load_candidates`` (the ETL orchestration). Creates
`:Candidate` nodes (category members not directly reported in TRI) and
`Chemical (category) -[:CONTAINS]-> Candidate` edges, then the Candidate vector indices.
SMILES resolution and embeddings are separate post-load steps
(scripts/resolve_candidate_smiles.py, scripts/embed_candidates.py).

Safe to run alongside other work: it only writes new `:Candidate` nodes and `CONTAINS`
edges — disjoint from the `:Chemical` vector index and traversals used elsewhere.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-24
"""

import logging
from typing import Any

from src.knowledge_graph.load.graph_loader import GraphLoader
from src.neo4j_connector import Neo4jConnector

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def _scalar(session: Any, query: str) -> int:
    row = session.run(query).single()
    return int(row["n"]) if row else 0


def main() -> None:
    GraphLoader().load_candidates()

    with Neo4jConnector().session() as session:
        n_nodes = _scalar(session, "MATCH (c:Candidate) RETURN count(*) AS n")
        n_edges = _scalar(session, "MATCH (:Chemical)-[:CONTAINS]->(:Candidate) RETURN count(*) AS n")
        n_cats = _scalar(session, "MATCH (cat:Chemical)-[:CONTAINS]->(:Candidate) RETURN count(DISTINCT cat) AS n")
    logger.info(f"Graph now has {n_nodes} Candidate nodes, {n_edges} CONTAINS edges across {n_cats} categories")


if __name__ == "__main__":
    main()
