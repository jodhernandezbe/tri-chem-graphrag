# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Graph traversal retriever using LangChain's Neo4jGraph connector.

Runs node-type-specific Cypher queries via Neo4jGraph to expand a seed node
into a subgraph containing its related facilities, sectors, and reduction records.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-09

"""

import logging
import os
from typing import Any, Dict, List

from langchain_neo4j import Neo4jGraph

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)

CHEMICAL_TRAVERSAL = """
MATCH (c:Chemical {cas_number: $key})
OPTIONAL MATCH (r:Report)-[:REPORTS_ON]->(c)
OPTIONAL MATCH (r)-[:FILED_BY]->(f:Facility)
OPTIONAL MATCH (f)-[:CLASSIFIED_AS]->(s:Sector)
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WITH c,
     collect(DISTINCT {
         year:           r.reporting_year,
         facility:       f.trifid,
         state:          f.state,
         sector_code:    s.naics_code,
         sector_title:   s.naics_title,
         reduction_code: red.code,
         reduction_name: red.name
     })[..$limit] AS reports
RETURN {
    node_type:   'Chemical',
    name:        c.name,
    cas_number:  c.cas_number,
    smiles:      c.canonical_smiles,
    tri_chem_id: c.tri_chem_id,
    reports:     reports
} AS subgraph
"""

SECTOR_TRAVERSAL = """
MATCH (s:Sector {naics_code: $key})
OPTIONAL MATCH (f:Facility)-[:CLASSIFIED_AS]->(s)
OPTIONAL MATCH (r:Report)-[:FILED_BY]->(f)
OPTIONAL MATCH (r)-[:REPORTS_ON]->(c:Chemical)
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WITH s,
     collect(DISTINCT {
         facility:       f.trifid,
         state:          f.state,
         chemical:       c.name,
         cas_number:     c.cas_number,
         year:           r.reporting_year,
         reduction_code: red.code,
         reduction_name: red.name
     })[..$limit] AS context
RETURN {
    node_type:   'Sector',
    naics_code:  s.naics_code,
    naics_title: s.naics_title,
    context:     context
} AS subgraph
"""

NODE_KEY: Dict[str, str] = {
    "Chemical": "cas_number",
    "Sector": "naics_code",
}

TRAVERSAL_QUERY: Dict[str, str] = {
    "Chemical": CHEMICAL_TRAVERSAL,
    "Sector": SECTOR_TRAVERSAL,
}


class GraphRetriever:
    """Traverses the TRI knowledge graph via Neo4jGraph from a seed node."""

    def __init__(
        self,
        node_label: str,
        limit: int = 50,
    ) -> None:
        if node_label not in TRAVERSAL_QUERY:
            raise ValueError(f"Unsupported node label for graph traversal: '{node_label}'")
        self.node_label = node_label
        self.limit = limit
        cfg = ConfigManager().config["DUMMY_NEO4J_CONFIG"]
        self._graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", cfg["NEO4J_URI"]),
            username=os.getenv("NEO4J_USER", cfg["NEO4J_USER"]),
            password=os.getenv("NEO4J_PASSWORD", cfg["NEO4J_PASSWORD"]),
        )

    def traverse(
        self,
        node: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Expand a seed node into a subgraph dict via Cypher.

        Args:
            node (Dict[str, Any]): Node properties dict containing the key field
                (e.g. ``cas_number`` for Chemical, ``naics_code`` for Sector).

        Returns:
            Dict[str, Any]: Subgraph dictionary, or empty dict if no record found.
        """
        key_prop = NODE_KEY[self.node_label]
        key_value = node.get(key_prop)
        if key_value is None:
            logger.warning(f"Node missing '{key_prop}', skipping traversal: {node}")
            return {}

        query = TRAVERSAL_QUERY[self.node_label]
        results = self._graph.query(query, {"key": key_value, "limit": self.limit})

        if not results:
            return {}

        return dict(results[0].get("subgraph", {}))

    def traverse_all(
        self,
        nodes: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Traverse all seed nodes and return their subgraphs.

        Args:
            nodes (List[Dict[str, Any]]): Seed node property dicts.

        Returns:
            List[Dict[str, Any]]: Non-empty subgraph dicts.
        """
        return [sg for node in nodes if (sg := self.traverse(node))]
