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
from typing import Any, Dict, List, Optional, Tuple

from langchain_neo4j import Neo4jGraph

from src.config_manager import ConfigManager

logger = logging.getLogger(__name__)

CHEMICAL_TRAVERSAL = """
MATCH (c:Chemical {cas_number: $key})
OPTIONAL MATCH (r:Report)-[:REPORTS_ON]->(c)
OPTIONAL MATCH (r)-[:FILED_BY]->(f:Facility)
OPTIONAL MATCH (f)-[:CLASSIFIED_AS]->(s:Sector)
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE size($activity_names) = 0
   OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names }
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
WHERE size($activity_names) = 0
   OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names }
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

SMILES_LOOKUP = """
MATCH (c:Chemical)
WHERE toLower(c.name) CONTAINS toLower($query)
   OR c.cas_number = $query
   OR c.canonical_smiles = $query
RETURN c.canonical_smiles AS smiles, c.name AS name, c.cas_number AS cas_number
ORDER BY size(c.name) ASC
LIMIT 1
"""

CROSS_ENTITY_TRAVERSAL = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE c.cas_number IN $cas_numbers AND s.naics_code IN $naics_codes
  AND (size($activity_names) = 0
       OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names })
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WITH c, s,
     collect(DISTINCT {
         year:           r.reporting_year,
         facility:       f.trifid,
         state:          f.state,
         reduction_code: red.code,
         reduction_name: red.name
     })[..$limit] AS reports
RETURN {
    node_type:    'Chemical_Sector',
    name:         c.name,
    cas_number:   c.cas_number,
    smiles:       c.canonical_smiles,
    sector_code:  s.naics_code,
    sector_title: s.naics_title,
    reports:      reports
} AS subgraph
"""

CANDIDATE_CATEGORIES = """
MATCH (cat:Chemical)-[:CONTAINS]->(m:Candidate)
WHERE m.cas_number IN $cas_numbers
RETURN m.cas_number AS member, m.name AS member_name, cat.cas_number AS category
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
        activity_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Expand a seed node into a subgraph dict via Cypher.

        Args:
            node (Dict[str, Any]): Node properties dict containing the key field
                (e.g. ``cas_number`` for Chemical, ``naics_code`` for Sector).
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the traversal to. An empty/``None`` list disables the filter.

        Returns:
            Dict[str, Any]: Subgraph dictionary, or empty dict if no record found.
        """
        key_prop = NODE_KEY[self.node_label]
        key_value = node.get(key_prop)
        if key_value is None:
            logger.warning(f"Node missing '{key_prop}', skipping traversal: {node}")
            return {}

        query = TRAVERSAL_QUERY[self.node_label]
        results = self._graph.query(
            query,
            {"key": key_value, "limit": self.limit, "activity_names": activity_names or []},
        )

        if not results:
            return {}

        return dict(results[0].get("subgraph", {}))

    def traverse_all(
        self,
        nodes: List[Dict[str, Any]],
        activity_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse all seed nodes and return their subgraphs.

        Args:
            nodes (List[Dict[str, Any]]): Seed node property dicts.
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the traversal to. An empty/``None`` list disables the filter.

        Returns:
            List[Dict[str, Any]]: Non-empty subgraph dicts.
        """
        return [sg for node in nodes if (sg := self.traverse(node, activity_names))]


class CrossEntityRetriever:
    """Resolves SMILES identifiers and traverses chemical–sector intersections."""

    def __init__(self, limit: int = 50) -> None:
        self.limit = limit
        cfg = ConfigManager().config["DUMMY_NEO4J_CONFIG"]
        self._graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", cfg["NEO4J_URI"]),
            username=os.getenv("NEO4J_USER", cfg["NEO4J_USER"]),
            password=os.getenv("NEO4J_PASSWORD", cfg["NEO4J_PASSWORD"]),
        )

    def resolve_smiles(self, query: str) -> str:
        """Return the canonical SMILES for a chemical name, CAS number, or SMILES string.

        Args:
            query (str): Chemical name, CAS number, or SMILES to look up.

        Returns:
            str: Canonical SMILES string, or empty string if not found.
        """
        results = self._graph.query(SMILES_LOOKUP, {"query": query})
        if results and results[0].get("smiles"):
            smiles = str(results[0]["smiles"])
            logger.info(f"Resolved '{query}' → SMILES '{smiles}' ({results[0].get('name')})")
            return smiles
        logger.warning(f"Could not resolve SMILES for query: '{query}'")
        return ""

    def traverse_cross(
        self,
        cas_numbers: List[str],
        naics_codes: List[str],
        activity_names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Find subgraphs at the intersection of chemicals and sectors.

        Args:
            cas_numbers (List[str]): CAS numbers from chemical vector retrieval.
            naics_codes (List[str]): NAICS codes from sector vector retrieval.
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the intersection to. An empty/``None`` list disables the filter.

        Returns:
            List[Dict[str, Any]]: Cross-entity subgraph dicts, empty if no intersection.
        """
        results = self._graph.query(
            CROSS_ENTITY_TRAVERSAL,
            {
                "cas_numbers": cas_numbers,
                "naics_codes": naics_codes,
                "activity_names": activity_names or [],
                "limit": self.limit,
            },
        )
        return [dict(r["subgraph"]) for r in results if r.get("subgraph")]

    def candidate_categories(self, cas_numbers: List[str]) -> Dict[str, Tuple[str, str]]:
        """Map candidate CAS numbers to their containing category and the member's name.

        Args:
            cas_numbers (List[str]): Candidate CAS numbers from candidate vector search.

        Returns:
            Dict[str, Tuple[str, str]]: ``{candidate cas_number: (category cas_number,
                member name)}`` for candidates that belong to a category in the graph.
        """
        if not cas_numbers:
            return {}
        results = self._graph.query(CANDIDATE_CATEGORIES, {"cas_numbers": cas_numbers})
        return {r["member"]: (r["category"], r["member_name"]) for r in results}
