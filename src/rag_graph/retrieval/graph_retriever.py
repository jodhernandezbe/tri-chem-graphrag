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
WHERE $report_split IS NULL OR r.split = $report_split
OPTIONAL MATCH (r)-[:FILED_BY]->(f:Facility)
OPTIONAL MATCH (f)-[:CLASSIFIED_AS]->(s:Sector)
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE size($activity_names) = 0
   OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names }
WITH c,
     [x IN collect(DISTINCT {
         year:           r.reporting_year,
         facility:       f.trifid,
         state:          f.state,
         sector_code:    s.naics_code,
         sector_title:   s.naics_title,
         reduction_code: red.code,
         reduction_name: red.name
     }) WHERE x.reduction_code IS NOT NULL][..$limit] AS reports
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
WHERE $report_split IS NULL OR r.split = $report_split
OPTIONAL MATCH (r)-[:REPORTS_ON]->(c:Chemical)
OPTIONAL MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE size($activity_names) = 0
   OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names }
WITH s,
     [x IN collect(DISTINCT {
         facility:       f.trifid,
         state:          f.state,
         chemical:       c.name,
         cas_number:     c.cas_number,
         year:           r.reporting_year,
         reduction_code: red.code,
         reduction_name: red.name
     }) WHERE x.reduction_code IS NOT NULL][..$limit] AS context
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
  AND ($report_split IS NULL OR r.split = $report_split)
  AND (size($activity_names) = 0
       OR EXISTS { MATCH (r)-[:RECORDS]->(a:Activity) WHERE a.name IN $activity_names })
MATCH (r)-[:DOCUMENTS]->(red:Reduction)
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

CONTEXT_SECTOR_COUNTS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE s.naics_code IS NOT NULL AND ($report_split IS NULL OR r.split = $report_split)
RETURN c.cas_number AS cas, left(toString(toInteger(s.naics_code)), 4) AS key, count(DISTINCT r) AS n
"""

CONTEXT_COU_COUNTS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:RECORDS]->(a:Activity)
WHERE $report_split IS NULL OR r.split = $report_split
RETURN c.cas_number AS cas, a.name AS key, count(DISTINCT r) AS n
"""

CODE_CHEMICAL_COUNTS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE red.code IS NOT NULL AND ($report_split IS NULL OR r.split = $report_split)
RETURN red.code AS code, count(DISTINCT c) AS n_chemicals
"""

CHEMICALS_WITH_CODES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE red.code IS NOT NULL AND ($report_split IS NULL OR r.split = $report_split)
RETURN count(DISTINCT c) AS n_chemicals
"""

REDUCTION_RECORDS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[d:DOCUMENTS]->(red:Reduction)
WHERE red.code IS NOT NULL AND ($report_split IS NULL OR r.split = $report_split)
OPTIONAL MATCH (r)-[:FILED_BY]->(f:Facility)
OPTIONAL MATCH (f)-[:CLASSIFIED_AS]->(s:Sector)
OPTIONAL MATCH (r)-[:RECORDS]->(a:Activity)
RETURN elementId(r) AS report_id, r.document_control_number AS dcn, c.cas_number AS cas, c.name AS chemical,
       red.code AS code, red.name AS name, collect(DISTINCT a.name) AS uses, s.naics_code AS sector_code,
       s.naics_title AS sector_title, r.reporting_year AS year, f.state AS state, d.annual_reduction AS annual_reduction
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
        report_split: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Expand a seed node into a subgraph dict via Cypher.

        Args:
            node (Dict[str, Any]): Node properties dict containing the key field
                (e.g. ``cas_number`` for Chemical, ``naics_code`` for Sector).
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the traversal to. An empty/``None`` list disables the filter.
            report_split (Optional[str]): When set (e.g. ``'index'``), only reports whose
                ``split`` property equals it are read. ``None`` (default) reads all reports.

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
            {"key": key_value, "limit": self.limit, "activity_names": activity_names or [], "report_split": report_split},
        )

        if not results:
            return {}

        return dict(results[0].get("subgraph", {}))

    def traverse_all(
        self,
        nodes: List[Dict[str, Any]],
        activity_names: Optional[List[str]] = None,
        report_split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse all seed nodes and return their subgraphs.

        Args:
            nodes (List[Dict[str, Any]]): Seed node property dicts.
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the traversal to. An empty/``None`` list disables the filter.
            report_split (Optional[str]): Restrict to reports with this ``split`` value; ``None`` reads all.

        Returns:
            List[Dict[str, Any]]: Non-empty subgraph dicts.
        """
        return [sg for node in nodes if (sg := self.traverse(node, activity_names, report_split))]


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
        report_split: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find subgraphs at the intersection of chemicals and sectors.

        Args:
            cas_numbers (List[str]): CAS numbers from chemical vector retrieval.
            naics_codes (List[str]): NAICS codes from sector vector retrieval.
            activity_names (Optional[List[str]]): Condition-of-use activity names to
                restrict the intersection to. An empty/``None`` list disables the filter.
            report_split (Optional[str]): Restrict to reports with this ``split`` value; ``None`` reads all.

        Returns:
            List[Dict[str, Any]]: Cross-entity subgraph dicts, empty if no intersection.
        """
        results = self._graph.query(
            CROSS_ENTITY_TRAVERSAL,
            {
                "cas_numbers": cas_numbers,
                "naics_codes": naics_codes,
                "activity_names": activity_names or [],
                "report_split": report_split,
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

    def context_rows(self, report_split: Optional[str] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Read per-chemical report counts by 4-digit NAICS and by condition of use (read-only).

        Args:
            report_split (Optional[str]): Count only reports with this ``split``; ``None`` counts all.

        Returns:
            Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: ``(sector rows, condition-of-use rows)``, each
                with ``cas``, ``key`` and ``n``.
        """
        params = {"report_split": report_split}
        return self._graph.query(CONTEXT_SECTOR_COUNTS, params), self._graph.query(CONTEXT_COU_COUNTS, params)

    def code_profiles(self, report_split: Optional[str] = None) -> Tuple[Dict[str, int], int]:
        """Count, per activity code, the chemicals that report it (read-only).

        Args:
            report_split (Optional[str]): Count only reports with this ``split``; ``None`` counts all.

        Returns:
            Tuple[Dict[str, int], int]: ``({code: number of chemicals reporting it}, number of chemicals
                reporting any code)``.
        """
        params = {"report_split": report_split}
        per_code = {str(r["code"]): int(r["n_chemicals"]) for r in self._graph.query(CODE_CHEMICAL_COUNTS, params)}
        total = self._graph.query(CHEMICALS_WITH_CODES, params)
        return per_code, int(total[0]["n_chemicals"]) if total else 0

    def reduction_records(self, report_split: Optional[str] = None) -> List[Dict[str, Any]]:
        """Read every reduction record (report, chemical, code, uses, sector, year, state) under a split (read-only).

        Args:
            report_split (Optional[str]): Only reports with this ``split``; ``None`` reads all.

        Returns:
            List[Dict[str, Any]]: One row per (report, code) with the keys of ``REDUCTION_RECORDS``.
        """
        return [dict(r) for r in self._graph.query(REDUCTION_RECORDS, {"report_split": report_split})]
