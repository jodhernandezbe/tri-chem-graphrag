# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Synthetic QA pair generation from the TRI knowledge graph for RAG evaluation.

Generates deterministic question-answer pairs from known graph facts in the test
split (2021-2024). QA pairs are generated once and frozen as a JSON benchmark
before any system tuning begins.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)

TEST_SPLIT_START: int = 2021
TEST_SPLIT_END: int = 2024

_CHEMICAL_SECTOR_REDUCTIONS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE r.reporting_year >= $start AND r.reporting_year <= $end
WITH c, s, r.reporting_year AS year, collect(DISTINCT red.name) AS reductions
WHERE size(reductions) > 0
RETURN c.name AS chemical, c.cas_number AS cas_number,
       s.naics_title AS sector_title, s.naics_code AS naics_code,
       year, reductions
ORDER BY rand()
LIMIT $limit
"""

_FACILITY_COUNT_BY_CHEMICAL_STATE = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)
MATCH (r)-[:DOCUMENTS]->(:Reduction)
WHERE r.reporting_year >= $start AND r.reporting_year <= $end
  AND f.state IS NOT NULL
WITH c, f.state AS state, r.reporting_year AS year, count(DISTINCT f.trifid) AS n
WHERE n >= 2
RETURN c.name AS chemical, c.cas_number AS cas_number,
       state, year, n AS n_facilities
ORDER BY rand()
LIMIT $limit
"""

_SECTOR_CHEMICALS_BY_YEAR = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
MATCH (r)-[:DOCUMENTS]->(:Reduction)
WHERE r.reporting_year >= $start AND r.reporting_year <= $end
WITH s, r.reporting_year AS year, collect(DISTINCT c.name) AS chemicals
WHERE size(chemicals) >= 2
RETURN s.naics_title AS sector_title, s.naics_code AS naics_code,
       year, chemicals
ORDER BY rand()
LIMIT $limit
"""

_REDUCTION_CODE_CHEMICALS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE r.reporting_year >= $start AND r.reporting_year <= $end
WITH red.code AS code, red.name AS name, r.reporting_year AS year,
     collect(DISTINCT c.name) AS chemicals
WHERE size(chemicals) >= 2
RETURN code AS reduction_code, name AS reduction_name, year, chemicals
ORDER BY rand()
LIMIT $limit
"""


class QAGenerator:
    """Generates synthetic QA pairs from TRI knowledge graph facts."""

    def __init__(
        self,
        test_start: int = TEST_SPLIT_START,
        test_end: int = TEST_SPLIT_END,
        seed: int = 42,
    ) -> None:
        self._start = test_start
        self._end = test_end
        self._connector = Neo4jConnector()
        random.seed(seed)

    def _run(self, query: str, limit: int) -> List[Dict[str, Any]]:
        params: Dict[str, Any] = {"start": self._start, "end": self._end, "limit": limit}
        with self._connector.session() as session:
            return [dict(r) for r in session.run(query, params)]  # type: ignore[arg-type]

    def _chemical_sector_pairs(self, n: int) -> List[Dict[str, Any]]:
        return [
            {
                "question": (
                    f"What source reduction activities were applied to {r['chemical']} "
                    f"in the {r['sector_title']} sector in {r['year']}?"
                ),
                "ground_truth": ", ".join(sorted(r["reductions"])),
                "question_type": "chemical_sector_reduction",
                "metadata": {"cas_number": r["cas_number"], "naics_code": r["naics_code"], "year": r["year"]},
            }
            for r in self._run(_CHEMICAL_SECTOR_REDUCTIONS, n)
        ]

    def _facility_count_pairs(self, n: int) -> List[Dict[str, Any]]:
        return [
            {
                "question": (
                    f"How many facilities in {r['state']} reported source reductions " f"for {r['chemical']} in {r['year']}?"
                ),
                "ground_truth": str(r["n_facilities"]),
                "question_type": "facility_count",
                "metadata": {"cas_number": r["cas_number"], "state": r["state"], "year": r["year"]},
            }
            for r in self._run(_FACILITY_COUNT_BY_CHEMICAL_STATE, n)
        ]

    def _sector_chemical_pairs(self, n: int) -> List[Dict[str, Any]]:
        return [
            {
                "question": (
                    f"Which chemicals had source reduction activities in the " f"{r['sector_title']} sector in {r['year']}?"
                ),
                "ground_truth": ", ".join(sorted(r["chemicals"])),
                "question_type": "sector_chemicals",
                "metadata": {"naics_code": r["naics_code"], "year": r["year"]},
            }
            for r in self._run(_SECTOR_CHEMICALS_BY_YEAR, n)
        ]

    def _reduction_code_pairs(self, n: int) -> List[Dict[str, Any]]:
        return [
            {
                "question": (
                    f"Which chemicals were reported with source reduction code "
                    f"{r['reduction_code']} ({r['reduction_name']}) in {r['year']}?"
                ),
                "ground_truth": ", ".join(sorted(r["chemicals"])),
                "question_type": "reduction_code_chemicals",
                "metadata": {"reduction_code": r["reduction_code"], "year": r["year"]},
            }
            for r in self._run(_REDUCTION_CODE_CHEMICALS, n)
        ]

    def generate(
        self,
        n_per_type: int = 25,
        output_path: Optional[Path] = None,
    ) -> List[Dict[str, Any]]:
        """Generate and optionally freeze a synthetic QA benchmark from KG facts.

        Produces four question types: chemical-sector-reduction, facility count,
        sector chemicals, and reduction-code chemicals. Call this once before any
        system tuning begins, passing output_path to persist the frozen benchmark.

        Args:
            n_per_type (int): Number of QA pairs to generate per question type.
            output_path (Optional[Path]): Path to write the frozen JSON benchmark.
                Parent directories are created if they do not exist.

        Returns:
            List[Dict[str, Any]]: Shuffled QA pairs, each with question,
                ground_truth, question_type, and metadata fields.
        """
        pairs = (
            self._chemical_sector_pairs(n_per_type)
            + self._facility_count_pairs(n_per_type)
            + self._sector_chemical_pairs(n_per_type)
            + self._reduction_code_pairs(n_per_type)
        )
        random.shuffle(pairs)
        logger.info(f"Generated {len(pairs)} QA pairs across 4 question types")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
            logger.info(f"Frozen QA benchmark saved to {output_path}")

        return pairs
