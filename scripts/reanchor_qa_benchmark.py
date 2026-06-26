# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Re-anchor the QA benchmark so every question targets a chemical with a SMILES.

The retriever uses a mandatory chemical (SMILES) anchor, with sector and condition-of-use
as optional *additional* context. The frozen benchmark contained questions that a
SMILES-anchored retriever cannot serve:

1. Two reverse-lookup types with no chemical anchor at all
   (``sector_chemicals`` / ``reduction_code_chemicals`` — "which chemicals had …?").
2. Questions across all types anchored on TRI *category* entries (e.g. "Nickel
   compounds") that have no ``canonical_smiles``.

This migration rewrites both classes into the chemical-anchored shape, grounded in graph
facts and constrained to chemicals that have a SMILES:

- ``sector_chemicals``         → ``sector_chemical_reduction`` (chemical + sector context)
- ``reduction_code_chemicals`` → ``chemical_reduction``        (chemical, year context)
- ``chemical_sector_reduction``/``facility_count`` with a SMILES-less anchor are
  re-anchored onto a SMILES-bearing chemical in the same sector/state and year.

Questions already anchored on a SMILES-bearing chemical are left unchanged.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-24
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.neo4j_connector import Neo4jConnector

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

BENCHMARK_PATH = Path("data/qa_benchmark.json")

_HAS_SMILES = "MATCH (c:Chemical {cas_number: $cas}) RETURN c.canonical_smiles AS s"

_SECTOR_CHEM = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(:Facility)-[:CLASSIFIED_AS]->(sec:Sector)
MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE c.canonical_smiles IS NOT NULL AND sec.naics_code = $naics AND r.reporting_year = $year
WITH c, sec, collect(DISTINCT red.name) AS reductions
WHERE size(reductions) > 0
RETURN c.name AS chem, c.cas_number AS cas, sec.naics_title AS sector_title, reductions
ORDER BY c.name LIMIT 1
"""

# Fallback: same sector, any test-split year (2021–2024) — for sectors whose requested
# year holds only SMILES-less category entries. Returns the actual year used.
_SECTOR_CHEM_ANY_YEAR = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(:Facility)-[:CLASSIFIED_AS]->(sec:Sector)
MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE c.canonical_smiles IS NOT NULL AND sec.naics_code = $naics
  AND r.reporting_year >= 2021 AND r.reporting_year <= 2024
WITH c, sec, r.reporting_year AS year, collect(DISTINCT red.name) AS reductions
WHERE size(reductions) > 0
RETURN c.name AS chem, c.cas_number AS cas, sec.naics_title AS sector_title, year, reductions
ORDER BY c.name, year LIMIT 1
"""

# Last-resort fallback: a sector whose every year holds only SMILES-less category
# entries cannot anchor a chemical, so draw a fresh chemical+sector+year fact that does.
_ANY_SECTOR_CHEM = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(:Facility)-[:CLASSIFIED_AS]->(sec:Sector)
MATCH (r)-[:DOCUMENTS]->(red:Reduction)
WHERE c.canonical_smiles IS NOT NULL AND r.reporting_year >= 2021 AND r.reporting_year <= 2024
WITH c, sec, r.reporting_year AS year, collect(DISTINCT red.name) AS reductions
WHERE size(reductions) > 0
RETURN c.name AS chem, c.cas_number AS cas, sec.naics_title AS sector_title,
       sec.naics_code AS naics, year, reductions
ORDER BY c.name, sec.naics_code, year LIMIT 1
"""

_CHEM_REDUCTIONS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(:Reduction {code: $code})
WHERE c.canonical_smiles IS NOT NULL AND r.reporting_year = $year
WITH c ORDER BY c.name LIMIT 1
MATCH (c)<-[:REPORTS_ON]-(r2:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE r2.reporting_year = $year
RETURN c.name AS chem, c.cas_number AS cas, collect(DISTINCT red.name) AS reductions
"""

_FACILITY_COUNT = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:FILED_BY]->(f:Facility)
MATCH (r)-[:DOCUMENTS]->(:Reduction)
WHERE c.canonical_smiles IS NOT NULL AND f.state = $state AND r.reporting_year = $year
WITH c, count(DISTINCT f.trifid) AS n
WHERE n >= 2
RETURN c.name AS chem, c.cas_number AS cas, n AS n_facilities
ORDER BY c.name LIMIT 1
"""


def _has_smiles(session: Any, cas: Optional[str]) -> bool:
    if not cas:
        return False
    row = session.run(_HAS_SMILES, {"cas": cas}).single()
    return bool(row and row["s"])


def _sector_pair(session: Any, naics: float, year: int, qtype: str) -> Optional[Dict[str, Any]]:
    row = session.run(_SECTOR_CHEM, {"naics": naics, "year": year}).single()
    used_naics, used_year = naics, year
    if not row or not row["reductions"]:
        row = session.run(_SECTOR_CHEM_ANY_YEAR, {"naics": naics}).single()
        if row and row["reductions"]:
            used_year = row["year"]
        else:  # sector has no SMILES-bearing chemical in any test year — draw a fresh fact
            row = session.run(_ANY_SECTOR_CHEM).single()
            if not row or not row["reductions"]:
                return None
            used_naics, used_year = row["naics"], row["year"]
    return {
        "question": (
            f"What source reduction activities were applied to {row['chem']} "
            f"in the {row['sector_title']} sector in {used_year}?"
        ),
        "ground_truth": ", ".join(sorted(row["reductions"])),
        "question_type": qtype,
        "metadata": {"cas_number": row["cas"], "naics_code": used_naics, "year": used_year},
    }


def _reduction_pair(session: Any, code: str, year: int) -> Optional[Dict[str, Any]]:
    row = session.run(_CHEM_REDUCTIONS, {"code": code, "year": year}).single()
    if not row or not row["reductions"]:
        return None
    return {
        "question": f"What source reduction activities were applied to {row['chem']} in {year}?",
        "ground_truth": ", ".join(sorted(row["reductions"])),
        "question_type": "chemical_reduction",
        "metadata": {"cas_number": row["cas"], "year": year},
    }


def _facility_pair(session: Any, state: str, year: int) -> Optional[Dict[str, Any]]:
    row = session.run(_FACILITY_COUNT, {"state": state, "year": year}).single()
    if not row:
        return None
    return {
        "question": f"How many facilities in {state} reported source reductions for {row['chem']} in {year}?",
        "ground_truth": str(row["n_facilities"]),
        "question_type": "facility_count",
        "metadata": {"cas_number": row["cas"], "state": state, "year": year},
    }


def _transform(session: Any, pair: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qtype, meta = pair["question_type"], pair["metadata"]
    if qtype == "sector_chemicals":
        return _sector_pair(session, meta["naics_code"], meta["year"], "sector_chemical_reduction")
    if qtype == "reduction_code_chemicals":
        return _reduction_pair(session, meta["reduction_code"], meta["year"])
    # Conforming types: only re-anchor when the existing anchor lacks a SMILES; keep the type.
    if _has_smiles(session, meta.get("cas_number")):
        return None
    if qtype == "chemical_sector_reduction":
        return _sector_pair(session, meta["naics_code"], meta["year"], "chemical_sector_reduction")
    if qtype == "facility_count":
        return _facility_pair(session, meta["state"], meta["year"])
    return None


def main() -> None:
    pairs: List[Dict[str, Any]] = json.loads(BENCHMARK_PATH.read_text())
    conn = Neo4jConnector()
    rewritten = 0
    with conn.session() as session:
        for i, pair in enumerate(pairs):
            new = _transform(session, pair)
            if new:
                pairs[i] = new
                rewritten += 1
            elif pair["question_type"] in ("sector_chemicals", "reduction_code_chemicals"):
                logger.warning(f"Could not re-anchor {pair['question_type']} {pair['metadata']}; left as-is")

    with_cas = sum(1 for p in pairs if "cas_number" in p.get("metadata", {}))
    logger.info(f"Rewrote {rewritten} questions; {with_cas}/{len(pairs)} chemical-anchored")
    BENCHMARK_PATH.write_text(json.dumps(pairs, indent=2, ensure_ascii=False))
    logger.info(f"Saved re-anchored benchmark to {BENCHMARK_PATH}")


if __name__ == "__main__":
    main()
