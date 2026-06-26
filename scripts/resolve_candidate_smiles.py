# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Resolve SMILES for Candidate nodes via PubChem and store them on the graph.

The category-member chemicals loaded as `:Candidate` carry no SMILES (the source CSV has
none). This script resolves each candidate's CAS number to a canonical SMILES through the
PubChem PUG REST API, caches results to disk (resumable), and writes them onto the
`:Candidate` nodes. SMILES is the prerequisite for the three chemical embeddings that make
candidates searchable.

Safe to run alongside other work: it only reads/writes `:Candidate.canonical_smiles`.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-24
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

from src.neo4j_connector import Neo4jConnector

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_PATH = Path("ancillary/candidate_smiles_cache.json")
# PubChem renamed the SMILES properties: request the canonical (connectivity) form,
# falling back to the absolute SMILES.
_PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{cas}/property/ConnectivitySMILES,SMILES/JSON"
_RATE_DELAY = 0.2  # ≤5 requests/second per PubChem guidance
_TIMEOUT = 15
_STORE_BATCH = 500


def _load_cache() -> Dict[str, Optional[str]]:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def _save_cache(cache: Dict[str, Optional[str]]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=0))


def _pending_cas() -> List[str]:
    with Neo4jConnector().session() as session:
        rows = session.run("MATCH (c:Candidate) WHERE c.canonical_smiles IS NULL RETURN c.cas_number AS cas")
        return [r["cas"] for r in rows]


def _fetch_smiles(cas: str) -> Optional[str]:
    try:
        resp = requests.get(_PUG.format(cas=cas), timeout=_TIMEOUT)
        if resp.status_code != 200:
            return None
        props = resp.json().get("PropertyTable", {}).get("Properties", [])
        if not props:
            return None
        return props[0].get("ConnectivitySMILES") or props[0].get("SMILES")
    except (requests.RequestException, ValueError):
        return None


def _store(pairs: List[Dict[str, str]]) -> None:
    if not pairs:
        return
    query = "UNWIND $rows AS row " "MATCH (c:Candidate {cas_number: row.cas}) SET c.canonical_smiles = row.smiles"
    with Neo4jConnector().session() as session:
        session.run(query, {"rows": pairs})


def main() -> None:
    cache = _load_cache()
    pending = _pending_cas()
    logger.info(f"{len(pending)} candidates need SMILES; {len(cache)} already cached")

    resolved: List[Dict[str, str]] = []
    n_hit = n_miss = 0
    for i, cas in enumerate(pending, 1):
        if cas not in cache:
            cache[cas] = _fetch_smiles(cas)
            time.sleep(_RATE_DELAY)
        smiles = cache[cas]
        if smiles:
            resolved.append({"cas": cas, "smiles": smiles})
            n_hit += 1
        else:
            n_miss += 1
        if i % _STORE_BATCH == 0:
            _store(resolved)
            resolved = []
            _save_cache(cache)
            logger.info(f"  {i}/{len(pending)} processed ({n_hit} resolved, {n_miss} unresolved)")

    _store(resolved)
    _save_cache(cache)
    logger.info(f"Done: {n_hit} resolved, {n_miss} unresolved of {len(pending)}")


if __name__ == "__main__":
    main()
