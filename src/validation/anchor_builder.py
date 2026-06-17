# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Ground-truth anchor construction for embedding and retrieval evaluation.

Builds two independent anchor sets plus per-chemical activity profiles:
- Structural (Tanimoto): pairwise graded relevance from RDKit Morgan fingerprints,
  computed outside any embedding model so it can fairly judge all embedding types.
- Operational (Jaccard Activity Similarity + NAICS co-occurrence): graded relevance
  based on shared source reduction activity profiles from the index split,
  aligning evaluation with the system's actual objective of recommending activities.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set

import numpy as np
from rdkit import Chem
from rdkit.Chem import DataStructs, rdMolDescriptors

from src.config_manager import ConfigManager
from src.neo4j_connector import Neo4jConnector

logger = logging.getLogger(__name__)

# --- Structural anchor thresholds (pre-registered, do not change) ---
TANIMOTO_GRADE_HIGH: float = 0.8
TANIMOTO_GRADE_LOW: float = 0.4

# --- Operational anchor thresholds (pre-registered, do not change) ---
JACCARD_GRADE_HIGH: float = 0.7
JACCARD_GRADE_LOW: float = 0.3
NAICS_COOCCURRENCE_HIGH: int = 3

# --- Stratified random split parameters (pre-registered, do not change) ---
TEST_FRACTION: float = 0.20
RANDOM_SEED: int = 42

_FETCH_CHEMICALS = """
MATCH (c:Chemical)
WHERE c.canonical_smiles IS NOT NULL
RETURN c.cas_number AS cas_number, c.canonical_smiles AS smiles
"""

_FETCH_ACTIVITY_CODES_INDEX = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE r.split = 'index'
RETURN c.cas_number AS cas_number, collect(DISTINCT red.code) AS activity_codes
"""

_FETCH_NAICS_ACTIVITY_PAIRS_INDEX = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.split = 'index'
RETURN c.cas_number AS cas_number,
       collect(DISTINCT s.naics_code + '|' + red.code) AS naics_activity_pairs
"""

# --- Activity profile queries (for ActivityCoverage evaluation) ---

_FETCH_INDEX_SECTOR_CODES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.split = 'index'
RETURN c.cas_number AS cas_number,
       s.naics_code AS naics_code,
       collect(DISTINCT red.code) AS codes
"""

_FETCH_TEST_SECTOR_CODES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
MATCH (r)-[:FILED_BY]->(f:Facility)-[:CLASSIFIED_AS]->(s:Sector)
WHERE r.split = 'test'
RETURN c.cas_number AS cas_number,
       s.naics_code AS naics_code,
       collect(DISTINCT red.code) AS codes
"""

# --- Stratified split queries ---

_COUNT_SPLIT_SET = "MATCH (r:Report) WHERE r.split IS NOT NULL RETURN count(r) AS n"

_FETCH_ALL_REPORT_IDS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)
RETURN id(r) AS report_id, c.cas_number AS cas_number
"""

_SET_SPLIT_BATCH = """
UNWIND $rows AS row
MATCH (r:Report) WHERE id(r) = row.report_id
SET r.split = row.split_value
"""


@dataclass
class StructuralAnchors:
    """Pairwise Tanimoto-derived graded relevance for all chemicals with SMILES.

    Attributes:
        cas_numbers (List[str]): Ordered list of CAS numbers.
        similarity_matrix (np.ndarray): Float32 (N, N) Tanimoto scores.
        grade_matrix (np.ndarray): Int8 (N, N) grades: 0, 1, or 2.
        index (Dict[str, int]): Maps CAS number → matrix row/column index.
    """

    cas_numbers: List[str]
    similarity_matrix: np.ndarray
    grade_matrix: np.ndarray
    index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.index = {cas: i for i, cas in enumerate(self.cas_numbers)}

    def grades_for(self, cas_number: str) -> Dict[str, int]:
        """Return graded relevance scores for all other chemicals.

        Args:
            cas_number (str): Query CAS number (must be in the anchor set).

        Returns:
            Dict[str, int]: Maps each other CAS number → grade (0, 1, or 2).
        """
        idx = self.index[cas_number]
        row = self.grade_matrix[idx]
        return {cas: int(row[j]) for j, cas in enumerate(self.cas_numbers) if cas != cas_number}


@dataclass
class ActivityProfiles:
    """Per-chemical activity profiles for ActivityCoverage@k evaluation.

    Stores global (all-sector) and per-sector activity code sets for the full
    index (used to aggregate retrieved chemicals' activities) and the test split
    (used as query ground truth, i.e. the target activities we want to recover).

    Attributes:
        index_global (Dict[str, FrozenSet[str]]): CAS → activity codes from full index.
        query_global (Dict[str, FrozenSet[str]]): CAS → activity codes from test split.
        index_sector (Dict[str, Dict[str, FrozenSet[str]]]): CAS → 6-digit NAICS → codes (full).
        query_sector (Dict[str, Dict[str, FrozenSet[str]]]): CAS → 6-digit NAICS → codes (test).
        code_to_family (Dict[str, str]): Activity code → family ID (F1–F5; F0 = other).
    """

    index_global: Dict[str, FrozenSet[str]]
    query_global: Dict[str, FrozenSet[str]]
    index_sector: Dict[str, Dict[str, FrozenSet[str]]]
    query_sector: Dict[str, Dict[str, FrozenSet[str]]]
    code_to_family: Dict[str, str]

    def families(self, codes: FrozenSet[str]) -> FrozenSet[str]:
        """Map a set of activity codes to their family IDs.

        Args:
            codes (FrozenSet[str]): Activity code set.

        Returns:
            FrozenSet[str]: Set of family IDs (F1–F5; F0 for unmapped codes).
        """
        return frozenset(self.code_to_family.get(c, "F0") for c in codes)

    def get_index_sector_codes(self, cas: str, naics_prefix: str) -> FrozenSet[str]:
        """Return full-index activity codes for a chemical within a NAICS prefix.

        Args:
            cas (str): CAS number.
            naics_prefix (str): Truncated NAICS code (2–6 digits).

        Returns:
            FrozenSet[str]: Union of codes for all matching (CAS, NAICS) pairs.
        """
        result: Set[str] = set()
        for naics_6, codes in self.index_sector.get(cas, {}).items():
            if naics_6.startswith(naics_prefix):
                result |= codes
        return frozenset(result)

    def get_query_sector_codes(self, cas: str, naics_prefix: str) -> FrozenSet[str]:
        """Return test-split activity codes for a chemical within a NAICS prefix.

        Args:
            cas (str): CAS number.
            naics_prefix (str): Truncated NAICS code (2–6 digits).

        Returns:
            FrozenSet[str]: Union of codes for all matching (CAS, NAICS) pairs.
        """
        result: Set[str] = set()
        for naics_6, codes in self.query_sector.get(cas, {}).items():
            if naics_6.startswith(naics_prefix):
                result |= codes
        return frozenset(result)

    def get_query_naics_prefixes(self, cas: str, level: int) -> FrozenSet[str]:
        """Return unique NAICS prefixes at a given hierarchy level for a test chemical.

        Args:
            cas (str): CAS number.
            level (int): Number of NAICS digits (2, 3, 4, 5, or 6).

        Returns:
            FrozenSet[str]: Unique NAICS prefixes for this chemical in the test split.
        """
        return frozenset(naics_6[:level] for naics_6 in self.query_sector.get(cas, {}).keys())


@dataclass
class OperationalAnchors:
    """Activity-profile graded relevance from development-split TRI reduction records.

    Grades chemical pairs by overlap of source reduction activities (Jaccard Activity
    Similarity) and shared (NAICS × activity code) combinations, which directly reflects
    the system's objective of recommending source reduction interventions.

    Grade rules (pre-registered):
        2 — Highly Relevant: Jaccard ≥ 0.7 AND NAICS co-occurrence ≥ 3
        1 — Partially Relevant: Jaccard ≥ 0.3 OR NAICS co-occurrence ≥ 1
        0 — Not Relevant: Jaccard < 0.3 AND NAICS co-occurrence = 0

    Attributes:
        cas_numbers (List[str]): Ordered list of CAS numbers in the anchor.
        jaccard_matrix (np.ndarray): Float32 (N, N) pairwise Jaccard Activity Similarity (code-level).
        jaccard_family_matrix (np.ndarray): Float32 (N, N) pairwise Jaccard at activity-family level.
        naics_cooccurrence_matrix (np.ndarray): Int32 (N, N) shared (NAICS × code) counts.
        index (Dict[str, int]): Maps CAS number → matrix row/column index.
        naics_digits (int): NAICS hierarchy level used (4 = Industry Group, 6 = National).
    """

    cas_numbers: List[str]
    jaccard_matrix: np.ndarray
    jaccard_family_matrix: np.ndarray
    naics_cooccurrence_matrix: np.ndarray
    index: Dict[str, int] = field(default_factory=dict)
    naics_digits: int = 6

    def __post_init__(self) -> None:
        self.index = {cas: i for i, cas in enumerate(self.cas_numbers)}

    def grades_for(self, cas_number: str) -> Dict[str, int]:
        """Return operational graded relevance scores for all other chemicals.

        Args:
            cas_number (str): Query CAS number (must be in the anchor set).

        Returns:
            Dict[str, int]: Maps each other CAS number → grade (0, 1, or 2).
        """
        if cas_number not in self.index:
            return {}
        idx = self.index[cas_number]
        jac_row = self.jaccard_matrix[idx]
        naics_row = self.naics_cooccurrence_matrix[idx]
        result: Dict[str, int] = {}
        for j, cas in enumerate(self.cas_numbers):
            if cas == cas_number:
                continue
            a = float(jac_row[j])
            n = int(naics_row[j])
            if a >= JACCARD_GRADE_HIGH and n >= NAICS_COOCCURRENCE_HIGH:
                result[cas] = 2
            elif a >= JACCARD_GRADE_LOW or n >= 1:
                result[cas] = 1
            else:
                result[cas] = 0
        return result


class AnchorBuilder:
    """Builds structural and operational ground-truth anchors plus activity profiles."""

    def __init__(
        self,
        morgan_radius: int = 2,
        morgan_bits: int = 2048,
    ) -> None:
        self._radius = morgan_radius
        self._n_bits = morgan_bits
        self._connector = Neo4jConnector()

    def write_stratified_split(
        self,
        test_fraction: float = TEST_FRACTION,
        seed: int = RANDOM_SEED,
        force: bool = False,
    ) -> None:
        """Assign r.split = 'index' or 'test' on every Report node via stratified random split.

        Stratified by cas_number so each chemical's reports are split proportionally.
        Idempotent: skips if split is already set unless force=True.

        Edge cases:
            - 1 report → assigned to index; chemical excluded from evaluation queries.
            - 2–N reports → ceil(N * test_fraction) assigned to test, rest to index.

        Args:
            test_fraction (float): Fraction of each chemical's reports to hold out as test.
            seed (int): Random seed for reproducibility.
            force (bool): Re-assign split even if already set on Report nodes.
        """
        with self._connector.session() as session:
            n_set = session.run(_COUNT_SPLIT_SET).single()["n"]  # type: ignore[index]
        if n_set > 0 and not force:
            logger.info(f"Split already set on {n_set:,} Report nodes — skipping (pass force=True to redo)")
            return

        with self._connector.session() as session:
            rows = [dict(r) for r in session.run(_FETCH_ALL_REPORT_IDS)]

        by_cas: Dict[str, List[int]] = {}
        for row in rows:
            by_cas.setdefault(row["cas_number"], []).append(int(row["report_id"]))

        rng = random.Random(seed)  # nosec B311 - reproducible data split, not security-sensitive
        assignments: List[Dict] = []
        for report_ids in by_cas.values():
            n = len(report_ids)
            shuffled = list(report_ids)
            rng.shuffle(shuffled)
            n_test = 0 if n == 1 else max(1, math.ceil(n * test_fraction))
            for i, rid in enumerate(shuffled):
                assignments.append({"report_id": rid, "split_value": "test" if i < n_test else "index"})

        batch_size = 500
        with self._connector.session() as session:
            for i in range(0, len(assignments), batch_size):
                session.run(_SET_SPLIT_BATCH, {"rows": assignments[i : i + batch_size]})  # type: ignore[arg-type]

        n_test_total = sum(1 for a in assignments if a["split_value"] == "test")
        logger.info(
            f"Wrote stratified split: {len(assignments) - n_test_total:,} index, "
            f"{n_test_total:,} test ({len(by_cas):,} chemicals, seed={seed})"
        )

    def _build_family_map(self) -> Dict[str, str]:
        """Build activity code → family ID mapping from config TRI_SOURCE_REDUCTION_GROUPS.

        Returns:
            Dict[str, str]: Maps each S-code to its family ID (F1–F5).
        """
        groups = ConfigManager().config.get("TRI_SOURCE_REDUCTION_GROUPS", [])
        return {code: f"F{i}" for i, grp in enumerate(groups, 1) for code in grp.get("CODES", [])}

    def _fetch_chemicals(self) -> List[Dict[str, str]]:
        with self._connector.session() as session:
            return [dict(r) for r in session.run(_FETCH_CHEMICALS)]

    def _to_fp(self, smiles: str) -> Optional[DataStructs.ExplicitBitVect]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, self._radius, nBits=self._n_bits)

    def _grade(self, tanimoto: float) -> int:
        if tanimoto >= TANIMOTO_GRADE_HIGH:
            return 2
        if tanimoto >= TANIMOTO_GRADE_LOW:
            return 1
        return 0

    def build_structural_anchors(self) -> StructuralAnchors:
        """Compute pairwise Tanimoto similarity and derive graded relevance.

        Fetches chemicals with valid SMILES from Neo4j, computes Morgan fingerprints
        independently of any stored embedding, and grades each pair 0 (< 0.4),
        1 (0.4–0.8), or 2 (≥ 0.8).

        Returns:
            StructuralAnchors: Pairwise graded relevance for all valid chemicals.
        """
        chemicals = self._fetch_chemicals()
        fps_map = {c["cas_number"]: fp for c in chemicals if (fp := self._to_fp(c["smiles"])) is not None}
        cas_numbers = list(fps_map.keys())
        fps = [fps_map[cas] for cas in cas_numbers]
        n = len(cas_numbers)

        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i, fp_i in enumerate(fps):
            sim_matrix[i] = DataStructs.BulkTanimotoSimilarity(fp_i, fps)

        grade_matrix = np.vectorize(self._grade)(sim_matrix).astype(np.int8)
        logger.info(f"Built structural anchors for {n} chemicals")
        return StructuralAnchors(cas_numbers=cas_numbers, similarity_matrix=sim_matrix, grade_matrix=grade_matrix)

    def build_operational_anchors(self, naics_digits: int = 6) -> OperationalAnchors:
        """Build operational anchors from development-split (2014–2020) activity profiles.

        Computes pairwise Jaccard Activity Similarity from unique TRI source reduction
        activity codes per chemical, and pairwise NAICS co-occurrence counts from
        shared (NAICS × activity code) combinations.  Uses the development split only
        so the anchor is independent of test-split observations.

        Args:
            naics_digits (int): NAICS hierarchy depth.  6 = National Industry (default),
                4 = Industry Group (coarser, more shared codes, larger N values).

        Returns:
            OperationalAnchors: Pairwise Jaccard and NAICS co-occurrence matrices with
                pre-registered 0/1/2 graded relevance via grades_for().
        """
        with self._connector.session() as session:
            act_records = [dict(r) for r in session.run(_FETCH_ACTIVITY_CODES_INDEX)]
            naics_records = [dict(r) for r in session.run(_FETCH_NAICS_ACTIVITY_PAIRS_INDEX)]

        act_sets: Dict[str, Set[str]] = {
            r["cas_number"]: set(r["activity_codes"]) for r in act_records if r.get("activity_codes")
        }

        code_to_family = self._build_family_map()
        family_sets: Dict[str, FrozenSet[str]] = {
            cas: frozenset(code_to_family.get(c, "F0") for c in codes) for cas, codes in act_sets.items()
        }

        naics_act_sets: Dict[str, Set[str]] = {}
        for r in naics_records:
            cas = r["cas_number"]
            pairs: Set[str] = set()
            for pair_str in r.get("naics_activity_pairs") or []:
                parts = pair_str.split("|", 1)
                if len(parts) == 2:
                    naics_code, act_code = parts
                    pairs.add(f"{naics_code[:naics_digits]}|{act_code}")
            naics_act_sets[cas] = pairs

        cas_numbers = sorted(set(act_sets.keys()) & set(naics_act_sets.keys()))
        n = len(cas_numbers)

        jaccard_matrix = np.zeros((n, n), dtype=np.float32)
        jaccard_family_matrix = np.zeros((n, n), dtype=np.float32)
        naics_cooccurrence_matrix = np.zeros((n, n), dtype=np.int32)

        for i in range(n):
            act_i = act_sets[cas_numbers[i]]
            fam_i = family_sets[cas_numbers[i]]
            naics_i = naics_act_sets[cas_numbers[i]]
            jaccard_matrix[i, i] = 1.0
            jaccard_family_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                act_j = act_sets[cas_numbers[j]]
                fam_j = family_sets[cas_numbers[j]]
                naics_j = naics_act_sets[cas_numbers[j]]
                union_act = act_i | act_j
                jac = float(len(act_i & act_j) / len(union_act)) if union_act else 0.0
                union_fam = fam_i | fam_j
                jac_fam = float(len(fam_i & fam_j) / len(union_fam)) if union_fam else 0.0
                naics_cooc = len(naics_i & naics_j)
                jaccard_matrix[i, j] = jac
                jaccard_matrix[j, i] = jac
                jaccard_family_matrix[i, j] = jac_fam
                jaccard_family_matrix[j, i] = jac_fam
                naics_cooccurrence_matrix[i, j] = naics_cooc
                naics_cooccurrence_matrix[j, i] = naics_cooc

        logger.info(f"Built operational anchors for {n} chemicals ({naics_digits}-digit NAICS, index split)")
        return OperationalAnchors(
            cas_numbers=cas_numbers,
            jaccard_matrix=jaccard_matrix,
            jaccard_family_matrix=jaccard_family_matrix,
            naics_cooccurrence_matrix=naics_cooccurrence_matrix,
            naics_digits=naics_digits,
        )

    def build_activity_profiles(self) -> ActivityProfiles:
        """Build per-chemical activity profiles for ActivityCoverage@k evaluation.

        Fetches activity codes from Neo4j for both the full index (all years,
        used for aggregating retrieved chemicals' activities) and the test split
        (2021–2024, used as query ground truth).  Profiles are sector-conditioned
        by 6-digit NAICS code; higher levels are extracted by prefix truncation.

        Returns:
            ActivityProfiles: Global and sector-conditioned code sets for queries
                and index chemicals, plus the activity-family mapping.
        """
        code_to_family = self._build_family_map()

        with self._connector.session() as session:
            all_rows = [dict(r) for r in session.run(_FETCH_INDEX_SECTOR_CODES)]
            test_rows = [dict(r) for r in session.run(_FETCH_TEST_SECTOR_CODES)]

        def _build_sector_dict(rows: List[Dict]) -> Dict[str, Dict[str, FrozenSet[str]]]:
            result: Dict[str, Dict[str, FrozenSet[str]]] = {}
            for row in rows:
                cas = row["cas_number"]
                raw_naics = row["naics_code"]
                # Neo4j stores NAICS as float (e.g. 332420.0); normalise to 6-char string
                naics = str(int(float(raw_naics))) if raw_naics is not None else ""
                if not naics:
                    continue
                codes = frozenset(row.get("codes") or [])
                if not codes:
                    continue
                result.setdefault(cas, {})[naics] = codes
            return result

        index_sector = _build_sector_dict(all_rows)
        query_sector = _build_sector_dict(test_rows)

        index_global: Dict[str, FrozenSet[str]] = {}
        for cas, naics_map in index_sector.items():
            merged: Set[str] = set()
            for codes in naics_map.values():
                merged |= codes
            index_global[cas] = frozenset(merged)

        query_global: Dict[str, FrozenSet[str]] = {}
        for cas, naics_map in query_sector.items():
            merged_q: Set[str] = set()
            for codes in naics_map.values():
                merged_q |= codes
            query_global[cas] = frozenset(merged_q)

        n_index = len(index_global)
        n_query = len(query_global)
        logger.info(f"Built activity profiles: {n_index} index chemicals, " f"{n_query} test chemicals with activities")
        return ActivityProfiles(
            index_global=index_global,
            query_global=query_global,
            index_sector=index_sector,
            query_sector=query_sector,
            code_to_family=code_to_family,
        )
