# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""End-to-end evaluation of the shipped source-reduction recommender.

Unlike the intrinsic metrics (which rank by pure vector similarity), this module
scores the *production* retrieval path — ``RetrieverFactory.retrieve_multi`` plus
graph traversal — exactly as it runs at inference. For each held-out chemical it
collects the ranked source-reduction activity codes the system would recommend and
measures, with the IDF-weighted coverage and code-precision metrics, how well the
top-k recommendations match the chemical's actual test-split activities.

Author: Jose D. Hernandez-Betancur
Date: 2026-06-13
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Set

import numpy as np

from src.neo4j_connector import Neo4jConnector
from src.rag_graph.generation.rag_pipeline import tabulate_records
from src.rag_graph.retrieval import RetrieverFactory
from src.validation.anchor_builder import ActivityProfiles
from src.validation.retrieval_metrics import (
    K_VALUES,
    _code_precision,
    _coverage,
    _idf_weights,
    _weighted_coverage,
)

logger = logging.getLogger(__name__)

# Number of unique recommended activity codes to keep before scoring at each k cutoff.
_MAX_RECORDS: int = 200

_FETCH_TEST_CHEMICALS_WITH_SMILES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)
WHERE r.split = 'test' AND c.canonical_smiles IS NOT NULL
WITH DISTINCT c
RETURN c.cas_number AS cas_number, c.canonical_smiles AS smiles
"""


@dataclass
class RecommendationCoverageResult:
    """Top-k recommendation quality for the shipped recommender or a baseline.

    The cutoff k is the number of distinct top-ranked recommended activity codes.

    Attributes:
        method (str): 'ShippedRecommender', 'Random', or 'Oracle'.
        weighted_coverage_at_k (Dict[int, float]): IDF-weighted coverage of the
            query's activities by the top-k recommended codes (primary).
        code_precision_at_k (Dict[int, float]): Fraction of the top-k recommended
            codes that are actually relevant (primary).
        raw_coverage_at_k (Dict[int, float]): Unweighted coverage (diagnostic).
        n_queries (int): Number of test chemicals evaluated.
    """

    method: str
    weighted_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    code_precision_at_k: Dict[int, float] = field(default_factory=dict)
    raw_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    n_queries: int = 0


class _RecAccumulator:
    """Accumulates per-query top-k recommendation metrics into a result record."""

    def __init__(self) -> None:
        self._weighted: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        self._precision: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        self._raw: Dict[int, List[float]] = {k: [] for k in K_VALUES}

    def add(
        self,
        ranked_codes: List[str],
        query_codes: FrozenSet[str],
        weights: Dict[str, float],
        fallback: float,
    ) -> None:
        """Record metrics for one query across all k cutoffs from its ranked codes."""
        for k in K_VALUES:
            retrieved = frozenset(ranked_codes[:k])
            self._weighted[k].append(_weighted_coverage(query_codes, retrieved, weights, fallback))
            self._precision[k].append(_code_precision(query_codes, retrieved))
            self._raw[k].append(_coverage(query_codes, retrieved))

    def to_result(self, method: str, n_queries: int) -> RecommendationCoverageResult:
        """Aggregate accumulated observations into a single result record."""

        def _mean(values: Dict[int, List[float]]) -> Dict[int, float]:
            return {k: float(np.mean(v)) if v else 0.0 for k, v in values.items()}

        return RecommendationCoverageResult(
            method=method,
            weighted_coverage_at_k=_mean(self._weighted),
            code_precision_at_k=_mean(self._precision),
            raw_coverage_at_k=_mean(self._raw),
            n_queries=n_queries,
        )


class RecommendationMetrics:
    """Scores the production retrieve_multi recommendation path against ground truth."""

    def __init__(self) -> None:
        self._connector = Neo4jConnector()
        self._factory = RetrieverFactory()

    def _fetch_test_chemicals(self) -> List[Dict[str, str]]:
        with self._connector.session() as session:
            return [dict(r) for r in session.run(_FETCH_TEST_CHEMICALS_WITH_SMILES)]

    def _recommended_codes(self, smiles: str) -> List[str]:
        """Return distinct recommended activity codes for a SMILES, ranked by similarity.

        Runs the exact production path (vector + graph traversal + tabulation) so the
        evaluation reflects what the deployed recommender returns.

        Args:
            smiles (str): Query chemical SMILES.

        Returns:
            List[str]: Distinct recommended source-reduction codes, highest similarity first.
        """
        nodes, subgraphs = self._factory.retrieve_multi(chemical_smiles=smiles, sector_query="")
        records = tabulate_records(nodes, subgraphs, limit=_MAX_RECORDS)
        codes: List[str] = []
        seen: Set[str] = set()
        for rec in records:
            code = rec.get("ReductionCode")
            if code and code not in seen:
                seen.add(code)
                codes.append(code)
        return codes

    def _random_baseline(
        self,
        query_chemicals: Dict[str, FrozenSet[str]],
        vocab: List[str],
        weights: Dict[str, float],
        fallback: float,
        n_samples: int = 50,
    ) -> RecommendationCoverageResult:
        """Expected top-k metrics when the k codes are drawn uniformly at random."""
        rng = np.random.default_rng(42)
        acc = _RecAccumulator()
        for query_codes in query_chemicals.values():
            for _ in range(n_samples):
                size = min(max(K_VALUES), len(vocab))
                sample = list(rng.choice(vocab, size=size, replace=False))  # type: ignore[arg-type]
                acc.add(sample, query_codes, weights, fallback)
        return acc.to_result("Random", len(query_chemicals))

    def _oracle_baseline(
        self,
        query_chemicals: Dict[str, FrozenSet[str]],
        weights: Dict[str, float],
        fallback: float,
    ) -> RecommendationCoverageResult:
        """Ceiling: top-k recommendations are the query's own highest-weight codes."""
        acc = _RecAccumulator()
        for query_codes in query_chemicals.values():
            ranked = sorted(query_codes, key=lambda c: weights.get(c, fallback), reverse=True)
            acc.add(ranked, query_codes, weights, fallback)
        return acc.to_result("Oracle", len(query_chemicals))

    def evaluate(self, profiles: ActivityProfiles) -> List[RecommendationCoverageResult]:
        """Evaluate the shipped recommender plus Random and Oracle baselines.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            List[RecommendationCoverageResult]: ShippedRecommender, Random, Oracle.
        """
        weights = _idf_weights(profiles.index_global)
        fallback = max(weights.values()) if weights else 1.0
        vocab = sorted({c for codes in profiles.index_global.values() for c in codes})

        chemicals = self._fetch_test_chemicals()
        shipped = _RecAccumulator()
        query_chemicals: Dict[str, FrozenSet[str]] = {}
        n_eval = 0

        for chem in chemicals:
            cas, smiles = chem["cas_number"], chem.get("smiles", "")
            query_codes = profiles.query_global.get(cas, frozenset())
            if not smiles or not query_codes:
                continue
            query_chemicals[cas] = query_codes
            ranked = self._recommended_codes(smiles)
            shipped.add(ranked, query_codes, weights, fallback)
            n_eval += 1

        result = shipped.to_result("ShippedRecommender", n_eval)
        logger.info(
            f"[ShippedRecommender] WeightedCoverage@10={result.weighted_coverage_at_k.get(10, 0):.4f}, "
            f"CodePrecision@10={result.code_precision_at_k.get(10, 0):.4f}, n={result.n_queries}"
        )
        return [
            result,
            self._random_baseline(query_chemicals, vocab, weights, fallback),
            self._oracle_baseline(query_chemicals, weights, fallback),
        ]
