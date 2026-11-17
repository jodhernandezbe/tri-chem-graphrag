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

import collections
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

import numpy as np

from src.neo4j_connector import Neo4jConnector
from src.rag_graph.generation.rag_pipeline import tabulate_records
from src.rag_graph.retrieval import RetrieverFactory
from src.rag_graph.retrieval.switches import (
    RANKING_SIMILARITY,
    RETRIEVAL_FUSED,
    RETRIEVAL_LEGACY,
    compute_code_idf,
    resolve_switch,
)
from src.validation.anchor_builder import ActivityProfiles
from src.validation.retrieval_metrics import (
    K_VALUES,
    _code_precision,
    _coverage,
    _idf_weights,
    _weighted_coverage,
)

logger = logging.getLogger(__name__)

# Report split the evaluator may read: withheld (test) reports must stay hidden from the recommender.
_INDEX_SPLIT: str = "index"

# Record cap per query for every method (shipped default and Legacy alike), fixed in advance as 5000 de-duplicated
# records so that each method can list at least 20 distinct codes (k = 20) where its evidence allows.
RECORD_CAP: int = 5000
_MAX_RECORDS: int = RECORD_CAP

_FETCH_TEST_CHEMICALS_WITH_SMILES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)
WHERE r.split = 'test' AND c.canonical_smiles IS NOT NULL
WITH DISTINCT c
RETURN c.cas_number AS cas_number, c.canonical_smiles AS smiles
"""


def distinct_codes(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Distinct reduction codes of ranked records, first occurrence order.

    Args:
        records (Sequence[Dict[str, Any]]): Rows with ``ReductionCode``.

    Returns:
        List[str]: Distinct non-empty codes, best first.
    """
    seen: Set[str] = set()
    codes: List[str] = []
    for rec in records:
        code = rec.get("ReductionCode")
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


@dataclass
class RecommendationCoverageResult:
    """Top-k recommendation quality for the shipped recommender or a baseline.

    The cutoff k is the number of distinct top-ranked recommended activity codes.

    Attributes:
        method (str): 'Shipped (fused)' or 'Shipped (legacy)' (the configured default), 'Legacy', 'MostCommon',
            'Random', or 'Oracle'; 'ShippedRecommender' when a ranking or analogue override is evaluated.
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

    def _recommended_codes(
        self,
        smiles: str,
        cas: Optional[str] = None,
        ranking: Optional[str] = None,
        analogues: Optional[str] = None,
        code_idf: Optional[Dict[str, float]] = None,
        max_records: int = _MAX_RECORDS,
        include_candidates: bool = False,
    ) -> List[str]:
        """Return distinct recommended activity codes for a SMILES, ranked by the chosen switches.

        Runs the production path (vector + graph traversal + tabulation) restricted to
        index-split reports, so the chemical's own withheld reports cannot be read back. Context
        profiles and ``code_idf`` must likewise come from index-split reports.

        Args:
            smiles (str): Query chemical SMILES.
            cas (Optional[str]): Query CAS, used by ``analogues='context'`` to find its index-split profile.
            ranking (Optional[str]): ``similarity`` (default when ``None``), ``specificity`` or ``consensus_specificity``.
            analogues (Optional[str]): ``molecular`` or ``context``; ``None`` follows config.
            code_idf (Optional[Dict[str, float]]): IDF per code, required for the non-similarity rankings.
            max_records (int): Record cap before distinct codes are taken.
            include_candidates (bool): Also expand through the Candidate indices, as the deployed legacy pipeline does.

        Returns:
            List[str]: Distinct recommended source-reduction codes, best first.
        """
        kwargs: Dict[str, Any] = {"report_split": _INDEX_SPLIT, "include_candidates": include_candidates}
        if analogues is not None:
            kwargs.update(analogues=analogues, query_cas=cas)
        nodes, subgraphs = self._factory.retrieve_multi(chemical_smiles=smiles, sector_query="", **kwargs)
        extra: Dict[str, Any] = {}
        if ranking not in (None, RANKING_SIMILARITY):
            extra = {"ranking": ranking, "code_idf": code_idf}
        return distinct_codes(tabulate_records(nodes, subgraphs, limit=max_records, **extra))

    def _fused_codes(self, smiles: str, max_records: int = RECORD_CAP) -> List[str]:
        """Return distinct recommended codes from the fused retriever, chemical only, on index-split reports.

        Args:
            smiles (str): Query chemical SMILES.
            max_records (int): Record cap before distinct codes are taken.

        Returns:
            List[str]: Distinct recommended source-reduction codes, best first.
        """
        result = self._factory.retrieve_fused(chemical_smiles=smiles, report_split=_INDEX_SPLIT, limit=max_records)
        return distinct_codes(result.records)

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

    def _most_common_baseline(
        self,
        query_chemicals: Dict[str, FrozenSet[str]],
        profiles: ActivityProfiles,
        weights: Dict[str, float],
        fallback: float,
    ) -> RecommendationCoverageResult:
        """Popularity floor: always recommend the codes reported by the most index chemicals."""
        counts = collections.Counter(code for codes in profiles.index_global.values() for code in codes)
        ranked = [code for code, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        acc = _RecAccumulator()
        for query_codes in query_chemicals.values():
            acc.add(ranked, query_codes, weights, fallback)
        return acc.to_result("MostCommon", len(query_chemicals))

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

    def _rankers(
        self,
        retrieval: str,
        shipped_path: bool,
        legacy_args: Tuple[Optional[str], Optional[str], Optional[Dict[str, float]]],
    ) -> Dict[str, Callable[[str, str], List[str]]]:
        """Method name to ``(cas, smiles) -> ranked codes`` for the recommender rows of Table 6."""
        ranking, analogues, code_idf = legacy_args
        if not shipped_path:
            return {"ShippedRecommender": lambda cas, smi: self._recommended_codes(smi, cas, ranking, analogues, code_idf)}
        legacy: Callable[[str, str], List[str]] = lambda cas, smi: self._recommended_codes(
            smi, include_candidates=True
        )  # noqa: E731
        if retrieval == RETRIEVAL_LEGACY:
            return {f"Shipped ({RETRIEVAL_LEGACY})": legacy}
        return {f"Shipped ({RETRIEVAL_FUSED})": lambda cas, smi: self._fused_codes(smi), "Legacy": legacy}

    def evaluate(
        self,
        profiles: ActivityProfiles,
        ranking: Optional[str] = None,
        analogues: Optional[str] = None,
        retrieval: Optional[str] = None,
    ) -> List[RecommendationCoverageResult]:
        """Evaluate the shipped default and Legacy plus MostCommon, Random and Oracle baselines.

        The query is the chemical alone (no use, no sector). Every method reads index-split reports only and
        shares the record cap ``RECORD_CAP``. With the default switches the rows are the configured retrieval
        (``Shipped (fused)`` as shipped) and ``Legacy``; overriding ``ranking`` or ``analogues`` evaluates that
        legacy variant as ``ShippedRecommender``.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.
            ranking (Optional[str]): Ranking switch; ``None`` is the shipped ``similarity``.
            analogues (Optional[str]): Analogue switch; ``None`` follows config (shipped ``molecular``).
            retrieval (Optional[str]): ``legacy`` or ``fused``; ``None`` follows config.

        Returns:
            List[RecommendationCoverageResult]: Recommender rows, then MostCommon, Random, Oracle.
        """
        weights = _idf_weights(profiles.index_global)
        fallback = max(weights.values()) if weights else 1.0
        vocab = sorted({c for codes in profiles.index_global.values() for c in codes})
        mode = resolve_switch("ranking", ranking) if ranking is not None else RANKING_SIMILARITY
        code_idf = compute_code_idf(profiles.index_global) if mode != RANKING_SIMILARITY else None
        shipped_path = mode == RANKING_SIMILARITY and analogues is None
        rankers = self._rankers(resolve_switch("retrieval", retrieval), shipped_path, (mode, analogues, code_idf))
        accumulators = {name: _RecAccumulator() for name in rankers}
        query_chemicals: Dict[str, FrozenSet[str]] = {}
        for chem in self._fetch_test_chemicals():
            cas, smiles = chem["cas_number"], chem.get("smiles", "")
            query_codes = profiles.query_global.get(cas, frozenset())
            if not smiles or not query_codes:
                continue
            query_chemicals[cas] = query_codes
            for name, rank in rankers.items():
                accumulators[name].add(rank(cas, smiles), query_codes, weights, fallback)
        rows = [acc.to_result(name, len(query_chemicals)) for name, acc in accumulators.items()]
        for row in rows:
            logger.info(
                f"[{row.method}] WeightedCoverage@10={row.weighted_coverage_at_k.get(10, 0):.4f}, "
                f"CodePrecision@10={row.code_precision_at_k.get(10, 0):.4f}, n={row.n_queries}"
            )
        return rows + [
            self._most_common_baseline(query_chemicals, profiles, weights, fallback),
            self._random_baseline(query_chemicals, vocab, weights, fallback),
            self._oracle_baseline(query_chemicals, weights, fallback),
        ]
