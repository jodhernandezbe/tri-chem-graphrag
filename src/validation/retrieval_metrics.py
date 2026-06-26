# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Extrinsic retrieval quality metrics for chemical embedding evaluation.

Evaluates each chemical embedding type against a graded-relevance anchor
(structural Tanimoto or operational Jaccard Activity Similarity) using
nDCG@k, Recall@k, and MRR.  A Wilcoxon signed-rank test assesses whether
differences between embedding types are statistically significant.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Protocol, Set, Tuple

import numpy as np
from scipy.stats import wilcoxon

from src.neo4j_connector import Neo4jConnector
from src.rag_graph.retrieval import RetrieverFactory
from src.validation.anchor_builder import ActivityProfiles, AnchorBuilder

logger = logging.getLogger(__name__)

K_VALUES: Tuple[int, ...] = (5, 10, 20)
NAICS_LEVELS: Tuple[int, ...] = (2, 3, 4, 6)

# Single source of truth for the chemical embedding methods under evaluation.
# Imported by main.py and recommendation_metrics.py so every table/chart stays in sync.
CHEMICAL_EMBEDDING_TYPES: Tuple[str, ...] = (
    "MorganFingerprint",
    "MolDeBERTa",
    "Hybrid",
)

# Neighbour chemicals to pull before flattening to a ranked code list; large enough
# that the top max(K_VALUES) distinct codes are always available.
_CODE_POOL: int = 50

_FETCH_TEST_CHEMICALS = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)
WHERE r.split = 'test' AND c.canonical_smiles IS NOT NULL
WITH DISTINCT c
RETURN c.cas_number AS cas_number, c.canonical_smiles AS smiles
"""

_FETCH_TEST_CHEMICALS_WITH_SMILES = _FETCH_TEST_CHEMICALS


class AnchorLike(Protocol):
    """Structural duck-type protocol shared by StructuralAnchors and OperationalAnchors."""

    index: Dict[str, int]

    def grades_for(self, cas_number: str) -> Dict[str, int]:
        """Return graded relevance for all other chemicals, keyed by CAS number."""
        ...


@dataclass
class RetrievalMetricsResult:
    """Per-embedding-type retrieval quality metrics.

    Attributes:
        embedding_type (str): Name of the evaluated embedding type.
        anchor_type (str): Label of the anchor used ('structural' or 'operational').
        ndcg_at_k (Dict[int, float]): Mean nDCG@k over all test queries.
        recall_at_k (Dict[int, float]): Fraction of queries with a grade-2 hit in top k.
        mrr (float): Mean reciprocal rank of the first grade-2 item.
        n_queries (int): Number of test queries evaluated.
        per_query_ndcg (Dict[int, List[float]]): Per-query nDCG@k for significance testing.
    """

    embedding_type: str
    anchor_type: str = "structural"
    ndcg_at_k: Dict[int, float] = field(default_factory=dict)
    recall_at_k: Dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    n_queries: int = 0
    per_query_ndcg: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class SignificanceTestResult:
    """Result of a Wilcoxon signed-rank test comparing two embedding types at one k.

    Attributes:
        embedding_a (str): First embedding type.
        embedding_b (str): Second embedding type.
        anchor_type (str): Anchor used for the comparison.
        k (int): Cutoff value for nDCG comparison.
        statistic (float): Wilcoxon test statistic.
        pvalue (float): Two-sided p-value.
        significant (bool): Whether pvalue < 0.05.
    """

    embedding_a: str
    embedding_b: str
    anchor_type: str
    k: int
    statistic: float
    pvalue: float
    significant: bool


def _dcg(relevances: List[int], k: int) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances[:k]))


def _ndcg(ranked_cas: List[str], query_cas: str, anchors: AnchorLike, k: int) -> float:
    grades = anchors.grades_for(query_cas)
    ranked = [cas for cas in ranked_cas if cas != query_cas]
    rel = [grades.get(cas, 0) for cas in ranked[:k]]
    ideal = sorted(grades.values(), reverse=True)[:k]
    dcg = _dcg(rel, k)
    idcg = _dcg(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _recall_at_k(ranked_cas: List[str], query_cas: str, anchors: AnchorLike, k: int) -> float:
    grades = anchors.grades_for(query_cas)
    ranked = [cas for cas in ranked_cas if cas != query_cas][:k]
    return 1.0 if any(grades.get(cas, 0) == 2 for cas in ranked) else 0.0


def _mrr(ranked_cas: List[str], query_cas: str, anchors: AnchorLike) -> float:
    grades = anchors.grades_for(query_cas)
    ranked = [cas for cas in ranked_cas if cas != query_cas]
    for i, cas in enumerate(ranked):
        if grades.get(cas, 0) == 2:
            return 1.0 / (i + 1)
    return 0.0


class RetrievalMetrics:
    """Evaluates vector retrieval ranking against graded-relevance anchors."""

    def __init__(self, anchor_builder: Optional[AnchorBuilder] = None) -> None:
        self._connector = Neo4jConnector()
        self._factory = RetrieverFactory()
        self._anchor_builder = anchor_builder or AnchorBuilder()

    def _fetch_test_chemicals(self) -> List[Dict[str, str]]:
        with self._connector.session() as session:
            return [dict(r) for r in session.run(_FETCH_TEST_CHEMICALS)]

    def _vector_ranked_list(self, embedding_type: str, smiles: str, k_max: int) -> List[str]:
        vector_store = self._factory.get_vector_store(embedding_type)
        hits = vector_store.similarity_search_with_score(smiles, k=k_max)
        return [doc.metadata.get("cas_number", "") for doc, _ in hits if doc.metadata.get("cas_number")]

    def _query_metrics(
        self,
        cas: str,
        ranked: List[str],
        anchors: AnchorLike,
    ) -> Tuple[float, Dict[int, float], Dict[int, float]]:
        mrr_val = _mrr(ranked, cas, anchors)
        ndcg = {k: _ndcg(ranked, cas, anchors, k) for k in K_VALUES}
        recall = {k: _recall_at_k(ranked, cas, anchors, k) for k in K_VALUES}
        return mrr_val, ndcg, recall

    def _aggregate(
        self,
        embedding_type: str,
        anchor_type: str,
        all_mrr: List[float],
        all_ndcg: Dict[int, List[float]],
        all_recall: Dict[int, List[float]],
    ) -> RetrievalMetricsResult:
        return RetrievalMetricsResult(
            embedding_type=embedding_type,
            anchor_type=anchor_type,
            ndcg_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in all_ndcg.items()},
            recall_at_k={k: float(np.mean(v)) if v else 0.0 for k, v in all_recall.items()},
            mrr=float(np.mean(all_mrr)) if all_mrr else 0.0,
            n_queries=len(all_mrr),
            per_query_ndcg=all_ndcg,
        )

    def evaluate_embedding_type(
        self,
        embedding_type: str,
        anchors: AnchorLike,
        anchor_type: str = "structural",
    ) -> RetrievalMetricsResult:
        """Evaluate nDCG@k, Recall@k, and MRR for one chemical embedding type.

        Uses the vector store directly (without graph traversal) so that the
        ranking reflects pure embedding quality against the provided anchor.

        Args:
            embedding_type (str): One of MorganFingerprint, MolDeBERTa, or Hybrid.
            anchors (AnchorLike): Graded relevance anchor (structural or operational).
            anchor_type (str): Label stored in the result for reporting ('structural'
                or 'operational').

        Returns:
            RetrievalMetricsResult: Averaged metrics and per-query nDCG for significance tests.
        """
        chemicals = self._fetch_test_chemicals()
        all_mrr: List[float] = []
        all_ndcg: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        all_recall: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        k_max = max(K_VALUES)

        for chem in chemicals:
            cas, smiles = chem["cas_number"], chem.get("smiles", "")
            if not smiles or cas not in anchors.index:
                continue
            ranked = self._vector_ranked_list(embedding_type, smiles, k_max)
            mrr_val, ndcg, recall = self._query_metrics(cas, ranked, anchors)
            all_mrr.append(mrr_val)
            for k in K_VALUES:
                all_ndcg[k].append(ndcg[k])
                all_recall[k].append(recall[k])

        result = self._aggregate(embedding_type, anchor_type, all_mrr, all_ndcg, all_recall)
        logger.info(
            f"[{embedding_type}/{anchor_type}] nDCG@10={result.ndcg_at_k.get(10, 0):.4f}, "
            f"Recall@10={result.recall_at_k.get(10, 0):.4f}, MRR={result.mrr:.4f}, "
            f"n={result.n_queries}"
        )
        return result

    def evaluate_all(
        self,
        anchors: Optional[AnchorLike] = None,
        anchor_type: str = "structural",
    ) -> List[RetrievalMetricsResult]:
        """Evaluate all chemical embedding types against the provided anchor.

        Args:
            anchors (Optional[AnchorLike]): Graded relevance anchor.  Builds structural
                anchors automatically if not provided.
            anchor_type (str): Label for the anchor used in result records.

        Returns:
            List[RetrievalMetricsResult]: One result per embedding type.
        """
        if anchors is None:
            anchors = self._anchor_builder.build_structural_anchors()
            anchor_type = "structural"
        results: List[RetrievalMetricsResult] = []
        for et in CHEMICAL_EMBEDDING_TYPES:
            try:
                results.append(self.evaluate_embedding_type(et, anchors, anchor_type))
            except Exception as exc:
                logger.warning(f"Skipping retrieval metrics for '{et}' (index/model unavailable): {exc}")
        return results

    def significance_test(
        self,
        result_a: RetrievalMetricsResult,
        result_b: RetrievalMetricsResult,
        k: int = 10,
    ) -> SignificanceTestResult:
        """Run Wilcoxon signed-rank test on per-query nDCG@k between two embedding types.

        Args:
            result_a (RetrievalMetricsResult): Metrics for the first embedding type.
            result_b (RetrievalMetricsResult): Metrics for the second embedding type.
            k (int): nDCG cutoff to compare (must be in K_VALUES).

        Returns:
            SignificanceTestResult: Test statistic, p-value, and significance flag.
        """
        scores_a = result_a.per_query_ndcg.get(k, [])
        scores_b = result_b.per_query_ndcg.get(k, [])
        n = min(len(scores_a), len(scores_b))
        wt = wilcoxon(scores_a[:n], scores_b[:n])
        stat_val = float(wt[0])  # type: ignore[arg-type]
        pval_val = float(wt[1])  # type: ignore[arg-type]
        anchor_type = result_a.anchor_type
        return SignificanceTestResult(
            embedding_a=result_a.embedding_type,
            embedding_b=result_b.embedding_type,
            anchor_type=anchor_type,
            k=k,
            statistic=stat_val,
            pvalue=pval_val,
            significant=pval_val < 0.05,
        )


# ---------------------------------------------------------------------------
# Activity Coverage Evaluation (PRIMARY)
# ---------------------------------------------------------------------------


@dataclass
class ActivityCoverageResult:
    """Per-embedding-type activity coverage metrics.

    WeightedCoverage@k (PRIMARY) is the IDF-weighted fraction of the query chemical's
    actual source reduction activities (test split) recovered from the union of the
    top-k retrieved chemicals' activities (full index); rare codes count more, so it is
    robust to the popularity floor that defeats raw union coverage.  CodePrecision@k
    (PRIMARY) is the fraction of recovered codes that are relevant, penalizing breadth.
    ActivityCoverage@k is the raw (unweighted) union coverage, retained as a diagnostic.
    FamilyCoverage@k is raw coverage at the activity-family level.

    Attributes:
        embedding_type (str): Name of the evaluated embedding type.
        weighted_coverage_at_k (Dict[int, float]): Mean IDF-weighted coverage@k (primary).
        code_precision_at_k (Dict[int, float]): Mean precision@k of recovered codes (primary).
        activity_coverage_at_k (Dict[int, float]): Mean raw ActivityCoverage@k (diagnostic).
        family_coverage_at_k (Dict[int, float]): Mean FamilyCoverage@k (diagnostic).
        n_queries (int): Number of test chemicals evaluated.
        per_query_coverage (Dict[int, List[float]]): Per-query WeightedCoverage@k for significance.
    """

    embedding_type: str
    weighted_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    code_precision_at_k: Dict[int, float] = field(default_factory=dict)
    activity_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    family_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    n_queries: int = 0
    per_query_coverage: Dict[int, List[float]] = field(default_factory=dict)


@dataclass
class SectorCoverageResult:
    """Sector-conditioned ActivityCoverage@k at one NAICS hierarchy level.

    Attributes:
        embedding_type (str): Name of the evaluated embedding type.
        naics_level (int): NAICS digit depth used (2, 3, 4, or 6).
        sector_coverage_at_k (Dict[int, float]): Mean SectorCoverage@k.
        n_queries (int): Number of (query, NAICS-prefix) pairs evaluated.
    """

    embedding_type: str
    naics_level: int
    sector_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    n_queries: int = 0


_FETCH_TEST_CHEMICALS_WITH_SMILES = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)
WHERE r.split = 'test' AND c.canonical_smiles IS NOT NULL
WITH DISTINCT c
RETURN c.cas_number AS cas_number, c.canonical_smiles AS smiles
"""


def _coverage(query_codes: FrozenSet[str], retrieved_codes: "Set[str] | FrozenSet[str]") -> float:
    if not query_codes:
        return 0.0
    return len(query_codes & retrieved_codes) / len(query_codes)


def _idf_weights(index_global: Dict[str, FrozenSet[str]]) -> Dict[str, float]:
    """Compute smoothed inverse-document-frequency weights per activity code.

    Down-weights ubiquitous source reduction codes (e.g. generic operating practices)
    so that covering a rare, specific code counts far more than covering a common one.
    This removes the popularity inflation that let uniformly random retrieval win the
    naive union-coverage metric.

    Args:
        index_global (Dict[str, FrozenSet[str]]): CAS → activity codes over the full index.

    Returns:
        Dict[str, float]: Activity code → IDF weight (always positive).
    """
    n_chem = len(index_global)
    doc_freq: Dict[str, int] = {}
    for codes in index_global.values():
        for code in codes:
            doc_freq[code] = doc_freq.get(code, 0) + 1
    return {code: math.log((n_chem + 1) / (freq + 1)) + 1.0 for code, freq in doc_freq.items()}


def _weighted_coverage(
    query_codes: FrozenSet[str],
    retrieved_codes: "Set[str] | FrozenSet[str]",
    weights: Dict[str, float],
    fallback: float,
) -> float:
    """Fraction of the query's IDF-weighted activity mass recovered by retrieval.

    Args:
        query_codes (FrozenSet[str]): Ground-truth activity codes for the query chemical.
        retrieved_codes (Set[str] | FrozenSet[str]): Union of retrieved chemicals' codes.
        weights (Dict[str, float]): IDF weight per activity code.
        fallback (float): Weight used for query codes never seen in the index.

    Returns:
        float: Weighted coverage in [0, 1].
    """
    total = sum(weights.get(code, fallback) for code in query_codes)
    if total <= 0.0:
        return 0.0
    covered = sum(weights.get(code, fallback) for code in query_codes & retrieved_codes)
    return covered / total


def _rank_codes(ranked_chems: List[str], profiles: ActivityProfiles) -> List[str]:
    """Flatten neighbour chemicals into a ranked list of distinct activity codes.

    Codes are emitted in neighbour-similarity order (most similar chemical first);
    within a chemical they are ordered deterministically.  Scoring the top-k of this
    list — rather than the union of the top-k chemicals' full code-sets — turns
    coverage into a genuine top-k *recommendation* metric that random retrieval can
    no longer win by breadth.

    Args:
        ranked_chems (List[str]): CAS numbers ordered by descending similarity.
        profiles (ActivityProfiles): Per-chemical activity profile sets.

    Returns:
        List[str]: Distinct activity codes in recommendation order.
    """
    codes: List[str] = []
    seen: Set[str] = set()
    for cas in ranked_chems:
        for code in sorted(profiles.index_global.get(cas, frozenset())):
            if code not in seen:
                seen.add(code)
                codes.append(code)
    return codes


def _code_precision(query_codes: FrozenSet[str], retrieved_codes: "Set[str] | FrozenSet[str]") -> float:
    """Fraction of recovered activity codes that are actually relevant to the query.

    Penalizes retrieval that dumps many irrelevant codes — the failure mode that
    inflates union-coverage for random retrieval.

    Args:
        query_codes (FrozenSet[str]): Ground-truth activity codes for the query chemical.
        retrieved_codes (Set[str] | FrozenSet[str]): Union of retrieved chemicals' codes.

    Returns:
        float: Precision in [0, 1]; 0.0 when nothing was retrieved.
    """
    if not retrieved_codes:
        return 0.0
    return len(query_codes & retrieved_codes) / len(retrieved_codes)


class _CoverageAccumulator:
    """Accumulates per-query coverage metrics across k cutoffs into an ActivityCoverageResult."""

    def __init__(self) -> None:
        self._weighted: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        self._precision: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        self._raw: Dict[int, List[float]] = {k: [] for k in K_VALUES}
        self._family: Dict[int, List[float]] = {k: [] for k in K_VALUES}

    def add(
        self,
        k: int,
        query_codes: FrozenSet[str],
        retrieved_codes: FrozenSet[str],
        query_fam: FrozenSet[str],
        profiles: ActivityProfiles,
        weights: Dict[str, float],
        fallback: float,
    ) -> None:
        """Record all coverage metrics for one (query, k) observation."""
        self._weighted[k].append(_weighted_coverage(query_codes, retrieved_codes, weights, fallback))
        self._precision[k].append(_code_precision(query_codes, retrieved_codes))
        self._raw[k].append(_coverage(query_codes, retrieved_codes))
        ret_fam = profiles.families(retrieved_codes)
        self._family[k].append(_coverage(query_fam, ret_fam))

    def to_result(self, embedding_type: str, n_queries: int) -> ActivityCoverageResult:
        """Aggregate accumulated observations into a single result record.

        Args:
            embedding_type (str): Label for the evaluated method (or baseline name).
            n_queries (int): Number of queries that contributed observations.

        Returns:
            ActivityCoverageResult: Mean metrics across queries.
        """

        def _mean(values: Dict[int, List[float]]) -> Dict[int, float]:
            return {k: float(np.mean(v)) if v else 0.0 for k, v in values.items()}

        return ActivityCoverageResult(
            embedding_type=embedding_type,
            weighted_coverage_at_k=_mean(self._weighted),
            code_precision_at_k=_mean(self._precision),
            activity_coverage_at_k=_mean(self._raw),
            family_coverage_at_k=_mean(self._family),
            n_queries=n_queries,
            per_query_coverage={k: list(v) for k, v in self._weighted.items()},
        )


class ActivityCoverageMetrics:
    """Evaluates activity coverage as a top-k recommendation task (pure vector retrieval).

    For each test chemical, embeds its SMILES, retrieves the nearest neighbour chemicals,
    flattens them into a similarity-ranked list of distinct activity codes, and measures how
    well the top-k codes match the chemical's actual test-split activities.  Scoring ranked
    codes (rather than the union of the top-k chemicals' full histories) keeps the metric from
    being won by indiscriminate breadth — uniformly random codes score poorly on both
    IDF-weighted coverage and precision.
    """

    def __init__(self) -> None:
        self._connector = Neo4jConnector()
        self._factory = RetrieverFactory()

    def _fetch_test_chemicals(self) -> List[Dict[str, str]]:
        with self._connector.session() as session:
            return [dict(r) for r in session.run(_FETCH_TEST_CHEMICALS_WITH_SMILES)]

    def _vector_ranked_list(self, embedding_type: str, smiles: str, k_max: int) -> List[str]:
        vector_store = self._factory.get_vector_store(embedding_type)
        hits = vector_store.similarity_search_with_score(smiles, k=k_max)
        return [doc.metadata.get("cas_number", "") for doc, _ in hits if doc.metadata.get("cas_number")]

    def _code_weights(self, profiles: ActivityProfiles) -> Tuple[Dict[str, float], float]:
        """Return IDF weights per activity code and the fallback weight for unseen codes.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            Tuple[Dict[str, float], float]: (code → IDF weight, fallback for index-absent codes).
        """
        weights = _idf_weights(profiles.index_global)
        fallback = max(weights.values()) if weights else 1.0
        return weights, fallback

    def _oracle_baseline(self, profiles: ActivityProfiles) -> ActivityCoverageResult:
        """Compute Oracle@k: theoretical upper bound on ActivityCoverage.

        For each test chemical, selects the k index chemicals whose index-split
        activity codes have maximum Jaccard overlap with the query chemical's
        test-split codes.  This represents the best coverage achievable by any
        retrieval method that operates on activity-code similarity.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            ActivityCoverageResult: Oracle coverage upper bound with embedding_type='Oracle'.
        """
        with self._connector.session() as session:
            smiles_cas = {r["cas_number"] for r in session.run(_FETCH_TEST_CHEMICALS_WITH_SMILES)}
        query_chemicals = {cas: codes for cas, codes in profiles.query_global.items() if cas in smiles_cas}
        index_cas = list(profiles.index_global.keys())
        weights, fallback = self._code_weights(profiles)
        acc = _CoverageAccumulator()

        for cas, query_codes in query_chemicals.items():
            if not query_codes:
                continue
            candidates = [c for c in index_cas if c != cas]
            if not candidates:
                continue

            def _jaccard(c: str) -> float:
                ic = profiles.index_global.get(c, frozenset())
                union = query_codes | ic
                return len(query_codes & ic) / len(union) if union else 0.0

            ranked_chems = sorted(candidates, key=_jaccard, reverse=True)
            ranked_codes = _rank_codes(ranked_chems, profiles)
            query_fam = profiles.families(query_codes)

            for k in K_VALUES:
                retrieved_codes = frozenset(ranked_codes[:k])
                acc.add(k, query_codes, retrieved_codes, query_fam, profiles, weights, fallback)

        result = acc.to_result("Oracle", len(query_chemicals))
        logger.info(
            f"[Oracle] WeightedCoverage@10={result.weighted_coverage_at_k.get(10, 0):.4f}, "
            f"CodePrecision@10={result.code_precision_at_k.get(10, 0):.4f}, n={result.n_queries}"
        )
        return result

    def _random_baseline(self, profiles: ActivityProfiles, n_samples: int = 200) -> ActivityCoverageResult:
        """Compute the top-k recommendation metrics for uniformly random codes (Monte-Carlo).

        For each query, recommends k activity codes drawn uniformly at random from the
        index code vocabulary.  Restricts queries to chemicals with SMILES to match the
        embedding evaluation population for a fair comparison.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.
            n_samples (int): Random draws per query.

        Returns:
            ActivityCoverageResult: Mean metrics with embedding_type='Random'.
        """
        with self._connector.session() as session:
            smiles_cas = {r["cas_number"] for r in session.run(_FETCH_TEST_CHEMICALS_WITH_SMILES)}
        query_chemicals = {cas: codes for cas, codes in profiles.query_global.items() if cas in smiles_cas}

        vocab = sorted({code for codes in profiles.index_global.values() for code in codes})
        weights, fallback = self._code_weights(profiles)
        rng = np.random.default_rng(42)
        acc = _CoverageAccumulator()
        k_max = min(max(K_VALUES), len(vocab))

        for query_codes in query_chemicals.values():
            if not query_codes:
                continue
            query_fam = profiles.families(query_codes)
            for _ in range(n_samples):
                sample = list(rng.choice(vocab, size=k_max, replace=False))  # type: ignore[arg-type]
                for k in K_VALUES:
                    retrieved_codes = frozenset(sample[:k])
                    acc.add(k, query_codes, retrieved_codes, query_fam, profiles, weights, fallback)

        return acc.to_result("Random", len(query_chemicals))

    def evaluate_embedding_type(
        self,
        embedding_type: str,
        profiles: ActivityProfiles,
    ) -> ActivityCoverageResult:
        """Evaluate the redesigned top-k recommendation metrics for one embedding type.

        Retrieves the nearest neighbour chemicals, flattens them into a ranked list of
        distinct activity codes, and scores the top-k codes with WeightedCoverage@k
        (IDF-weighted, primary), CodePrecision@k (primary), and raw coverage (diagnostic).

        Args:
            embedding_type (str): One of CHEMICAL_EMBEDDING_TYPES.
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            ActivityCoverageResult: Mean coverage metrics and per-query data.
        """
        chemicals = self._fetch_test_chemicals()
        weights, fallback = self._code_weights(profiles)
        acc = _CoverageAccumulator()
        n_eval = 0

        for chem in chemicals:
            cas, smiles = chem["cas_number"], chem.get("smiles", "")
            query_codes = profiles.query_global.get(cas, frozenset())
            if not smiles or not query_codes:
                continue
            ranked_chems = [c for c in self._vector_ranked_list(embedding_type, smiles, _CODE_POOL) if c != cas]
            ranked_codes = _rank_codes(ranked_chems, profiles)
            query_fam = profiles.families(query_codes)
            n_eval += 1

            for k in K_VALUES:
                retrieved_codes = frozenset(ranked_codes[:k])
                acc.add(k, query_codes, retrieved_codes, query_fam, profiles, weights, fallback)

        result = acc.to_result(embedding_type, n_eval)
        logger.info(
            f"[{embedding_type}] WeightedCoverage@10={result.weighted_coverage_at_k.get(10, 0):.4f}, "
            f"CodePrecision@10={result.code_precision_at_k.get(10, 0):.4f}, "
            f"ActivityCoverage@10={result.activity_coverage_at_k.get(10, 0):.4f}, n={result.n_queries}"
        )
        return result

    def evaluate_all(self, profiles: ActivityProfiles) -> List[ActivityCoverageResult]:
        """Evaluate all chemical embedding types plus random and oracle baselines.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            List[ActivityCoverageResult]: One result per available embedding type plus
                Random and Oracle.  Embedding types whose index or model cannot be loaded
                are skipped with a warning rather than aborting the run.
        """
        results: List[ActivityCoverageResult] = []
        for et in CHEMICAL_EMBEDDING_TYPES:
            try:
                results.append(self.evaluate_embedding_type(et, profiles))
            except Exception as exc:
                logger.warning(f"Skipping activity coverage for '{et}' (index/model unavailable): {exc}")
        results.append(self._random_baseline(profiles))
        results.append(self._oracle_baseline(profiles))
        return results

    def _random_baseline_sector(self, profiles: ActivityProfiles, n_samples: int = 200) -> List[SectorCoverageResult]:
        """Compute SectorCoverage@10 for uniformly random retrieval at each NAICS hierarchy level.

        Args:
            profiles (ActivityProfiles): Activity profiles for index and test splits.
            n_samples (int): Random retrieval trials per (query, NAICS-prefix) pair.

        Returns:
            List[SectorCoverageResult]: One result per NAICS level with embedding_type='Random'.
        """
        all_index_cas = list(profiles.index_global.keys())
        chemicals = self._fetch_test_chemicals()
        k_eval = 10
        rng = np.random.default_rng(42)
        results: List[SectorCoverageResult] = []

        for level in NAICS_LEVELS:
            all_cov: List[float] = []
            for chem in chemicals:
                cas = chem["cas_number"]
                prefixes = profiles.get_query_naics_prefixes(cas, level)
                if not prefixes:
                    continue
                candidates = [c for c in all_index_cas if c != cas]
                if len(candidates) < k_eval:
                    continue
                for prefix in prefixes:
                    query_codes = profiles.get_query_sector_codes(cas, prefix)
                    if not query_codes:
                        continue
                    trial_coverages: List[float] = []
                    for _ in range(n_samples):
                        sample = list(rng.choice(candidates, size=k_eval, replace=False))  # type: ignore[arg-type]
                        retrieved_codes: Set[str] = set()
                        for ret_cas in sample:
                            retrieved_codes |= profiles.get_index_sector_codes(ret_cas, prefix)
                        trial_coverages.append(_coverage(query_codes, frozenset(retrieved_codes)))
                    all_cov.append(sum(trial_coverages) / len(trial_coverages))

            result = SectorCoverageResult(
                embedding_type="Random",
                naics_level=level,
                sector_coverage_at_k={k_eval: float(np.mean(all_cov)) if all_cov else 0.0},
                n_queries=len(all_cov),
            )
            logger.info(
                f"[Random/L{level}] SectorCoverage@{k_eval}={result.sector_coverage_at_k.get(k_eval, 0):.4f}, "
                f"n_pairs={result.n_queries}"
            )
            results.append(result)

        return results

    def evaluate_naics_levels(self, profiles: ActivityProfiles) -> List[SectorCoverageResult]:
        """Evaluate SectorCoverage@10 at each NAICS hierarchy level.

        For each test chemical, considers only activities within sectors that the
        query chemical appears in (at the given NAICS level).  Retrieved chemicals
        also filtered to the same sector prefix to isolate sector-specific coverage.

        Args:
            profiles (ActivityProfiles): Per-chemical activity profile sets.

        Returns:
            List[SectorCoverageResult]: Results for each (embedding, NAICS level) pair.
        """
        chemicals = self._fetch_test_chemicals()
        results: List[SectorCoverageResult] = []
        k_eval = 10

        for et in CHEMICAL_EMBEDDING_TYPES:
            try:
                self._factory.get_vector_store(et)
            except Exception as exc:
                logger.warning(f"Skipping sector coverage for '{et}' (index/model unavailable): {exc}")
                continue
            for level in NAICS_LEVELS:
                all_cov: List[float] = []
                k_max = max(K_VALUES)

                for chem in chemicals:
                    cas, smiles = chem["cas_number"], chem.get("smiles", "")
                    prefixes = profiles.get_query_naics_prefixes(cas, level)
                    if not smiles or not prefixes:
                        continue
                    ranked = [c for c in self._vector_ranked_list(et, smiles, k_max) if c != cas]

                    for prefix in prefixes:
                        query_codes = profiles.get_query_sector_codes(cas, prefix)
                        if not query_codes:
                            continue
                        retrieved_codes: Set[str] = set()
                        for c in ranked[:k_eval]:
                            retrieved_codes |= profiles.get_index_sector_codes(c, prefix)
                        all_cov.append(_coverage(query_codes, retrieved_codes))

                result = SectorCoverageResult(
                    embedding_type=et,
                    naics_level=level,
                    sector_coverage_at_k={k_eval: float(np.mean(all_cov)) if all_cov else 0.0},
                    n_queries=len(all_cov),
                )
                logger.info(
                    f"[{et}/L{level}] SectorCoverage@{k_eval}={result.sector_coverage_at_k.get(k_eval, 0):.4f}, "
                    f"n_pairs={result.n_queries}"
                )
                results.append(result)

        results.extend(self._random_baseline_sector(profiles))
        return results
