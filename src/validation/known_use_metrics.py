# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Known-use test: the recommender queried with a chemical and a use it actually declared.

For every test-split chemical and every distinct condition of use (Activity name) declared in its test-split reports,
the query is the chemical's SMILES plus that use. The truth is the set of reduction codes on its test-split reports that
declare that use. All methods read index-split reports only and share the record cap ``RECORD_CAP``. Methods: the fused
retriever (``Fused``), the legacy path with the hard condition-of-use filter (``Legacy``), ``MostCommon`` and ``Random``.
Queries are averaged within each chemical, and the chemical is the unit of the paired Wilcoxon tests against Fused.

Author: Jose D. Hernandez-Betancur
Date: 2026-11-04
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import wilcoxon

from src.neo4j_connector import Neo4jConnector
from src.rag_graph.generation.rag_pipeline import tabulate_records
from src.rag_graph.retrieval import RetrieverFactory
from src.validation.anchor_builder import ActivityProfiles
from src.validation.recommendation_metrics import RECORD_CAP, distinct_codes
from src.validation.retrieval_metrics import (
    K_VALUES,
    _code_precision,
    _coverage,
    _idf_weights,
    _weighted_coverage,
)

logger = logging.getLogger(__name__)

METHOD_FUSED, METHOD_LEGACY, METHOD_COMMON, METHOD_RANDOM = "Fused", "Legacy", "MostCommon", "Random"
RANDOM_DRAWS = 50
RANDOM_SEED = 42
TOLERANCE = 1e-9
Scores = Dict[int, Dict[str, float]]

_TEST_USE_QUERY = """
MATCH (c:Chemical)<-[:REPORTS_ON]-(r:Report)-[:DOCUMENTS]->(red:Reduction)
WHERE r.split = 'test' AND c.canonical_smiles IS NOT NULL AND red.code IS NOT NULL
OPTIONAL MATCH (r)-[:RECORDS]->(a:Activity)
RETURN c.cas_number AS cas, c.canonical_smiles AS smiles, elementId(r) AS rid,
       collect(DISTINCT red.code) AS codes, collect(DISTINCT a.name) AS acts
"""


@dataclass(frozen=True)
class KnownUseQuery:
    """One (chemical, declared use) query.

    Attributes:
        cas (str): Chemical CAS number.
        smiles (str): Chemical SMILES.
        use (str): A condition of use (Activity name) declared in the chemical's test reports.
        truth (FrozenSet[str]): Reduction codes on its test reports declaring that use.
    """

    cas: str
    smiles: str
    use: str
    truth: FrozenSet[str]


@dataclass
class KnownUseResult:
    """Known-use quality of one method.

    Attributes:
        method (str): 'Fused', 'Legacy', 'MostCommon' or 'Random'.
        weighted_coverage_at_k (Dict[int, float]): IDF-weighted coverage, mean over chemicals.
        code_precision_at_k (Dict[int, float]): Code precision, mean over chemicals.
        raw_coverage_at_k (Dict[int, float]): Unweighted coverage, mean over chemicals.
        n_queries (int): Number of (chemical, use) queries.
        n_chemicals (int): Number of chemicals.
        mean_distinct_codes (Optional[float]): Mean distinct codes listed per query (retrievers only).
        p_vs_fused (Optional[float]): Two-sided paired Wilcoxon p of WeightedCov@10 against Fused, by chemical.
    """

    method: str
    weighted_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    code_precision_at_k: Dict[int, float] = field(default_factory=dict)
    raw_coverage_at_k: Dict[int, float] = field(default_factory=dict)
    n_queries: int = 0
    n_chemicals: int = 0
    mean_distinct_codes: Optional[float] = None
    p_vs_fused: Optional[float] = None


def query_scores(truth: FrozenSet[str], ranked: Sequence[str], weights: Dict[str, float], fallback: float) -> Scores:
    """Metrics at every k for one ranked list.

    Args:
        truth (FrozenSet[str]): Ground-truth codes.
        ranked (Sequence[str]): Ranked distinct codes.
        weights (Dict[str, float]): IDF weights.
        fallback (float): Weight for codes unseen in the index.

    Returns:
        Scores: k to ``wcov``, ``prec`` and ``raw``.
    """
    out: Scores = {}
    for k in K_VALUES:
        rec = frozenset(ranked[:k])
        out[k] = {
            "wcov": _weighted_coverage(truth, rec, weights, fallback),
            "prec": _code_precision(truth, rec),
            "raw": _coverage(truth, rec),
        }
    return out


def mean_scores(rows: Sequence[Scores]) -> Scores:
    """Average several score dictionaries (random draws, or the queries of one chemical).

    Args:
        rows (Sequence[Scores]): Score dictionaries.

    Returns:
        Scores: Element-wise mean.
    """
    return {k: {m: float(np.mean([r[k][m] for r in rows])) for m in rows[0][k]} for k in rows[0]}


def by_chemical(queries: Sequence[KnownUseQuery], per_query: Sequence[Scores]) -> List[Scores]:
    """Average per-query scores within each chemical (sorted by CAS).

    Args:
        queries (Sequence[KnownUseQuery]): Queries, parallel to ``per_query``.
        per_query (Sequence[Scores]): Scores of one method.

    Returns:
        List[Scores]: One score dictionary per chemical.
    """
    groups: Dict[str, List[Scores]] = {}
    for q, scores in zip(queries, per_query):
        groups.setdefault(q.cas, []).append(scores)
    return [mean_scores(groups[cas]) for cas in sorted(groups)]


def paired_p(x: Sequence[float], y: Sequence[float]) -> float:
    """Two-sided Wilcoxon signed-rank p of paired values (1.0 when every difference is zero).

    Args:
        x (Sequence[float]): Values of the method under test.
        y (Sequence[float]): Values of the reference, same order.

    Returns:
        float: The p-value.
    """
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.abs(d) > TOLERANCE]
    return float(wilcoxon(d).pvalue) if len(d) else 1.0  # pyright: ignore[reportAttributeAccessIssue]


def summarise(
    method: str, queries: Sequence[KnownUseQuery], per_query: Sequence[Scores], sizes: Optional[Sequence[int]]
) -> Tuple[KnownUseResult, List[float]]:
    """Chemical-level means of one method and its per-chemical WeightedCov@10 (for the paired tests).

    Args:
        method (str): Method name.
        queries (Sequence[KnownUseQuery]): Queries.
        per_query (Sequence[Scores]): Scores per query.
        sizes (Optional[Sequence[int]]): Distinct codes listed per query, when applicable.

    Returns:
        Tuple[KnownUseResult, List[float]]: The result row and the per-chemical WeightedCov@10.
    """
    chem = by_chemical(queries, per_query)
    mean = lambda m: {k: float(np.mean([c[k][m] for c in chem])) for k in K_VALUES}  # noqa: E731
    result = KnownUseResult(
        method,
        mean("wcov"),
        mean("prec"),
        mean("raw"),
        len(queries),
        len(chem),
        float(np.mean(sizes)) if sizes is not None else None,
    )
    return result, [c[10]["wcov"] for c in chem]


def build_queries(rows: Sequence[Mapping[str, Any]]) -> List[KnownUseQuery]:
    """Build one query per (chemical, declared use) from per-report rows.

    Args:
        rows (Sequence[Mapping[str, Any]]): One row per test report with ``cas``, ``smiles``, ``codes`` and ``acts``.

    Returns:
        List[KnownUseQuery]: Queries sorted by CAS then use; truth is the codes of the reports declaring that use.
    """
    truth: Dict[Tuple[str, str], set] = {}
    smiles: Dict[str, str] = {}
    for row in rows:
        smiles[row["cas"]] = row["smiles"]
        for use in row["acts"]:
            truth.setdefault((row["cas"], use), set()).update(row["codes"])
    return [KnownUseQuery(c, smiles[c], u, frozenset(truth[(c, u)])) for c, u in sorted(truth)]


class KnownUseMetrics:
    """Scores Fused, Legacy, MostCommon and Random on (chemical, declared use) queries."""

    def __init__(self) -> None:
        self._connector = Neo4jConnector()
        self._factory = RetrieverFactory()

    def _fetch_queries(self) -> List[KnownUseQuery]:
        with self._connector.session() as session:
            rows = [dict(row) for row in session.run(_TEST_USE_QUERY)]
        return build_queries(rows)

    def _fused_codes(self, query: KnownUseQuery) -> List[str]:
        result = self._factory.retrieve_fused(
            chemical_smiles=query.smiles, condition_of_use=query.use, report_split="index", limit=RECORD_CAP
        )
        return distinct_codes(result.records)

    def _legacy_codes(self, query: KnownUseQuery) -> List[str]:
        result = self._factory.retrieve_multi_result(
            chemical_smiles=query.smiles,
            condition_of_use=query.use,
            include_candidates=True,
            report_split="index",
            cou_filter="hard",
        )
        return distinct_codes(tabulate_records(result.retrieved_nodes, result.subgraphs, limit=RECORD_CAP))

    @staticmethod
    def _popularity(profiles: ActivityProfiles) -> List[str]:
        counts: Dict[str, int] = {}
        for codes in profiles.index_global.values():
            for code in codes:
                counts[code] = counts.get(code, 0) + 1
        return [code for code, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    def _random_scores(
        self, queries: Sequence[KnownUseQuery], vocab: List[str], weights: Dict[str, float], fb: float
    ) -> List[Scores]:
        rng = np.random.default_rng(RANDOM_SEED)
        size = min(max(K_VALUES), len(vocab))
        draws = [[list(rng.choice(vocab, size=size, replace=False)) for _ in range(RANDOM_DRAWS)] for _ in queries]
        return [mean_scores([query_scores(q.truth, d, weights, fb) for d in ds]) for q, ds in zip(queries, draws)]

    def evaluate(self, profiles: ActivityProfiles) -> List[KnownUseResult]:
        """Score all methods on every (test chemical, declared use) query.

        Args:
            profiles (ActivityProfiles): Activity profiles (the index split gives weights and popularity).

        Returns:
            List[KnownUseResult]: Fused, Legacy, MostCommon, Random with chemical-level means and paired p-values.
        """
        weights = _idf_weights(profiles.index_global)
        fallback = max(weights.values()) if weights else 1.0
        queries = self._fetch_queries()
        ranked = {
            METHOD_FUSED: [self._fused_codes(q) for q in queries],
            METHOD_LEGACY: [self._legacy_codes(q) for q in queries],
        }
        popular = self._popularity(profiles)
        scores = {m: [query_scores(q.truth, r, weights, fallback) for q, r in zip(queries, rs)] for m, rs in ranked.items()}
        scores[METHOD_COMMON] = [query_scores(q.truth, popular, weights, fallback) for q in queries]
        vocab = sorted({c for codes in profiles.index_global.values() for c in codes})
        scores[METHOD_RANDOM] = self._random_scores(queries, vocab, weights, fallback)
        results, cov10 = {}, {}
        for method, per_query in scores.items():
            sizes = [len(r) for r in ranked[method]] if method in ranked else None
            results[method], cov10[method] = summarise(method, queries, per_query, sizes)
        for method, row in results.items():
            row.p_vs_fused = None if method == METHOD_FUSED else paired_p(cov10[METHOD_FUSED], cov10[method])
        logger.info(f"[Known use] {len(queries)} queries, {results[METHOD_FUSED].n_chemicals} chemicals")
        return list(results.values())
