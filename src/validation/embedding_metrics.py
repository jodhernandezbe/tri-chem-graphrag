# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Intrinsic embedding quality metrics for TRI chemical and sector embeddings.

Computes Silhouette score and Davies-Bouldin index grouped by functional class,
Spearman rank correlation between embedding cosine distance and both structural
(Tanimoto) and operational (Jaccard Activity Similarity) distance, and optional
2-D t-SNE projections for visual inspection.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import normalize

from src.neo4j_connector import Neo4jConnector
from src.validation.anchor_builder import (
    AnchorBuilder,
    OperationalAnchors,
    StructuralAnchors,
)

logger = logging.getLogger(__name__)

_EMBEDDING_QUERIES: Dict[str, str] = {
    "MorganFingerprint": (
        "MATCH (c:Chemical) WHERE c.morgan_fp_embedding IS NOT NULL "
        "RETURN c.cas_number AS cas_number, c.morgan_fp_embedding AS embedding"
    ),
    "MolDeBERTa": (
        "MATCH (c:Chemical) WHERE c.moldeberta_embedding IS NOT NULL "
        "RETURN c.cas_number AS cas_number, c.moldeberta_embedding AS embedding"
    ),
    "Hybrid": (
        "MATCH (c:Chemical) WHERE c.hybrid_embedding IS NOT NULL "
        "RETURN c.cas_number AS cas_number, c.hybrid_embedding AS embedding"
    ),
    "NAICSEmbeddings": (
        "MATCH (s:Sector) WHERE s.naics_title_embedding IS NOT NULL "
        "RETURN s.naics_code AS cas_number, s.naics_title_embedding AS embedding"
    ),
}

_FETCH_NAMES = "MATCH (c:Chemical) RETURN c.cas_number AS cas_number, c.name AS name"

_FUNCTIONAL_CLASS_KEYWORDS: Dict[str, List[str]] = {
    "VOC": ["benzene", "toluene", "xylene", "styrene", "ethylbenzene", "hexane", "methanol", "acetone"],
    "heavy_metal": ["lead", "mercury", "cadmium", "chromium", "arsenic", "nickel", "zinc", "copper", "manganese"],
    "dioxin": ["dioxin", "furan", "tcdd", "pcdd", "pcdf"],
    "solvent": ["chloroform", "methylene", "perchloroethylene", "trichloroethylene", "acetonitrile"],
    "acid": ["hydrochloric", "sulfuric", "nitric", "phosphoric", "acetic"],
}


def _classify_chemical(name: str) -> str:
    name_lower = name.lower()
    for cls, keywords in _FUNCTIONAL_CLASS_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return cls
    return "other"


@dataclass
class EmbeddingMetricsResult:
    """Intrinsic quality metrics for one embedding type.

    Attributes:
        embedding_type (str): Name of the evaluated embedding type.
        silhouette (float): Silhouette score by functional class (higher is better).
        davies_bouldin (float): Davies-Bouldin index by functional class (lower is better).
        spearman_rho (float): Spearman ρ vs Tanimoto distance (structural anchor).
        spearman_pvalue (float): Two-sided p-value for the structural Spearman.
        spearman_rho_operational (float): Spearman ρ vs Jaccard Activity Similarity distance.
        spearman_pvalue_operational (float): Two-sided p-value for the operational Spearman.
        n_chemicals (int): Number of chemicals included.
        label_coverage (float): Fraction of chemicals with a functional class label.
        notes (List[str]): Human-readable reasons any metric is NaN/unreliable.
        projection (Optional[np.ndarray]): (N, 2) t-SNE coordinates, if requested.
        cas_numbers (List[str]): CAS numbers in embedding order.
        labels (List[str]): Functional class label per chemical, same order as cas_numbers.
    """

    embedding_type: str
    silhouette: float = float("nan")
    davies_bouldin: float = float("nan")
    spearman_rho: float = float("nan")
    spearman_pvalue: float = float("nan")
    spearman_rho_operational: float = float("nan")
    spearman_pvalue_operational: float = float("nan")
    spearman_rho_family: float = float("nan")
    spearman_pvalue_family: float = float("nan")
    n_chemicals: int = 0
    label_coverage: float = 0.0
    notes: List[str] = field(default_factory=list)
    projection: Optional[np.ndarray] = field(default=None, repr=False)
    cas_numbers: List[str] = field(default_factory=list, repr=False)
    labels: List[str] = field(default_factory=list, repr=False)


class EmbeddingMetrics:
    """Computes intrinsic quality metrics for embeddings stored in Neo4j."""

    def __init__(self, anchor_builder: Optional[AnchorBuilder] = None) -> None:
        self._connector = Neo4jConnector()
        self._anchor_builder = anchor_builder or AnchorBuilder()

    def _fetch_embeddings(self, embedding_type: str) -> List[Dict[str, Any]]:
        query = _EMBEDDING_QUERIES.get(embedding_type)
        if query is None:
            raise ValueError(f"Unknown embedding type: '{embedding_type}'")
        with self._connector.session() as session:
            return [dict(r) for r in session.run(query)]  # type: ignore[arg-type]

    def _fetch_names(self) -> Dict[str, str]:
        with self._connector.session() as session:
            return {r["cas_number"]: r["name"] for r in session.run(_FETCH_NAMES)}  # type: ignore[arg-type]

    def _cluster_metrics(self, embeddings: np.ndarray, labels: List[str]) -> Tuple[float, float, str]:
        label_set = list(set(labels))
        if len(label_set) < 2:
            return float("nan"), float("nan"), f"clustering: <2 functional classes ({len(label_set)})"
        label_ints = [label_set.index(lbl) for lbl in labels]
        sil = float(silhouette_score(embeddings, label_ints))
        db = float(davies_bouldin_score(embeddings, label_ints))
        return sil, db, ""

    def _spearman_against_matrix(
        self,
        cas_numbers: List[str],
        embeddings: np.ndarray,
        anchor_index: Dict[str, int],
        sim_matrix: np.ndarray,
    ) -> Tuple[float, float, str]:
        """Compute Spearman ρ between embedding cosine distance and anchor dissimilarity.

        Args:
            cas_numbers (List[str]): CAS numbers matching embedding rows.
            embeddings (np.ndarray): (M, D) embedding matrix.
            anchor_index (Dict[str, int]): Maps CAS number → row in sim_matrix.
            sim_matrix (np.ndarray): (N, N) pairwise similarity (higher = more similar).

        Returns:
            Tuple[float, float, str]: (Spearman ρ, two-sided p-value, reason note if NaN).
        """
        shared = [cas for cas in cas_numbers if cas in anchor_index]
        if len(shared) < 10:
            logger.warning(f"Only {len(shared)} shared chemicals; Spearman ρ unreliable")
            return float("nan"), float("nan"), f"only {len(shared)} shared chemicals (<10)"
        idx_emb = [cas_numbers.index(cas) for cas in shared]
        idx_anc = [anchor_index[cas] for cas in shared]
        sub_emb = np.asarray(normalize(embeddings[idx_emb]), dtype=np.float32)
        emb_dist = (1.0 - sub_emb @ sub_emb.T).flatten()
        anchor_dist = (1.0 - sim_matrix[np.ix_(idx_anc, idx_anc)]).flatten()
        if np.std(emb_dist) < 1e-8 or np.std(anchor_dist) < 1e-8:
            logger.warning("Near-zero variance in distance array — Spearman undefined, returning NaN")
            return float("nan"), float("nan"), "near-zero distance variance"
        sp = spearmanr(emb_dist, anchor_dist)
        return float(sp[0]), float(sp[1]), ""  # type: ignore[arg-type]

    def _build_result(
        self,
        embedding_type: str,
        cas_numbers: List[str],
        embeddings: np.ndarray,
        labels: List[str],
        structural: StructuralAnchors,
        operational: Optional[OperationalAnchors],
        project: bool,
    ) -> EmbeddingMetricsResult:
        covered = sum(1 for lbl in labels if lbl != "other")
        result = EmbeddingMetricsResult(
            embedding_type=embedding_type,
            n_chemicals=len(cas_numbers),
            label_coverage=covered / len(labels) if labels else 0.0,
            cas_numbers=cas_numbers,
            labels=labels,
        )
        result.silhouette, result.davies_bouldin, cluster_note = self._cluster_metrics(embeddings, labels)
        result.spearman_rho, result.spearman_pvalue, struct_note = self._spearman_against_matrix(
            cas_numbers, embeddings, structural.index, structural.similarity_matrix
        )
        notes = [n for n in (cluster_note, f"struct ρ: {struct_note}" if struct_note else "") if n]
        if operational is not None:
            result.spearman_rho_operational, result.spearman_pvalue_operational, op_note = self._spearman_against_matrix(
                cas_numbers, embeddings, operational.index, operational.jaccard_matrix
            )
            result.spearman_rho_family, result.spearman_pvalue_family, fam_note = self._spearman_against_matrix(
                cas_numbers, embeddings, operational.index, operational.jaccard_family_matrix
            )
            notes += [n for n in (f"oper ρ: {op_note}" if op_note else "", f"family ρ: {fam_note}" if fam_note else "") if n]
        result.notes = notes
        if project:
            perplexity = min(30, max(2, len(cas_numbers) - 1))
            result.projection = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(embeddings)
        logger.info(
            f"[{embedding_type}] silhouette={result.silhouette:.4f}, "
            f"DB={result.davies_bouldin:.4f}, "
            f"ρ_struct={result.spearman_rho:.4f}, "
            f"ρ_oper={result.spearman_rho_operational:.4f}, "
            f"ρ_family={result.spearman_rho_family:.4f}"
        )
        return result

    def compute(
        self,
        embedding_type: str,
        structural: Optional[StructuralAnchors] = None,
        operational: Optional[OperationalAnchors] = None,
        project: bool = False,
    ) -> EmbeddingMetricsResult:
        """Compute all intrinsic metrics for one stored embedding type.

        Args:
            embedding_type (str): One of MorganFingerprint, MolDeBERTa, Hybrid,
                or NAICSEmbeddings.
            structural (Optional[StructuralAnchors]): Pre-built Tanimoto anchors.
                Built automatically if not provided.
            operational (Optional[OperationalAnchors]): Pre-built operational anchors
                for Jaccard-based Spearman ρ.  Skipped if not provided.
            project (bool): Whether to compute a 2-D t-SNE projection.

        Returns:
            EmbeddingMetricsResult: Silhouette, Davies-Bouldin, both Spearman ρ values,
                and metadata.
        """
        records = self._fetch_embeddings(embedding_type)
        if not records:
            logger.warning(f"No embeddings found for type '{embedding_type}'")
            return EmbeddingMetricsResult(embedding_type=embedding_type)
        cas_numbers = [r["cas_number"] for r in records]
        embeddings = np.array([r["embedding"] for r in records], dtype=np.float32)
        names = self._fetch_names()
        labels = [_classify_chemical(names.get(cas, "")) for cas in cas_numbers]
        if structural is None:
            structural = self._anchor_builder.build_structural_anchors()
        return self._build_result(embedding_type, cas_numbers, embeddings, labels, structural, operational, project)
