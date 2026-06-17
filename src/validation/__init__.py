# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Evaluation modules for embedding quality, retrieval ranking, and end-to-end RAG.

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

from src.validation.anchor_builder import (
    ActivityProfiles,
    AnchorBuilder,
    OperationalAnchors,
    StructuralAnchors,
)
from src.validation.embedding_metrics import EmbeddingMetrics, EmbeddingMetricsResult
from src.validation.ragas_evaluator import RAGASEvaluator
from src.validation.recommendation_metrics import (
    RecommendationCoverageResult,
    RecommendationMetrics,
)
from src.validation.retrieval_metrics import (
    ActivityCoverageMetrics,
    ActivityCoverageResult,
    RetrievalMetrics,
    RetrievalMetricsResult,
    SectorCoverageResult,
    SignificanceTestResult,
)

__all__ = [
    "ActivityProfiles",
    "AnchorBuilder",
    "OperationalAnchors",
    "StructuralAnchors",
    "EmbeddingMetrics",
    "EmbeddingMetricsResult",
    "RAGASEvaluator",
    "RecommendationCoverageResult",
    "RecommendationMetrics",
    "ActivityCoverageMetrics",
    "ActivityCoverageResult",
    "RetrievalMetrics",
    "RetrievalMetricsResult",
    "SectorCoverageResult",
    "SignificanceTestResult",
]
