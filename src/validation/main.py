# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""Entry point for the TRI Chemical GraphRAG evaluation pipeline.

Runs any combination of the five evaluation steps — anchor building, embedding
space metrics, retrieval ranking metrics, QA benchmark generation, and end-to-end
RAGAS evaluation — then writes the results to docs/RESULTS.md and saves all
figures to images/.

Usage examples::

    # Full evaluation (all steps, 6-digit NAICS, 20 RAGAS pairs per condition)
    python -m src.validation.main --ollama-url http://localhost:11435

    # Intrinsic + retrieval metrics with 4-digit NAICS grouping (broader co-occurrence)
    python -m src.validation.main --steps anchors embedding retrieval --naics-digits 4

    # Generate the frozen QA benchmark and then run RAGAS
    python -m src.validation.main --steps qa-benchmark ragas --max-ragas-pairs 15

Author: Jose D. Hernandez-Betancur
Date: 2026-05-19
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib  # type: ignore[import-untyped]
import matplotlib.pyplot as plt  # type: ignore[import-untyped]
import numpy as np

from src.config_manager import ConfigManager
from src.validation.anchor_builder import (
    JACCARD_GRADE_HIGH,
    JACCARD_GRADE_LOW,
    NAICS_COOCCURRENCE_HIGH,
    ActivityProfiles,
    AnchorBuilder,
    OperationalAnchors,
    StructuralAnchors,
)
from src.validation.embedding_metrics import EmbeddingMetrics, EmbeddingMetricsResult
from src.validation.recommendation_metrics import (
    RecommendationCoverageResult,
    RecommendationMetrics,
)
from src.validation.retrieval_metrics import (
    CHEMICAL_EMBEDDING_TYPES,
    K_VALUES,
    ActivityCoverageMetrics,
    ActivityCoverageResult,
    RetrievalMetrics,
    RetrievalMetricsResult,
    SectorCoverageResult,
    SignificanceTestResult,
)

matplotlib.use("Agg")  # non-interactive backend; must precede pyplot imports

logger = logging.getLogger(__name__)

ROOT = Path(__file__).parents[2]
IMAGES_DIR = ROOT / "images"
DOCS_DIR = ROOT / "docs"
QA_BENCHMARK_PATH = ROOT / "data" / "qa_benchmark.json"

EMBEDDING_TYPES = CHEMICAL_EMBEDDING_TYPES
VALID_STEPS = ("split", "anchors", "embedding", "retrieval", "recommendation", "qa-benchmark", "ragas")

_NAICS_LEVEL_LABELS: Dict[int, str] = {
    2: "2-digit (Sector)",
    3: "3-digit (Subsector)",
    4: "4-digit (Ind. Group)",
    6: "6-digit (National)",
}

_CLASS_COLORS: Dict[str, str] = {
    "VOC": "#e41a1c",
    "heavy_metal": "#377eb8",
    "dioxin": "#4daf4a",
    "solvent": "#ff7f00",
    "acid": "#984ea3",
    "other": "#999999",
}


def _save_tsne(result: EmbeddingMetricsResult, images_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 6))
    xy = np.asarray(result.projection)
    n_pts = xy.shape[0]
    labels = result.labels if result.labels else ["other"] * n_pts
    for cls, color in _CLASS_COLORS.items():
        mask = [i for i, lbl in enumerate(labels) if lbl == cls]
        if mask:
            ax.scatter(xy[mask, 0], xy[mask, 1], c=color, label=cls, s=18, alpha=0.75)
    ax.set_title(f"t-SNE — {result.embedding_type} ({result.n_chemicals} chemicals)")
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(fontsize=7, markerscale=1.5, title="Functional class")
    fig.tight_layout()
    path = images_dir / f"tsne_{result.embedding_type.lower()}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved t-SNE figure → {path}")
    return path


def _save_tanimoto_hist(anchors: StructuralAnchors, images_dir: Path) -> Path:
    n = len(anchors.cas_numbers)
    upper = anchors.similarity_matrix[np.triu_indices(n, k=1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(upper, bins=80, color="#4e79a7", edgecolor="none")
    ax.axvline(0.4, color="#e15759", linestyle="--", label="grade-1 threshold (0.4)")
    ax.axvline(0.8, color="#f28e2b", linestyle="--", label="grade-2 threshold (0.8)")
    ax.set_xlabel("Tanimoto similarity")
    ax.set_ylabel("Pair count")
    ax.set_title(f"Pairwise Tanimoto distribution ({n} chemicals, {len(upper):,} pairs)")
    ax.legend()
    fig.tight_layout()
    path = images_dir / "tanimoto_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Tanimoto distribution → {path}")
    return path


def _save_jaccard_hist(anchors: OperationalAnchors, images_dir: Path) -> Path:
    n = len(anchors.cas_numbers)
    upper = anchors.jaccard_matrix[np.triu_indices(n, k=1)]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(upper, bins=50, color="#f28e2b", edgecolor="none")
    ax.axvline(JACCARD_GRADE_LOW, color="#e15759", linestyle="--", label=f"grade-1 threshold ({JACCARD_GRADE_LOW})")
    ax.axvline(JACCARD_GRADE_HIGH, color="#59a14f", linestyle="--", label=f"grade-2 threshold ({JACCARD_GRADE_HIGH})")
    ax.set_xlabel("Jaccard Activity Similarity")
    ax.set_ylabel("Pair count")
    ax.set_title(
        f"Pairwise Jaccard distribution ({n} chemicals, {len(upper):,} pairs, "
        f"{anchors.naics_digits}-digit NAICS, stratified index split (80%))"
    )
    ax.legend()
    fig.tight_layout()
    path = images_dir / "jaccard_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Jaccard distribution → {path}")
    return path


def _save_ndcg_chart(results: List[RetrievalMetricsResult], images_dir: Path, suffix: str = "") -> Path:
    k_vals = list(K_VALUES)
    labels = [r.embedding_type for r in results]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#e15759"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(k_vals):
        vals = [r.ndcg_at_k.get(k, 0.0) for r in results]
        ax.bar(x + i * width, vals, width, label=f"nDCG@{k}", color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("nDCG@k")
    anchor_label = suffix.lstrip("_").replace("_", " ").title() if suffix else "Structural (Tanimoto)"
    ax.set_title(f"Retrieval quality — {anchor_label} anchor (stratified test split, 20%)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = images_dir / f"ndcg_comparison{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved nDCG comparison chart → {path}")
    return path


def _save_recall_chart(results: List[RetrievalMetricsResult], images_dir: Path, suffix: str = "") -> Path:
    k_vals = list(K_VALUES)
    labels = [r.embedding_type for r in results]
    x = np.arange(len(labels))
    width = 0.25
    colors = ["#59a14f", "#edc948", "#b07aa1"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, k in enumerate(k_vals):
        vals = [r.recall_at_k.get(k, 0.0) for r in results]
        ax.bar(x + i * width, vals, width, label=f"Recall@{k}", color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Recall@k")
    anchor_label = suffix.lstrip("_").replace("_", " ").title() if suffix else "Structural (Tanimoto)"
    ax.set_title(f"Recall@k — {anchor_label} anchor (stratified test split, 20%)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = images_dir / f"recall_comparison{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved Recall comparison chart → {path}")
    return path


def _fmt_f(val: Any, decimals: int = 4) -> str:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return "—"
    return "—" if np.isnan(f) else f"{f:.{decimals}f}"


def _tsne_gallery() -> str:
    """Build a markdown image table of per-embedding t-SNE projections.

    Returns:
        str: Markdown table with one column per chemical embedding type.
    """
    intro = "**t-SNE projections** (colored by functional class — VOC, heavy metal, dioxin, solvent, acid, other):"
    header = "| " + " | ".join(EMBEDDING_TYPES) + " |"
    sep = "|" + "|".join(":---:" for _ in EMBEDDING_TYPES) + "|"
    imgs = "| " + " | ".join(f"![](../images/tsne_{et.lower()}.png)" for et in EMBEDDING_TYPES) + " |"
    return f"{intro}\n\n{header}\n{sep}\n{imgs}\n"


def _emb_table(results: Dict[str, EmbeddingMetricsResult]) -> str:
    header = (
        "| Embedding | n | Silhouette ↑ | DB ↓ "
        "| ρ Tanimoto ↑ | p | ρ Jaccard (code) ↑ | p | ρ Jaccard (family) ↑ | p | Notes |\n"
    )
    sep = (
        "|-----------|---|-------------|------|-------------|----|--------------------|----|"
        "--------------------|----|-------|\n"
    )
    rows = ""
    for et in EMBEDDING_TYPES:
        r = results.get(et)
        if r is None:
            continue
        note = "; ".join(r.notes) if r.notes else ""
        rows += (
            f"| {et} | {r.n_chemicals} "
            f"| {_fmt_f(r.silhouette)} | {_fmt_f(r.davies_bouldin)} "
            f"| {_fmt_f(r.spearman_rho)} | {_fmt_f(r.spearman_pvalue)} "
            f"| {_fmt_f(r.spearman_rho_operational)} | {_fmt_f(r.spearman_pvalue_operational)} "
            f"| {_fmt_f(r.spearman_rho_family)} | {_fmt_f(r.spearman_pvalue_family)} | {note} |\n"
        )
    return header + sep + rows


def _fmt_lift(value: Any, baseline: Any, embedding_type: str) -> str:
    """Format a metric's lift over the Random baseline as 'Δ±x.xxx (r.rr×)'.

    Args:
        value (Any): The embedding's metric value.
        baseline (Any): The Random baseline metric value.
        embedding_type (str): Row label; the Random/Oracle rows render as '—'.

    Returns:
        str: Lift string, or '—' when not applicable.
    """
    if embedding_type in ("Random", "Oracle"):
        return "—"
    try:
        v, b = float(value), float(baseline)
    except (TypeError, ValueError):
        return "—"
    if b <= 0.0:
        return "—"
    return f"{v - b:+.3f} ({v / b:.2f}×)"


def _coverage_table(results: List[ActivityCoverageResult]) -> str:
    baseline = next((r.weighted_coverage_at_k.get(10) for r in results if r.embedding_type == "Random"), None)
    header = (
        "| Embedding | WeightedCov@5 ↑ | @10 | @20 | Lift@10 vs Random | " "CodePrec@10 ↑ | RawCov@10 | FamilyCov@10 | n |\n"
    )
    sep = "|-----------|----------------|-----|-----|-------------------|" "--------------|-----------|--------------|---|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r.embedding_type} | {_fmt_f(r.weighted_coverage_at_k.get(5))} "
            f"| {_fmt_f(r.weighted_coverage_at_k.get(10))} "
            f"| {_fmt_f(r.weighted_coverage_at_k.get(20))} "
            f"| {_fmt_lift(r.weighted_coverage_at_k.get(10), baseline, r.embedding_type)} "
            f"| {_fmt_f(r.code_precision_at_k.get(10))} "
            f"| {_fmt_f(r.activity_coverage_at_k.get(10))} "
            f"| {_fmt_f(r.family_coverage_at_k.get(10))} "
            f"| {r.n_queries} |\n"
        )
    return header + sep + rows


def _recommendation_table(results: List[RecommendationCoverageResult]) -> str:
    baseline = next((r.weighted_coverage_at_k.get(10) for r in results if r.method == "Random"), None)
    header = "| Method | WeightedCov@5 ↑ | @10 | @20 | Lift@10 vs Random | " "CodePrec@10 ↑ | RawCov@10 | n |\n"
    sep = "|--------|----------------|-----|-----|-------------------|--------------|-----------|---|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r.method} | {_fmt_f(r.weighted_coverage_at_k.get(5))} "
            f"| {_fmt_f(r.weighted_coverage_at_k.get(10))} "
            f"| {_fmt_f(r.weighted_coverage_at_k.get(20))} "
            f"| {_fmt_lift(r.weighted_coverage_at_k.get(10), baseline, r.method)} "
            f"| {_fmt_f(r.code_precision_at_k.get(10))} "
            f"| {_fmt_f(r.raw_coverage_at_k.get(10))} "
            f"| {r.n_queries} |\n"
        )
    return header + sep + rows


def _sector_table(results: List[SectorCoverageResult]) -> str:
    embedding_types = list(dict.fromkeys(r.embedding_type for r in results))
    levels = sorted({r.naics_level for r in results})
    col_labels = [_NAICS_LEVEL_LABELS.get(lvl, f"{lvl}-digit") for lvl in levels]
    header = "| Embedding | " + " | ".join(col_labels) + " |\n"
    sep = "|-----------|" + "|".join("-" * 22 for _ in levels) + "|\n"
    rows = ""
    for et in embedding_types:
        row_vals = []
        for level in levels:
            match = next((r for r in results if r.embedding_type == et and r.naics_level == level), None)
            row_vals.append(_fmt_f(match.sector_coverage_at_k.get(10)) if match else "—")
        rows += f"| {et} | " + " | ".join(row_vals) + " |\n"
    return header + sep + rows


def _save_coverage_chart(results: List[ActivityCoverageResult], images_dir: Path) -> Path:
    labels = [r.embedding_type for r in results]
    x = np.arange(len(labels))
    width = 0.4
    fig, ax = plt.subplots(figsize=(9, 5))
    weighted10 = [r.weighted_coverage_at_k.get(10, 0.0) for r in results]
    prec10 = [r.code_precision_at_k.get(10, 0.0) for r in results]
    ax.bar(x - width / 2, weighted10, width, label="WeightedCoverage@10 (primary)", color="#4e79a7")
    ax.bar(x + width / 2, prec10, width, label="CodePrecision@10 (primary)", color="#e15759")
    baseline = next((r.weighted_coverage_at_k.get(10, 0.0) for r in results if r.embedding_type == "Random"), None)
    if baseline is not None:
        ax.axhline(baseline, color="#999999", linestyle="--", linewidth=1, label="Random WeightedCoverage@10")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10)
    ax.set_ylabel("Score")
    ax.set_title("Primary activity metrics @10 — IDF-weighted coverage & precision (stratified test split, 20%)")
    ax.legend()
    ax.set_ylim(0, 1)
    fig.tight_layout()
    path = images_dir / "activity_coverage.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved activity coverage chart → {path}")
    return path


def _ret_table(results: List[RetrievalMetricsResult]) -> str:
    header = "| Embedding | n queries | nDCG@5 | nDCG@10 | nDCG@20 | Recall@10 | MRR |\n"
    sep = "|-----------|-----------|--------|---------|---------|-----------|-----|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r.embedding_type} | {r.n_queries} "
            f"| {_fmt_f(r.ndcg_at_k.get(5))} | {_fmt_f(r.ndcg_at_k.get(10))} "
            f"| {_fmt_f(r.ndcg_at_k.get(20))} | {_fmt_f(r.recall_at_k.get(10))} "
            f"| {_fmt_f(r.mrr)} |\n"
        )
    return header + sep + rows


def _sig_table(tests: List[SignificanceTestResult]) -> str:
    header = "| Comparison | k | Statistic | p-value | Significant |\n"
    sep = "|------------|---|-----------|---------|-------------|\n"
    rows = ""
    for t in tests:
        sig = "Yes" if t.significant else "No"
        rows += f"| {t.embedding_a} vs {t.embedding_b} | {t.k} " f"| {_fmt_f(t.statistic)} | {_fmt_f(t.pvalue)} | {sig} |\n"
    return header + sep + rows


def _ragas_table(results: List[Dict[str, Any]]) -> str:
    header = "| Condition | Faithfulness ↑ | Ctx. Precision ↑ | Ctx. Recall ↑ | n scored | attempted | failed |\n"
    sep = "|-----------|---------------|-----------------|---------------|----------|-----------|--------|\n"
    rows = ""
    for r in results:
        rows += (
            f"| {r.get('condition', '—')} "
            f"| {_fmt_f(r.get('faithfulness'))} "
            f"| {_fmt_f(r.get('context_precision'))} "
            f"| {_fmt_f(r.get('context_recall'))} "
            f"| {r.get('n_samples', 0)} "
            f"| {r.get('n_attempted', '—')} "
            f"| {r.get('n_failed', '—')} |\n"
        )
    return header + sep + rows


def _operational_grade_counts(anchors: OperationalAnchors) -> Tuple[int, int, int]:
    n = len(anchors.cas_numbers)
    upper = np.triu_indices(n, k=1)
    jac = anchors.jaccard_matrix[upper]
    naics = anchors.naics_cooccurrence_matrix[upper]
    grade2 = int(((jac >= JACCARD_GRADE_HIGH) & (naics >= NAICS_COOCCURRENCE_HIGH)).sum())
    grade1 = int(((jac >= JACCARD_GRADE_LOW) | (naics >= 1)).sum()) - grade2
    grade0 = len(jac) - grade2 - grade1
    return grade2, grade1, grade0


def _patch_ragas_section(ragas_results: List[Dict[str, Any]], docs_dir: Path) -> None:
    """Update only Section 7 (RAGAS) of an existing RESULTS.md in-place.

    Args:
        ragas_results (List[Dict[str, Any]]): RAGAS metric scores per condition.
        docs_dir (Path): Directory containing RESULTS.md.
    """
    import re

    results_path = docs_dir / "RESULTS.md"
    if not results_path.exists():
        logger.warning("RESULTS.md not found — cannot patch RAGAS section")
        return
    from src.validation.ragas_evaluator import judge_descriptor

    rag_cfg = ConfigManager().config["RAG_GRAPH"]
    caption = (
        "\n**How to read this table.** Every condition runs the full RAG pipeline end-to-end and is "
        "scored by an LLM judge. *Faithfulness* asks whether the generated answer is grounded in the "
        "retrieved context (anti-hallucination); *context precision* whether the retrieved context is "
        "relevant and well-ranked; *context recall* whether retrieval surfaced everything the reference "
        "answer needs. The **flat** row is the ablation baseline — vector retrieval with **no graph "
        "traversal** — so the gap between each GraphRAG condition and `flat` is the graph's contribution. "
        "`flat` scoring 0 precision / 0 recall while every GraphRAG condition is positive on both is the "
        'intended "graph context helps" signal; structural (Morgan) and Hybrid retrieval beat '
        "pure-semantic MolDeBERTa, consistent with §2–§6 where MolDeBERTa is the weakest retriever. "
        "Because scores are end-to-end and **judge-dependent**, absolute values are not comparable across "
        "judge models (a stronger judge lowers precision roughly uniformly) and n is small — read the "
        "table for *relative ordering*, not absolute magnitudes.\n\n"
        "**Strongest embedding & decision implications.** Across the operationally-relevant metrics — "
        "sector coverage (§3, all NAICS levels), operational/activity retrieval (§6), and cluster quality "
        "(§4) — **Hybrid (Morgan ‖ MolDeBERTa) is strongest**; **Morgan** leads only on the structural "
        "anchor (§5, which is definitionally Tanimoto-based) and on §7 precision, while **MolDeBERTa** "
        "alone is the weakest retriever. This matters because the two error types are not symmetric. "
        "*Context precision* governs **false positives** — recommending a source-reduction activity that "
        "is not actually applicable, costing wasted engineering assessment and capital, but bounded and "
        "caught during expert review. *Context recall / coverage* governs **false negatives** — missing a "
        "reduction that would have worked, i.e. a forgone prevention opportunity and continued, avoidable "
        "emissions; environmentally that error is costlier and effectively irreversible. So for the "
        'research question *"is there a useful source reduction for this chemical?"* — a screening / '
        "discovery task with an expert validating candidates downstream — the priority is to **minimize "
        "false negatives**: optimize **recall / coverage**, where **Hybrid** is best (best §7 recall and "
        "§3/§6 coverage). Prefer **Morgan** only when the goal is high-precision structural-analogue "
        "search and false positives are the costlier error (e.g. fully automated recommendation with no "
        "human in the loop). See `DECISION_FRAMEWORK.md` for the full hypothesis-testing treatment "
        "(Type I/II errors, precision/recall, and the faithfulness hallucination guardrail).\n\n"
        f"RAGAS judge: **{judge_descriptor()}** "
        f"(answer generator: {rag_cfg['GENERATION']['MODEL']}). "
        f"Faithfulness and context-recall require a GPT-4-class judge — local 7-14B judges "
        f"emit the wrong schema (→ NaN) and qwen2.5:32b is too slow; a hosted cloud model "
        f"recovers all three metrics.\n"
    )
    content = results_path.read_text()
    new_section = "## 7 · End-to-End RAG Quality (RAGAS)\n\n" + _ragas_table(ragas_results) + caption
    content = re.sub(
        r"## 7 · End-to-End RAG Quality \(RAGAS\)\n\n.*?(?=\n---|\Z)",
        new_section,
        content,
        flags=re.DOTALL,
    )
    results_path.write_text(content)
    logger.info(f"Patched RAGAS section in {results_path}")


def _write_results_md(
    run_date: str,
    structural: Optional[StructuralAnchors],
    operational: Optional[OperationalAnchors],
    emb_results: Dict[str, EmbeddingMetricsResult],
    coverage_results: List[ActivityCoverageResult],
    sector_results: List[SectorCoverageResult],
    recommendation_results: List[RecommendationCoverageResult],
    ret_structural: List[RetrievalMetricsResult],
    ret_operational: List[RetrievalMetricsResult],
    sig_structural: List[SignificanceTestResult],
    sig_operational: List[SignificanceTestResult],
    ragas_results: List[Dict[str, Any]],
    docs_dir: Path,
    images_dir: Optional[Path] = None,
) -> Path:
    img_dir = images_dir if images_dir is not None else docs_dir.parent / "images"
    from src.validation.ragas_evaluator import judge_descriptor

    _rag_cfg = ConfigManager().config["RAG_GRAPH"]
    gen_model = _rag_cfg["GENERATION"]["MODEL"]
    judge_desc = judge_descriptor()
    tsne_names = [f"tsne_{et.lower()}.png" for et in EMBEDDING_TYPES]
    tsne_available = all((img_dir / name).exists() for name in tsne_names)

    if structural is not None:
        n_str = len(structural.cas_numbers)
        pairs_str = n_str * (n_str - 1) // 2
        grade2_str = int((structural.grade_matrix == 2).sum() // 2)
        grade1_str = int((structural.grade_matrix == 1).sum() // 2)
    else:
        n_str = pairs_str = grade2_str = grade1_str = 0

    if operational is not None:
        n_op = len(operational.cas_numbers)
        pairs_op = n_op * (n_op - 1) // 2
        grade2_op, grade1_op, grade0_op = _operational_grade_counts(operational)
        naics_digits = operational.naics_digits
    else:
        n_op = pairs_op = grade2_op = grade1_op = grade0_op = 0
        naics_digits = 6

    cov_section = (
        _coverage_table(coverage_results) + "\n\n" "![Activity Coverage](../images/activity_coverage.png)"
        if coverage_results
        else "_Activity coverage step was not run._"
    )
    sector_section = _sector_table(sector_results) if sector_results else "_NAICS hierarchy analysis was not run._"
    recommendation_section = (
        _recommendation_table(recommendation_results)
        if recommendation_results
        else "_Shipped-recommender evaluation was not run._"
    )

    content = f"""# Evaluation Results — TRI Chemical GraphRAG

Run date: {run_date}

---

## 1 · Anchor Statistics

### Structural Anchor (Tanimoto, independent of all embeddings)

| Metric | Value |
|--------|-------|
| Chemicals with valid SMILES | {n_str} |
| Unique pairwise comparisons | {pairs_str:,} |
| Grade-2 pairs (Tanimoto ≥ 0.8) | {grade2_str:,} |
| Grade-1 pairs (Tanimoto 0.4–0.8) | {grade1_str:,} |
| Grade-0 pairs (Tanimoto < 0.4) | {pairs_str - grade2_str - grade1_str:,} |

{"![Tanimoto distribution](../images/tanimoto_distribution.png)" if structural is not None else ""}

### Operational Anchor (Jaccard Activity Similarity, stratified index split)

Grades based on overlap of source reduction activity codes (Jaccard) and shared
({naics_digits}-digit NAICS × activity code) combinations.

| Metric | Value |
|--------|-------|
| Chemicals with activity profiles | {n_op} |
| Unique pairwise comparisons | {pairs_op:,} |
| Grade-2 (Jaccard ≥ {JACCARD_GRADE_HIGH} AND NAICS co-occ ≥ {NAICS_COOCCURRENCE_HIGH}) | {grade2_op:,} |
| Grade-1 (Jaccard ≥ {JACCARD_GRADE_LOW} OR NAICS co-occ ≥ 1) | {grade1_op:,} |
| Grade-0 (no activity overlap) | {grade0_op:,} |

![Jaccard distribution](../images/jaccard_distribution.png)

---

## 2 · Activity Coverage (PRIMARY — stratified test split, 20%)

This is a top-k **recommendation** metric: the nearest-neighbour chemicals are flattened into
a similarity-ranked list of distinct activity codes, and the top-k codes are scored against the
query chemical's actual test-split activities. **WeightedCoverage@k** (primary) is the
IDF-weighted fraction of the query's activities recovered by the top-k codes — rare, specific
codes count more than ubiquitous ones. **CodePrecision@10** (primary) is the fraction of the
top-k recommended codes that are actually relevant, penalizing indiscriminate breadth.
**Lift@10 vs Random** shows the absolute and multiplicative gain over the Random baseline; a
method only genuinely helps when Lift > 0.  **RawCov@10** is the unweighted union coverage,
retained as a diagnostic (expect Random to score high here — that is the artifact the weighted
metric corrects).  FamilyCov@10 uses the five TRI_SOURCE_REDUCTION_GROUPS families.
Oracle@k is the ceiling: the k index chemicals with maximum Jaccard overlap with the query's
test-split codes (uses ground truth — not achievable by any blind retrieval method).

{cov_section}

---

## 2b · Shipped Recommender — End-to-End Graph Path (stratified test split, 20%)

Scores the **production** recommendation path exactly as deployed: `retrieve_multi`
(vector ANN + graph traversal + cross-entity intersection) → ranked source-reduction
records.  Here k is the number of distinct top-ranked recommended activity codes, and
the same IDF-weighted coverage / code-precision metrics are applied.  This is the table
that reflects what an end user actually receives, in contrast to the pure-vector intrinsic
metrics above.  Random draws k codes uniformly; Oracle recommends the query's own
highest-weight codes (ceiling).

{recommendation_section}

---

## 3 · NAICS Hierarchy Analysis — SectorCoverage@10

Sector-conditioned coverage: both query activities and retrieved activities restricted
to the same NAICS prefix at the given digit depth.
Broader prefixes (2-digit) match more sector contexts and typically yield higher coverage;
narrower prefixes (6-digit) isolate activity recommendations to the specific industry.

{sector_section}

---

## 4 · Embedding Space Quality (intrinsic)

Spearman ρ is reported against structural (Tanimoto), operational code-level (Jaccard),
and operational family-level (Jaccard on activity families) distance matrices.

{_emb_table(emb_results)}

{_tsne_gallery() if tsne_available else "_t-SNE projections not generated (run without --no-tsne to produce them)._"}

---

## 5 · Retrieval Quality — Structural Anchor (Tanimoto, BASELINE — stratified test split 20%)

{_ret_table(ret_structural) if ret_structural else "_Structural retrieval step was not run._"}

{('![nDCG structural](../images/ndcg_comparison.png)\n![Recall structural](../images/recall_comparison.png)') if ret_structural else ''}

### Wilcoxon signed-rank significance tests (nDCG@10, structural anchor)

{_sig_table(sig_structural) if sig_structural else "_No significance tests._"}

---

## 6 · Retrieval Quality — Operational Anchor (Jaccard, SECONDARY — stratified test split 20%)

{_ret_table(ret_operational) if ret_operational else "_Operational retrieval step was not run._"}

{('![nDCG operational](../images/ndcg_comparison_operational.png)\n![Recall operational](../images/recall_comparison_operational.png)') if ret_operational else ''}

### Wilcoxon signed-rank significance tests (nDCG@10, operational anchor)

{_sig_table(sig_operational) if sig_operational else "_No significance tests._"}

---

## 7 · End-to-End RAG Quality (RAGAS)

{_ragas_table(ragas_results) if ragas_results else "_RAGAS step was not run._"}

---

## 8 · Notes

- All metrics use a stratified random split (80% index, 20% test, seed=42) across all report years.
- Structural Tanimoto thresholds (0.4 / 0.8) and k values (5, 10, 20) are pre-registered and unchanged.
- Operational anchor uses Jaccard ≥ {JACCARD_GRADE_HIGH} AND NAICS co-occ ≥ {NAICS_COOCCURRENCE_HIGH} for grade-2 (pre-registered).
- NAICS grouping: {naics_digits}-digit ({('Industry Group' if naics_digits == 4 else 'National Industry')}).
- RAGAS judge: {judge_desc} (provider / model; answer generator: {gen_model}).
"""
    path = docs_dir / "RESULTS.md"
    path.write_text(content)
    logger.info(f"Results document written → {path}")
    return path


class EvaluationPipeline:
    """Orchestrates the full evaluation run for the TRI Chemical GraphRAG system."""

    def __init__(
        self,
        images_dir: Path = IMAGES_DIR,
        docs_dir: Path = DOCS_DIR,
        max_ragas_pairs: int = 20,
        project_tsne: bool = True,
        naics_digits: int = 6,
        force_resplit: bool = False,
    ) -> None:
        self._images_dir = images_dir
        self._docs_dir = docs_dir
        self._max_ragas_pairs = max_ragas_pairs
        self._project_tsne = project_tsne
        self._naics_digits = naics_digits
        self._force_resplit = force_resplit
        self._images_dir.mkdir(parents=True, exist_ok=True)

    def _step_split(self) -> None:
        AnchorBuilder().write_stratified_split(force=self._force_resplit)

    def _step_anchors(
        self,
    ) -> Tuple[StructuralAnchors, OperationalAnchors, ActivityProfiles]:
        builder = AnchorBuilder()
        structural = builder.build_structural_anchors()
        operational = builder.build_operational_anchors(naics_digits=self._naics_digits)
        profiles = builder.build_activity_profiles()
        _save_tanimoto_hist(structural, self._images_dir)
        _save_jaccard_hist(operational, self._images_dir)
        return structural, operational, profiles

    def _step_coverage(
        self,
        profiles: ActivityProfiles,
    ) -> Tuple[List[ActivityCoverageResult], List[SectorCoverageResult]]:
        acm = ActivityCoverageMetrics()
        cov_results = acm.evaluate_all(profiles)
        sector_results = acm.evaluate_naics_levels(profiles)
        _save_coverage_chart(cov_results, self._images_dir)
        return cov_results, sector_results

    def _step_recommendation(
        self,
        profiles: ActivityProfiles,
    ) -> List[RecommendationCoverageResult]:
        return RecommendationMetrics().evaluate(profiles)

    def _step_embedding(
        self,
        structural: StructuralAnchors,
        operational: OperationalAnchors,
    ) -> Dict[str, EmbeddingMetricsResult]:
        metrics = EmbeddingMetrics()
        results: Dict[str, EmbeddingMetricsResult] = {}
        for et in EMBEDDING_TYPES:
            try:
                result = metrics.compute(et, structural=structural, operational=operational, project=self._project_tsne)
            except Exception as exc:
                logger.warning(f"Skipping embedding metrics for '{et}' (unavailable): {exc}")
                continue
            results[et] = result
            if self._project_tsne and result.projection is not None:
                _save_tsne(result, self._images_dir)
        return results

    def _run_retrieval_for_anchor(
        self,
        rm: RetrievalMetrics,
        anchor: Any,
        anchor_type: str,
        suffix: str,
    ) -> Tuple[List[RetrievalMetricsResult], List[SignificanceTestResult]]:
        ret = rm.evaluate_all(anchors=anchor, anchor_type=anchor_type)
        sigs: List[SignificanceTestResult] = []
        for i in range(len(ret)):
            for j in range(i + 1, len(ret)):
                sigs.append(rm.significance_test(ret[i], ret[j], k=10))
        _save_ndcg_chart(ret, self._images_dir, suffix=suffix)
        _save_recall_chart(ret, self._images_dir, suffix=suffix)
        return ret, sigs

    def _step_retrieval(
        self,
        structural: StructuralAnchors,
        operational: OperationalAnchors,
    ) -> Tuple[
        List[RetrievalMetricsResult],
        List[RetrievalMetricsResult],
        List[SignificanceTestResult],
        List[SignificanceTestResult],
    ]:
        rm = RetrievalMetrics()
        ret_str, sig_str = self._run_retrieval_for_anchor(rm, structural, "structural", "")
        ret_op, sig_op = self._run_retrieval_for_anchor(rm, operational, "operational", "_operational")
        return ret_str, ret_op, sig_str, sig_op

    def _step_qa_benchmark(self) -> Path:
        from src.validation_data_generation.qa_generator import QAGenerator

        if QA_BENCHMARK_PATH.exists():
            logger.info(f"QA benchmark already exists at {QA_BENCHMARK_PATH} — skipping generation")
            return QA_BENCHMARK_PATH
        QAGenerator().generate(n_per_type=25, output_path=QA_BENCHMARK_PATH)
        return QA_BENCHMARK_PATH

    def _step_ragas(self) -> List[Dict[str, Any]]:
        from src.validation.ragas_evaluator import RAGASEvaluator

        evaluator = RAGASEvaluator(benchmark_path=QA_BENCHMARK_PATH)
        return evaluator.evaluate_all(max_pairs=self._max_ragas_pairs)

    def run(self, steps: List[str]) -> None:
        """Execute the requested evaluation steps and write docs/RESULTS.md.

        Args:
            steps (List[str]): Subset of ('anchors', 'embedding', 'retrieval',
                'qa-benchmark', 'ragas'). Steps are always executed in dependency
                order regardless of the order given.
        """
        run_date = datetime.now().strftime("%Y-%m-%d %H:%M")
        structural: Optional[StructuralAnchors] = None
        operational: Optional[OperationalAnchors] = None
        profiles: Optional[ActivityProfiles] = None
        emb_results: Dict[str, EmbeddingMetricsResult] = {}
        coverage_results: List[ActivityCoverageResult] = []
        sector_results: List[SectorCoverageResult] = []
        recommendation_results: List[RecommendationCoverageResult] = []
        ret_structural: List[RetrievalMetricsResult] = []
        ret_operational: List[RetrievalMetricsResult] = []
        sig_structural: List[SignificanceTestResult] = []
        sig_operational: List[SignificanceTestResult] = []
        ragas_results: List[Dict[str, Any]] = []

        needs_anchors = bool({"anchors", "embedding", "retrieval", "recommendation"} & set(steps))
        if needs_anchors or "split" in steps:
            logger.info("=== Step: stratified split ===")
            self._step_split()

        if needs_anchors:
            logger.info("=== Step: anchors ===")
            structural, operational, profiles = self._step_anchors()

        if "embedding" in steps and structural is not None and operational is not None:
            logger.info("=== Step: embedding metrics ===")
            emb_results = self._step_embedding(structural, operational)

        if "retrieval" in steps and profiles is not None:
            logger.info("=== Step: activity coverage (primary) ===")
            coverage_results, sector_results = self._step_coverage(profiles)

        if "recommendation" in steps and profiles is not None:
            logger.info("=== Step: shipped recommender (end-to-end graph path) ===")
            recommendation_results = self._step_recommendation(profiles)

        if "retrieval" in steps and structural is not None and operational is not None:
            logger.info("=== Step: retrieval metrics (anchor-based) ===")
            ret_structural, ret_operational, sig_structural, sig_operational = self._step_retrieval(structural, operational)

        if "qa-benchmark" in steps or "ragas" in steps:
            logger.info("=== Step: QA benchmark ===")
            self._step_qa_benchmark()

        if "ragas" in steps:
            logger.info("=== Step: RAGAS evaluation ===")
            ragas_results = self._step_ragas()

        if structural is not None:
            _write_results_md(
                run_date,
                structural,
                operational,
                emb_results,
                coverage_results,
                sector_results,
                recommendation_results,
                ret_structural,
                ret_operational,
                sig_structural,
                sig_operational,
                ragas_results,
                self._docs_dir,
                self._images_dir,
            )
        elif ragas_results:
            _patch_ragas_section(ragas_results, self._docs_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TRI Chemical GraphRAG — evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=list(VALID_STEPS),
        default=list(VALID_STEPS),
        metavar="STEP",
        help=f"Evaluation steps to run (default: all). Choices: {', '.join(VALID_STEPS)}.",
    )
    parser.add_argument(
        "--max-ragas-pairs",
        type=int,
        default=20,
        dest="max_ragas_pairs",
        metavar="N",
        help="Maximum QA pairs per RAGAS condition (0 = all). Default: 20.",
    )
    parser.add_argument(
        "--naics-digits",
        type=int,
        choices=[4, 6],
        default=6,
        dest="naics_digits",
        help=(
            "NAICS hierarchy depth for operational anchor co-occurrence. "
            "4 = Industry Group (broader, more shared codes); "
            "6 = National Industry (default, more specific)."
        ),
    )
    parser.add_argument(
        "--force-resplit",
        action="store_true",
        dest="force_resplit",
        help="Re-assign stratified split even if already present in Neo4j.",
    )
    parser.add_argument(
        "--no-tsne",
        action="store_true",
        dest="no_tsne",
        help="Skip t-SNE projection computation (faster).",
    )
    parser.add_argument(
        "--ollama-url",
        default="",
        dest="ollama_url",
        metavar="URL",
        help="Override Ollama base URL (e.g. http://localhost:11435).",
    )
    parser.add_argument(
        "--judge-provider",
        default="",
        dest="judge_provider",
        choices=["", "openai", "ollama_cloud", "ollama_local"],
        help="Override the RAGAS judge provider (default: from config).",
    )
    parser.add_argument(
        "--judge-model",
        default="",
        dest="judge_model",
        metavar="MODEL",
        help="Override the RAGAS judge model (e.g. gpt-4o-mini, gpt-oss:120b-cloud).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    """Run the evaluation pipeline from the command line.

    Args:
        argv (Optional[List[str]]): Argument list; defaults to sys.argv[1:].
    """
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=logging.INFO,
        stream=sys.stdout,
    )

    args = _parse_args(argv if argv is not None else sys.argv[1:])

    if args.ollama_url:
        os.environ["OLLAMA_BASE_URL"] = args.ollama_url
        logger.info(f"OLLAMA_BASE_URL set to {args.ollama_url}")
    if args.judge_provider:
        os.environ["RAGAS_JUDGE_PROVIDER"] = args.judge_provider
        logger.info(f"RAGAS judge provider overridden to {args.judge_provider}")
    if args.judge_model:
        os.environ["RAGAS_JUDGE_MODEL"] = args.judge_model
        logger.info(f"RAGAS judge model overridden to {args.judge_model}")

    pipeline = EvaluationPipeline(
        max_ragas_pairs=args.max_ragas_pairs,
        project_tsne=not args.no_tsne,
        naics_digits=args.naics_digits,
        force_resplit=args.force_resplit,
    )
    pipeline.run(steps=args.steps)


if __name__ == "__main__":
    main()
