# TRI Chemical GraphRAG — Research Evaluation Plan

This document records the evaluation strategy for the manuscript. It is meant to be read
after the system implementation is complete so that the evaluation phase has a clear,
pre-committed design.

---

## System Under Evaluation

A GraphRAG pipeline built on:
- **Knowledge graph**: Neo4j, covering TRI reports 2014–2024 (facilities with source reduction activity)
- **Orchestration**: LangGraph + LangChain
- **Embeddings**: three parallel chemical embedding types (see below)
- **Generator**: Claude (Anthropic API)

### Embedding Types

| Name | Model | Dims | Target |
|------|-------|-----:|--------|
| Morgan Fingerprint | RDKit (radius 2, 2048 bits) | 2048 | Structural similarity |
| MolDeBERTa | SaeedLab/MolDeBERTa-base-123M-mtr | 768 | Semantic / functional similarity |
| Hybrid | Morgan ‖ MolDeBERTa concatenated | 1792 | Combined |
| NAICS | sentence-transformers/all-MiniLM-L6-v2 | 384 | Sector similarity |

---

## Ground-Truth Anchor Data

### Why anchors are non-trivial here

The same chemical can have multiple valid retrieval targets (structural analogues, functional
equivalents, sector co-reporters). Binary relevance labels would penalise valid retrievals.
The evaluation therefore uses **graded relevance** throughout.

### Anchor construction strategy

**A. Structural relevance (external, independent of embeddings)**
Compute pairwise Tanimoto similarity over all 529 chemicals using RDKit Morgan fingerprints.
Define graded relevance:

| Tanimoto | Relevance grade |
|----------|----------------|
| ≥ 0.8 | 2 — highly relevant |
| 0.4–0.8 | 1 — partially relevant |
| < 0.4 | 0 — not relevant |

This is the primary external reference: it is derived from chemical structure, not from any
embedding, so it can fairly judge all three embedding types.

**B. Behavioural relevance (internal, derived from the KG)**
Two chemicals are behaviourally related if the TRI reporting record links them:
- Co-reported by ≥ N facilities in the same NAICS sector → sector-level relevance
- Share the same source reduction code → activity-level relevance

These capture domain relationships that structure alone does not encode, and make the
evaluation specific to the TRI context.

**C. Synthetic QA pairs (for end-to-end RAG evaluation)**
Generate question–answer pairs from known graph facts (e.g. "Which chemicals in the
automotive sector had source reductions ≥ 50 % in 2020?"). Answers are deterministic
from the graph. Use these as the RAGAS benchmark.

The QA pairs are generated once, manually reviewed, and frozen before any system tuning.

### Test split

| Split | Years | Purpose |
|-------|-------|---------|
| Development | 2014–2020 | System building, embedding tuning |
| Test | 2021–2024 | All reported metrics |

Queries are held out, not chemicals. Chemicals from test years remain in the index so
that the evaluation tests ranking quality, not index coverage.

---

## Evaluation Metrics

### 1. Embedding space quality (intrinsic)

Computed per embedding type on the full 529-chemical set.

| Metric | Library | Notes |
|--------|---------|-------|
| Silhouette score | scikit-learn | Group chemicals by functional class (VOC, heavy metal, dioxin, …) |
| Davies-Bouldin index | scikit-learn | Lower is better |
| Tanimoto rank correlation | scipy | Spearman ρ between embedding distance and Tanimoto distance |
| UMAP / t-SNE projection | umap-learn | Qualitative figure for manuscript |

### 2. Retrieval quality (extrinsic)

Evaluated against the structural (Tanimoto) anchor using graded relevance.

| Metric | @k values | Notes |
|--------|-----------|-------|
| nDCG@k | 5, 10, 20 | Primary metric; handles graded relevance |
| Recall@k | 5, 10, 20 | Did any grade-2 item appear in top-k? |
| MRR | — | Position of first grade-2 item |

Each metric is computed separately for Morgan, MolDeBERTa, Hybrid, and NAICS embeddings.
A Wilcoxon signed-rank test is used to determine whether differences are significant.

### 3. End-to-end RAG quality (RAGAS)

Evaluated on the frozen synthetic QA benchmark against the test split.

| Metric | What it measures |
|--------|----------------|
| Faithfulness | Is the answer supported by the retrieved context? |
| Answer relevancy | Does the answer address the question? |
| Context precision | What fraction of the context is actually useful? |
| Context recall | Does the context contain the information needed? |

Run independently with each embedding type as the retriever to isolate retrieval
contribution from generation quality.

### 4. Ablation

| Condition | Purpose |
|-----------|---------|
| Morgan only | Structural retrieval baseline |
| MolDeBERTa only | Semantic retrieval baseline |
| Hybrid | Combined; expected best overall |
| NAICS only | Sector-level retrieval |
| No graph traversal (flat RAG) | Quantify the contribution of graph context |

The last condition (flat RAG) is essential for the manuscript: it isolates what the
knowledge graph adds beyond plain vector search.

---

## Implementation Checklist (to complete before evaluation)

- [ ] `src/rag_graph/retrieval/` — vector retriever, graph traverser, retriever factory
- [ ] `src/rag_graph/context/` — subgraph formatter, context builder
- [ ] `src/rag_graph/generation/` — prompt manager (YAML-backed), RAG pipeline (LangGraph)
- [ ] `src/rag_graph/evaluation/` — anchor builder, retrieval metrics, embedding metrics, RAGAS evaluator
- [ ] Freeze synthetic QA benchmark before any system tuning
- [ ] Pre-register evaluation protocol (anchor thresholds, k values, test split) — this file serves that purpose

---

## Notes

- Anchor thresholds (Tanimoto 0.4 / 0.8) and k values (5, 10, 20) are fixed here and
  must not be changed after system tuning begins.
- All evaluation code must run against the **test split only** after development is done.
  No peeking at test-split retrieval results during system development.
- Prompts live in `config/prompts/` as YAML files managed by `PromptManager`.
  Prompt versions used at evaluation time must be recorded alongside results.
