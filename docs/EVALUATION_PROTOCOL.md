# Evaluation Protocol v2 — TRI Chemical GraphRAG

Pre-registered evaluation design for the manuscript.  All thresholds, k values, test-split
boundaries, and NAICS levels below are **locked** — do not change them after development begins.

---

## Research Goal and Evaluation Rationale

The system's production objective is:

> Given a query chemical (and optionally an industry context), recommend source reduction
> activities that have historically been effective for chemically similar substances in
> the same or related industries.

The evaluation must therefore measure whether **the retrieved chemicals provide useful
activity coverage**, not merely whether they are structurally similar to the query.

| What previous protocol measured | What this protocol measures |
|---------------------------------|-----------------------------|
| "Are structurally similar chemicals ranked higher?" | "Do retrieved chemicals cover the query's actual activities?" |
| Graded pairwise similarity (nDCG) | Activity Coverage@k (fraction of ground-truth activities found) |
| One anchor type at a time | Multi-level: exact codes + activity families + NAICS hierarchy |

The contrast between structural retrieval quality (Tanimoto) and activity coverage is
the **key scientific finding**: topology alone is insufficient; graph context is essential.

---

## System Under Evaluation

A GraphRAG pipeline that answers natural-language questions about **source reduction
activity** in the EPA Toxics Release Inventory (TRI), grounded in a Neo4j knowledge
graph covering TRI reports from 2014 to 2024.

| Component | Choice |
|-----------|--------|
| Knowledge graph | Neo4j 5.x, TRI 2014–2024 |
| Orchestration | LangGraph + LangChain |
| Generator LLM | Ollama (llama3.1) |
| RAGAS judge LLM | Ollama (granite3.3:8b) — configurable via `RAG_GRAPH.RAGAS.JUDGE_MODEL`, kept separate from the generator |

### Embedding Types

| Label | Model | Dims | What it captures |
|-------|-------|-----:|-----------------|
| MorganFingerprint | RDKit (radius 2, 2048 bits) | 2048 | Structural topology |
| MolDeBERTa | SaeedLab/MolDeBERTa-base-123M-mtr | 768 | Semantic / functional meaning |
| Hybrid | Morgan ‖ MolDeBERTa | 1792 | Combined structural + semantic |

The three types are the single source of truth in `CHEMICAL_EMBEDDING_TYPES`
(`src/validation/retrieval_metrics.py`). The pipeline **skips** any embedding whose
Neo4j index or model cannot be loaded and logs a warning, rather than aborting the run.

---

## Data Split

### Design Rationale — Why Stratified Random, Not Temporal

A temporal split (e.g., 2014–2020 index / 2021–2024 test) introduces two systematic
biases that corrupt ActivityCoverage@k:

1. **Chemical imbalance**: chemicals added to TRI after 2018 have very few index reports
   and relatively many test reports.  Their ActivityCoverage is artificially low because
   the index has little history to retrieve, not because the embeddings are poor.

2. **Activity code drift**: TRI has expanded its reportable activity codes over the years,
   so recent cohorts over-represent codes that are rare in the index.  This inflates
   the denominator of ActivityCoverage (`|A_q|`) relative to what the index can supply.

A **stratified random split**, with Report nodes randomly assigned 80 / 20 per chemical,
ensures every chemical has proportional representation in both splits regardless of when
it started reporting.  This measures retrieval quality, not data recency bias.

The tradeoff is that it no longer simulates strict temporal deployment (train on past,
predict future).  For this manuscript — which evaluates **retrieval quality**, not
forecasting — statistical balance takes precedence over temporal realism.

### Split Specification

| Split | Fraction | Neo4j property | Role |
|-------|----------|----------------|------|
| Index | 80 % | `r.split = "index"` | Anchor construction, embedding index, activity history for retrieval |
| Test  | 20 % | `r.split = "test"`  | Ground-truth activity profiles for all evaluation metrics |

**Stratification key**: `cas_number` — for each unique chemical, 20 % of its `Report`
nodes are randomly assigned to the test split.

**Edge cases** (pre-registered):
- Chemicals with exactly 1 Report: assigned to index; excluded from evaluation queries.
- Chemicals with 2–4 Reports: at least 1 assigned to each split via ceiling rounding.
- Chemicals with 0 test Reports after rounding: excluded from evaluation queries.

**Parameters** (locked):

| Parameter | Value |
|-----------|-------|
| Test fraction | 0.20 |
| Random seed | 42 |
| Stratification key | `cas_number` |

### Implementation

The split is computed **once** in Python and written back to Neo4j as a `split` property
on every `Report` node.  All subsequent Cypher queries use `WHERE r.split = "index"` or
`WHERE r.split = "test"` — no year filtering.

```bash
python -m src.validation.main --steps split anchors embedding retrieval
```

The split step is **idempotent**: if `r.split` is already set on all Report nodes, it
is a no-op (unless `--force-resplit` is passed).

---

## Activity Profile Framework

### Activity Codes and Families

TRI source reduction activities are organised at two levels of granularity:

**Level 1 — Exact codes** (74 unique codes observed)

**Level 2 — Activity families** (5 groups from `TRI_SOURCE_REDUCTION_GROUPS` in `config.yaml`):

| Family | Name | Codes |
|--------|------|-------|
| F1 | Material Substitutions and Modifications | S01–S06 |
| F2 | Product Modifications | S11–S14 |
| F3 | Process and Equipment Modifications | S21–S25 |
| F4 | Inventory and Material Management | S31–S35 |
| F5 | Operating Practices and Training | S41–S44 |

Two activity profile representations are used throughout the evaluation:

- **Code-level profile** `A_c`: set of distinct activity codes applied to chemical c
- **Family-level profile** `F_c`: set of distinct family IDs mapped from `A_c`

### NAICS Hierarchy

Industry context is stored as 6-digit National Industry codes in `Sector.naics_code`.
Higher (broader) levels are extracted by prefix truncation:

| Digits | Name | Example |
|--------|------|---------|
| 2 | Sector | 32 (Manufacturing) |
| 3 | Subsector | 325 (Chemical Mfg.) |
| 4 | Industry Group | 3251 (Basic Chemical) |
| 5 | NAICS Industry | 32519 (Other Basic Inorganic) |
| 6 | National Industry | 325194 |

SectorCoverage is evaluated at digit depths: **2, 3, 4, 6** (5-digit skipped as it is
sparsely populated in TRI data).

Activity profiles can be **global** (all sectors) or **sector-conditioned** at digit
depth d:

    A_c(d, prefix) = activity codes for chemical c in facilities whose NAICS code
                     starts with `prefix` of length d

---

## Ground-Truth Anchors

### Primary Ground Truth — Per-Chemical Activity Profile (test split)

For each test chemical c_q, the ground truth is its observed activity profile in the
test split:

- `A_q` = set of activity codes for c_q in test split
- `F_q` = set of activity families for c_q in test split (mapped from A_q)
- `A_q(d, s)` = sector-conditioned subset at digit depth d and prefix s

These are **per-chemical activity sets**, not pairwise similarity scores.  The evaluation
asks: "does retrieval recover A_q?"

### Secondary Anchor — Operational (Jaccard, pairwise, from index split)

Pairwise Jaccard Activity Similarity between chemicals built from the index split.
Used for nDCG-based retrieval evaluation to compare with activity coverage.

    J(c_i, c_j) = |A_i ∩ A_j| / |A_i ∪ A_j|        (code-level, index split)
    J_F(c_i, c_j) = |F_i ∩ F_j| / |F_i ∪ F_j|       (family-level, index split)

Grading (pre-registered):

| Criteria | Grade | Label |
|----------|-------|-------|
| J ≥ 0.7 AND N ≥ 3 | 2 | Highly Relevant |
| J ≥ 0.3 OR N ≥ 1 | 1 | Partially Relevant |
| J < 0.3 AND N = 0 | 0 | Not Relevant |

### Baseline Anchor — Structural (Tanimoto, pairwise)

RDKit Morgan fingerprints, independent of any stored embedding.  Used **only** to show
that structural similarity does not predict activity coverage (expected null result).

| Tanimoto | Grade |
|----------|-------|
| ≥ 0.8 | 2 |
| 0.4–0.8 | 1 |
| < 0.4 | 0 |

---

## Evaluation Dimensions

### 1 · Embedding Space Quality (intrinsic)

*Purpose:* assess how well each embedding organises chemical space before any retrieval.

| Metric | Library | Reference |
|--------|---------|-----------|
| Silhouette score ↑ | scikit-learn | Functional class (VOC, heavy metal, dioxin, solvent, acid) |
| Davies-Bouldin index ↓ | scikit-learn | Same |
| Spearman ρ vs Tanimoto ↑ | scipy | Cosine distance vs 1 − Tanimoto (structural) |
| Spearman ρ vs Jaccard (code) ↑ | scipy | Cosine distance vs 1 − J (operational, code-level) |
| Spearman ρ vs Jaccard (family) ↑ | scipy | Cosine distance vs 1 − J_F (family-level) |
| t-SNE projection | scikit-learn | Qualitative figure |

**Hypotheses:** MorganFingerprint will show high ρ_Tanimoto but near-zero ρ_Jaccard(code)
and ρ_Jaccard(family).  MolDeBERTa and Hybrid may show marginally higher family-level ρ.

Implemented in `src/validation/embedding_metrics.py → EmbeddingMetrics.compute()`.

---

### 2 · Activity Coverage Evaluation (PRIMARY)

*Purpose:* directly measure whether retrieval can support source reduction
recommendation — this is the production utility metric.

**Top-k recommendation framing.**
For each test chemical c_q (≥ 1 test-split activity AND present in the vector index),
retrieve its nearest-neighbour chemicals and flatten them, in similarity order, into a
ranked list of **distinct activity codes**.  The top-k codes R_k are the recommendation.
Scoring ranked codes — rather than the union of the top-k chemicals' full histories —
prevents the metric from being won by indiscriminate breadth (see *Why not union
coverage*). Code sets are drawn from the **index split** to match production.

**Primary metrics:**

    WeightedCoverage@k = Σ_{c ∈ A_q ∩ R_k} w(c) / Σ_{c ∈ A_q} w(c)
        where w(c) = ln((N + 1) / (df(c) + 1)) + 1   (IDF over N index chemicals)

    CodePrecision@k    = |A_q ∩ R_k| / |R_k|

WeightedCoverage rewards recovering rare, specific codes over ubiquitous ones;
CodePrecision penalises padding the recommendation with irrelevant codes.  Both are
reported with **Lift@10 vs Random** (absolute Δ and ratio) — a method genuinely helps
only when Lift > 0.

**Diagnostic metrics** (not used to rank methods):

    RawCoverage@k    = |A_q ∩ R_k| / |A_q|             (unweighted; breadth-dominated)
    FamilyCoverage@k = |F_q ∩ family(R_k)| / |F_q|     (activity-family level)

k values: 5, 10, 20 (pre-registered).  Neighbour pool before flattening to codes: 50.

**Baselines:**
- *Random@k*: k codes drawn uniformly from the index code vocabulary (Monte-Carlo,
  200 draws per query, seed=42).  Population restricted to chemicals with SMILES.
- *Oracle@k*: top-k codes taken from the index chemicals ranked by Jaccard overlap
  between their index-split codes and the query's test-split codes (ground-truth
  ceiling — not achievable in production).

**Why not union coverage.**
The original primary metric, `ActivityCoverage@k = |A_q ∩ ∪_{c∈R_k} A_c| / |A_q|`,
unions the *full* code-sets of the top-k retrieved chemicals.  Because TRI source-
reduction codes are popularity-skewed, unioning any k chemicals recovers most of A_q, so
uniformly random retrieval **beat every embedding** (Random@10 ≈ 0.86 vs best embedding
≈ 0.70).  IDF weighting alone does not fix this — the union form is structurally
breadth-biased.  The top-k-codes reframing restores discrimination: every embedding now
exceeds Random by ≈ 2.5–2.8× on WeightedCoverage@10.  The union form is retained only as
the RawCoverage diagnostic and for the sector-conditioned view below.

**SectorCoverage@k** (secondary): the union-form coverage restricted to a NAICS prefix s
at digit depth d ∈ {2, 3, 4, 6}, for both query and retrieved codes.

**Hypotheses (revised):**
- H1: FamilyCoverage@k > RawCoverage@k for all embeddings (families more predictable).
- H2: SectorCoverage(2-digit) > SectorCoverage(6-digit) — broader context, higher coverage.
- H3: Every embedding exceeds Random on WeightedCoverage@10 and CodePrecision@10
  (the breadth-robust primary metric; superseding the prior, since-falsified H3 that no
  embedding could beat the union-coverage Random baseline).
- H4: The deployed graph-augmented recommender (§2b) exceeds pure-vector retrieval.

Implemented in `src/validation/retrieval_metrics.py → ActivityCoverageMetrics.evaluate_all()`.

---

### 2b · Shipped Recommender — End-to-End Graph Path (PRIMARY, deployed)

*Purpose:* score the production recommendation path exactly as deployed, not just
pure-vector retrieval.

For each test chemical, run `RetrieverFactory.retrieve_multi(smiles)` (vector ANN +
graph traversal + cross-entity intersection), tabulate the ranked source-reduction
records via the shared `tabulate_records` helper, and take the top-k distinct
recommended codes.  Score with the same **WeightedCoverage@k** / **CodePrecision@k**
metrics plus Random (uniform k codes) and Oracle (the query's own highest-weight codes)
baselines.  This is the table that reflects what an end user actually receives.

Implemented in `src/validation/recommendation_metrics.py → RecommendationMetrics.evaluate()`.

---

### 3 · NAICS Hierarchy Analysis

*Purpose:* characterise how the NAICS digit depth affects activity coverage and
show that broader industrial context improves recommendation quality.

For each digit depth d ∈ {2, 3, 4, 6}:
1. Compute SectorCoverage@10(d) for each embedding type
2. Include Random baseline at each level

Expected pattern: SectorCoverage(2-digit) ≥ SectorCoverage(4-digit) ≥ SectorCoverage(6-digit).

Statistical test: Friedman test across NAICS depths for each embedding type (p < 0.05).

---

### 4 · Retrieval Quality, Anchor-Based (secondary)

*Purpose:* complement the coverage view with a pairwise-similarity ranking perspective
and provide a structural baseline for comparison with prior work.

**Structural anchor (Tanimoto):**
- nDCG@k, Recall@k, MRR at k = 5, 10, 20 — BASELINE ONLY

**Operational anchor (Jaccard, code-level):**
- Same metrics — SECONDARY (confirms coverage findings from Dimension 2)

Wilcoxon signed-rank tests on per-query nDCG@10 between embedding types.

Implemented in `src/validation/retrieval_metrics.py → RetrievalMetrics.evaluate_all()`.

---

### 5 · End-to-End RAG Quality (RAGAS)

*Purpose:* measure full pipeline quality on the frozen synthetic QA benchmark.

| Metric | What it measures |
|--------|----------------|
| Faithfulness ↑ | Answer supported by retrieved context |
| Context precision ↑ | Fraction of retrieved context that is useful |
| Context recall ↑ | Retrieved context contains the needed information |

Conditions: MorganFingerprint, MolDeBERTa, Hybrid, Flat RAG (no graph traversal).
Benchmark: frozen `data/qa_benchmark.json` (100 questions, 25 per type, generated
once from graph facts).  The `chemical_sector_reduction` question type directly tests
the production task: "What source reduction activities were applied to [chemical] in
[sector] in [year]?"

Implemented in `src/validation/ragas_evaluator.py → RAGASEvaluator.evaluate_all()`.

---

### 6 · Ablation Study

| Condition | What it isolates |
|-----------|-----------------|
| MorganFingerprint only | Structural retrieval baseline |
| MolDeBERTa only | Semantic retrieval baseline |
| Hybrid | Combined structural + semantic |
| Flat RAG (no graph traversal) | Vector retrieval without graph context |
| Full GraphRAG | Complete system: vector retrieval + graph traversal |

The contrast between Flat RAG and Full GraphRAG on ActivityCoverage@k and RAGAS
Context Recall is the primary claim: graph traversal is essential for activity recommendation.

---

## Statistical Testing

| Test | Applied to | When |
|------|-----------|------|
| Wilcoxon signed-rank (pairwise) | nDCG@10 (per-query, anchor-based) | Comparing embedding types |
| Lift vs Random | WeightedCoverage@10, CodePrecision@10 | Reporting every coverage/recommender table |
| Friedman test | SectorCoverage@10 across digit depths {2, 3, 4, 6} | Comparing NAICS levels per embedding |
| Wilcoxon signed-rank | GraphRAG vs Flat RAG on WeightedCoverage@10 | Ablation |

Significance threshold: p < 0.05.

---

## Implementation Map

| Evaluation step | Module | Entry point |
|-----------------|--------|-------------|
| Build Tanimoto anchor | `anchor_builder.py` | `AnchorBuilder.build_structural_anchors()` |
| Build operational anchor (pairwise) | `anchor_builder.py` | `AnchorBuilder.build_operational_anchors(naics_digits)` |
| Build per-chemical activity profiles | `anchor_builder.py` | `AnchorBuilder.build_activity_profiles()` |
| Write stratified split to Neo4j | `anchor_builder.py` | `AnchorBuilder.write_stratified_split()` |
| Embedding space metrics | `embedding_metrics.py` | `EmbeddingMetrics.compute()` |
| Activity coverage evaluation (PRIMARY, pure vector) | `retrieval_metrics.py` | `ActivityCoverageMetrics.evaluate_all()` |
| Shipped recommender (PRIMARY, graph path) | `recommendation_metrics.py` | `RecommendationMetrics.evaluate()` |
| NAICS hierarchy analysis | `retrieval_metrics.py` | `ActivityCoverageMetrics.evaluate_naics_levels()` |
| Retrieval quality, anchor-based (secondary) | `retrieval_metrics.py` | `RetrievalMetrics.evaluate_all()` |
| Significance tests | `retrieval_metrics.py` | `RetrievalMetrics.significance_test()` |
| Generate QA benchmark | `validation_data_generation/qa_generator.py` | `QAGenerator.generate()` |
| RAGAS evaluation | `ragas_evaluator.py` | `RAGASEvaluator.evaluate_all()` |
| Full pipeline entry point | `main.py` | `python -m src.validation.main` |

---

## Locked Parameters

These values are pre-registered and **must not change** after development begins:

| Parameter | Value | Location |
|-----------|-------|---------|
| Tanimoto grade-2 threshold | 0.8 | `anchor_builder.TANIMOTO_GRADE_HIGH` |
| Tanimoto grade-1 threshold | 0.4 | `anchor_builder.TANIMOTO_GRADE_LOW` |
| Jaccard grade-2 threshold | 0.7 | `anchor_builder.JACCARD_GRADE_HIGH` |
| Jaccard grade-1 threshold | 0.3 | `anchor_builder.JACCARD_GRADE_LOW` |
| NAICS co-occurrence grade-2 | 3 | `anchor_builder.NAICS_COOCCURRENCE_HIGH` |
| Test split fraction | 0.20 | `anchor_builder.TEST_FRACTION` |
| Random seed for split | 42 | `anchor_builder.RANDOM_SEED` |
| Stratification key | `cas_number` | `anchor_builder.write_stratified_split()` |
| nDCG / Recall k values | 5, 10, 20 | `retrieval_metrics.K_VALUES` |
| Coverage / recommender k values | 5, 10, 20 | `retrieval_metrics.K_VALUES` |
| NAICS digit depths for SectorCoverage | 2, 3, 4, 6 | `retrieval_metrics.NAICS_LEVELS` |
| Activity family mapping | TRI_SOURCE_REDUCTION_GROUPS | `config.yaml` |
| Morgan fingerprint radius | 2 | `config.yaml RAG_GRAPH.EMBEDDINGS` |
| Morgan fingerprint bits | 2048 | `config.yaml RAG_GRAPH.EMBEDDINGS` |

---

## CLI Reference

```bash
# Step 0 — compute and write stratified split to Neo4j (run once; idempotent)
python -m src.validation.main --steps split

# Force re-split even if r.split already exists
python -m src.validation.main --steps split --force-resplit

# Full evaluation (all steps, 6-digit NAICS, 20 RAGAS pairs per condition)
python -m src.validation.main --steps split anchors embedding retrieval ragas \
    --ollama-url http://localhost:11435

# Core metrics without RAGAS (faster — no LLM calls)
python -m src.validation.main --steps split anchors embedding retrieval

# RAGAS only (re-uses existing benchmark if present)
python -m src.validation.main --steps ragas --max-ragas-pairs 20 \
    --ollama-url http://localhost:11435

# With t-SNE projections (adds ~5 min)
python -m src.validation.main --steps anchors embedding retrieval

# Skip t-SNE (faster iteration)
python -m src.validation.main --steps anchors embedding retrieval --no-tsne
```
