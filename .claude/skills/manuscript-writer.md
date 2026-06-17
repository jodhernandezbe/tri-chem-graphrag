# Manuscript Writer Skill

Guidelines for drafting, editing, and reviewing manuscript sections for the TRI
Chemical GraphRAG paper. Use this when asked to write or improve any part of the paper.

---

## Paper Profile

| Item | Detail |
|------|--------|
| Type | Research article (computational / informatics) |
| Contribution | GraphRAG pipeline + three-way embedding ablation for TRI source reduction data |
| Primary audience | Computational chemists, NLP/IR researchers, environmental informaticians |
| Target length | ~8 000 words + tables + figures |
| Reference style | Numbered (Vancouver) or author-year (APA) — confirm with target journal |

---

## Canonical Section Outline

Use this as the skeleton when drafting. Each section maps to specific project artefacts.

### Abstract (~250 words)
- Problem: answering complex source reduction questions over TRI requires cross-referencing chemicals, sectors, and facilities — infeasible with keyword search alone
- Solution: GraphRAG pipeline with a Neo4j knowledge graph + three parallel chemical embedding types
- Key result: one table cell per primary metric (fill from `docs/EVALUATION_PROTOCOL.md` Table 1–3)
- Significance: first GraphRAG system applied to TRI; shows graph context improves faithfulness over flat retrieval

### 1 · Introduction
- Hook: industrial source reduction at scale; 529 chemicals; 10 years of TRI data
- Gap: existing TRI tools are static dashboards; no language-based querying
- Contribution list (numbered, 4–5 items):
  1. KG schema covering TRI Chemical–Facility–Sector–Reduction relationships
  2. Three parallel chemical embeddings (Morgan, MolDeBERTa, Hybrid) + NAICS sector embeddings
  3. LangGraph pipeline with cross-entity retrieval
  4. Pre-registered evaluation with Tanimoto graded relevance and RAGAS
  5. Open-source code + `neo4j.dump` snapshot for reproducibility

### 2 · Background
- 2.1 TRI and source reduction: what Form R captures, R1–R3 codes, NAICS structure
- 2.2 Knowledge graph RAG: contrast with flat vector RAG; subgraph traversal
- 2.3 Chemical similarity: Morgan fingerprints, transformer-based molecular embeddings

### 3 · System Design
- 3.1 Knowledge graph schema — reference `graph_model.svg`
- 3.2 Embedding strategy — Table from `docs/EVALUATION_PROTOCOL.md` "Embedding Types"
- 3.3 Retrieval pipeline — plan → retrieve → build_context → generate (LangGraph DAG)
- 3.4 Cross-entity retrieval — Chemical × Sector intersection via Cypher

### 4 · Evaluation Design
- Map directly from `docs/EVALUATION_PROTOCOL.md`
- Emphasise: Tanimoto anchor is external to all embeddings (independence argument)
- State locked parameters (Table of pre-registered values)
- Note QA benchmark generation and freeze date

### 5 · Results
- 5.1 Embedding space quality → Table 1
- 5.2 Retrieval ranking → Table 2 + nDCG@k figure
- 5.3 End-to-end RAG quality → Table 3
- 5.4 Ablation: flat vs graph RAG → contrast rows in Table 3
- 5.5 Statistical significance → Table 4

### 6 · Discussion
- Which embedding type performs best and why (structural vs semantic vs hybrid)
- What graph context adds over flat RAG (faithfulness / context recall delta)
- Failure modes: chemicals without SMILES, rare sectors
- Limitations: local LLM judge, single TRI jurisdiction (US only)

### 7 · Conclusion
- 2–3 paragraphs; no new claims
- Restate the main finding in plain language
- Future work: real-time TRI ingestion, multi-jurisdiction extension, human evaluation

---

## Writing Conventions

**Voice:** Third person, passive acceptable in Methods, active preferred in Results/Discussion.

**Numbers:** Spell out one through nine; use digits for 10 and above. Always pair
a number with units (e.g., "a 768-dimensional vector", "529 chemicals").

**Metric reporting:** Report as mean ± standard deviation where applicable.
For RAGAS scores, report to three decimal places. For significance: "p < 0.05 (Wilcoxon signed-rank)".

**Chemical names:** Use the IUPAC name on first mention, then the common name or
CAS number. Write Morgan *fingerprint* (not "Morgan fingerprinting").

**Acronyms:** Define on first use: TRI (Toxics Release Inventory), GraphRAG,
NAICS (North American Industry Classification System), KG (knowledge graph),
RAGAS (Retrieval-Augmented Generation Assessment).

**Figures:** Every figure needs a self-contained caption (title + method + what to observe).
Suggested figures: (1) KG schema diagram, (2) LangGraph pipeline DAG,
(3) t-SNE projections panel (one per embedding type), (4) nDCG@k bar chart.

---

## Section Drafting Prompts

Use these when asked to draft a specific section:

**To draft Introduction:** "Write the Introduction for the TRI Chemical GraphRAG paper.
The contribution list should mirror the 5 items in the manuscript outline above.
Close with a paragraph describing the paper's structure."

**To draft Methods:** "Write Section 4 (Evaluation Design) from the information in
`docs/EVALUATION_PROTOCOL.md`. Use past tense for what was done and present tense
for the pre-registered design. Keep formulas to one paragraph on nDCG."

**To draft Results:** "Write Section 5 (Results) using the filled-in tables from
`docs/EVALUATION_PROTOCOL.md`. Narrate the tables; do not only repeat the numbers.
Lead each subsection with the main finding."

**To edit a draft:** "Edit the following draft for clarity, concision, and scientific
register. Remove hedging words (might, could potentially, seems to). Ensure every
claim is tied to a table, figure, or citation."

---

## Common Pitfalls to Avoid

- **Over-claiming:** "best" and "state-of-the-art" require a direct benchmark comparison
- **Vague motivation:** quantify the scale of the problem (records, chemicals, years)
- **Passive-only prose:** alternate with active to maintain readability
- **Missing uncertainty:** report confidence intervals or p-values alongside point estimates
- **Mixing train and test split:** development results must never appear in the Results section
