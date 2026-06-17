# Scientific Research Skill

Context for literature search, methodology critique, and contribution framing in
the TRI Chemical GraphRAG project. Use this when searching for related work,
evaluating methodological choices, or positioning the paper's contributions.

---

## Project Identity (for search queries)

This work sits at the intersection of three fields. Use the clusters below to
seed literature searches:

**GraphRAG / Knowledge-Graph-augmented LLMs**
- Keywords: GraphRAG, knowledge graph RAG, graph-enhanced retrieval, KG-augmented generation
- Key venues: ACL, EMNLP, NAACL, ICLR, NeurIPS, WWW, KDD

**Chemical informatics + embedding**
- Keywords: molecular embedding, chemical fingerprint, MolBERT, ChemBERTa, MolDeBERTa,
  Tanimoto similarity, Morgan fingerprint, molecular similarity search
- Key venues: Journal of Chemical Information and Modeling, Journal of Cheminformatics,
  Nature Machine Intelligence, J. Chem. Theory Comput.

**Environmental data / TRI**
- Keywords: Toxics Release Inventory, source reduction, pollution prevention, NAICS sector,
  environmental compliance, EPA reporting
- Key venues: Environmental Science & Technology, Journal of Hazardous Materials,
  Environment International, Risk Analysis

---

## Methodological Positioning

When asked to find related work or assess novelty, evaluate along these axes:

| Axis | Our choice | What to compare against |
|------|-----------|------------------------|
| Chemical retrieval | Morgan + MolDeBERTa + Hybrid (parallel) | Single-embedding baselines |
| Graph context | Full Neo4j subgraph traversal | Flat vector search, no graph |
| Domain | TRI industrial chemicals, source reduction | PubChem, ChEMBL, drug discovery literature |
| Evaluation | Tanimoto graded relevance + RAGAS | Binary relevance, BEIR benchmarks |

The **key novelty claims** to protect:
1. Three-way chemical embedding ablation (structural, semantic, hybrid) on the same KG
2. NAICS sector embeddings as a separate retrieval axis for cross-entity traversal
3. Graded Tanimoto relevance as an external, model-independent retrieval anchor
4. TRI domain: underrepresented in NLP/IR literature

---

## Search Strategy Guidelines

When running a web or paper search:

1. **Start broad, then narrow.** First establish who else has done GraphRAG in chemistry/environment, then look for TRI-specific work.
2. **Check citation graphs.** For any key paper found, check its citing papers (forward) and cited papers (backward).
3. **Look for negative results.** Papers showing that flat vector search matches or beats graph-augmented retrieval are important for the Discussion.
4. **Separate retrieval papers from generation papers.** The RAGAS metrics address generation quality; nDCG/MRR address retrieval quality — don't conflate their literature.

---

## Methodology Critique Checklist

When reviewing experimental design in related work or evaluating our own:

- [ ] Is the evaluation anchor independent of the model being evaluated?
- [ ] Are relevance judgements graded or binary? (graded is preferred for ranking tasks)
- [ ] Is the train/test split defined over time (no data leakage)?
- [ ] Are statistical significance tests reported?
- [ ] Is the baseline (flat RAG) reported alongside the proposed system?
- [ ] Are prompt versions recorded alongside evaluation results?

---

## TRI Data Facts (for Methods framing)

- TRI Form R: mandatory annual report for facilities releasing or managing > threshold amounts of listed chemicals
- Source reduction codes R1–R3: the three activity types in this project (raw material substitution, process modifications, product changes)
- NAICS: North American Industry Classification System — standard sector taxonomy
- 529 chemicals in the current KG (with source reduction records 2014–2024)
- Facilities are uniquely identified by TRIFID

---

## Output Format for Research Tasks

When delivering literature-search results, structure them as:

```
## [Topic]
**[Paper title]** (Author et al., Year) — [venue]
Relevance: [one sentence on how it relates to our work]
Difference: [one sentence on what we do differently]
```
