# Supporting Information — End-to-End Smoke Validation of the TRI Chemical GraphRAG System

This appendix documents a deployment-level **smoke validation** of the TRI Chemical
GraphRAG recommendation system. The objective is to confirm that the fully assembled
production service — natural-language input, structured chemical extraction, deterministic
structure validation, graph-grounded retrieval, and answer generation — operates coherently
on chemicals of contemporary regulatory relevance, and that it recommends source-reduction
activities drawn from *structurally analogous* substances, not merely the query chemical
itself.

This is a qualitative integration check on the running endpoint; it is distinct from the
quantitative, pre-registered evaluation in [`docs/EVALUATION_PROTOCOL.md`](../docs/EVALUATION_PROTOCOL.md)
and does not alter any reported metric.

---

## 1. System under evaluation

Each test issues an HTTP `POST /query` request to the deployed FastAPI service and records
the structured response. The request traverses the complete production pipeline:

| Stage | Function |
|-------|----------|
| Structured extraction | An LLM (`ChatOllama`, `llama3.1`) with constrained (`with_structured_output`) decoding parses the query into a chemical SMILES plus optional industry sector and condition of use. |
| Structure validation | The candidate SMILES is validated deterministically with RDKit and canonicalized; chemically invalid input is rejected before any retrieval. |
| Graph-grounded retrieval | The validated structure is embedded and matched (cosine ANN) against a Neo4j vector index of TRI chemicals; an optional industry sector (NAICS-title index) or condition of use (Activity-name index) further conditions the graph traversal. |
| Recommendation assembly | Source-reduction records from the retrieved chemical neighbourhood are ranked by structural similarity and interleaved across chemicals so that analogues surface alongside the query substance. |
| Answer generation | A grounded narrative answer is produced from the assembled subgraph context. |

---

## 2. Query construction

Every request carries a chemical **SMILES**. To exercise the two optional conditioning
pathways, each chemical is additionally annotated with **either** an industry sector (free
text, resolved by semantic search over NAICS industry titles) **or** a condition of use (free
text, resolved by semantic search over TRI chemical-activity descriptions).

> **Note on the context strings.** Each industry-sector or condition-of-use value corresponds to
> a **challenging industrial scenario that EPA identified as driving — or significantly
> contributing to — the unreasonable-risk determination** for that chemical in its TSCA risk
> evaluation (per-chemical references in §3, *Challenging-scenario provenance*). The strings are
> concise descriptions of those risk-driving conditions of use, *not necessarily verbatim*
> regulatory category labels; both fields are resolved by *semantic* (embedding) similarity
> rather than exact string matching.

---

## 3. Chemical test set

The query chemicals are drawn from chemicals addressed under the EPA Toxic Substances Control
Act (TSCA) risk-evaluation program, spanning three designation cohorts (see *Provenance* and
*Sources* below). All eleven are reported below.

| # | Chemical | CAS RN | SMILES | Conditioning pathway | Context value (EPA risk-driving scenario) |
|---|----------|--------|--------|----------------------|---------------|
| 1 | Trichloroethylene | 79-01-6 | `ClC=C(Cl)Cl` | Condition of use | Vapor degreasing of fabricated metal parts |
| 2 | Tetrachloroethylene (PCE) | 127-18-4 | `ClC(Cl)=C(Cl)Cl` | Industry sector | Dry cleaning and laundry services |
| 3 | Methylene chloride | 75-09-2 | `ClCCl` | Condition of use | Commercial paint and coating removal |
| 4 | 1-Bromopropane | 106-94-5 | `CCCBr` | Condition of use | Vapor degreasing and spray adhesive application |
| 5 | Carbon tetrachloride | 56-23-5 | `ClC(Cl)(Cl)Cl` | Condition of use | Reactant and feedstock in chemical manufacturing |
| 6 | 1,4-Dioxane | 123-91-1 | `C1COCCO1` | Condition of use | Generated as a byproduct during industrial processing |
| 7 | N-Methylpyrrolidone (NMP) | 872-50-4 | `CN1CCCC1=O` | Condition of use | Industrial use as a solvent |
| 8 | 1,3-Butadiene | 106-99-0 | `C=CC=C` | Industry sector | Synthetic rubber and plastics material manufacturing |
| 9 | Formaldehyde | 50-00-0 | `C=O` | Industry sector | Synthetic resin and adhesive manufacturing |
| 10 | Tris(2-chloroethyl) phosphate (TCEP) | 115-96-8 | `O=P(OCCCl)(OCCCl)OCCCl` | Condition of use | Flame-retardant additive in paint and coating manufacturing and polymers |
| 11 | Vinyl chloride | 75-01-4 | `C=CCl` | Industry sector | Polyvinyl chloride (PVC) and plastics resin manufacturing |

> The harness additionally executes a single **input-validation negative control** (a malformed
> SMILES string) to confirm that chemically invalid input is rejected and routed to a
> clarification response rather than into retrieval. As a deliberately failing input, it is
> excluded from the chemistry results reported above and serves only to verify the validation
> gate.

### Provenance

The eleven substances were selected, without modification, from three EPA TSCA risk-evaluation
designation cohorts:

| Cohort (designation date) | Substances used here |
|---------------------------|----------------------|
| First ten chemicals undergoing risk evaluation (Dec 19, 2016) | Trichloroethylene; tetrachloroethylene; methylene chloride; carbon tetrachloride; 1-bromopropane; 1,4-dioxane; N-methylpyrrolidone |
| Next twenty high-priority substances (Dec 2019) | Tris(2-chloroethyl) phosphate (#14); 1,3-butadiene (#17); formaldehyde (#19) |
| High-priority designation (prioritization initiated Dec 2023; finalized Dec 2024) | Vinyl chloride |

### Challenging-scenario provenance

Each context value corresponds to a challenging industrial scenario associated with that
chemical's TSCA risk evaluation — a condition of use or industrial setting that EPA links to
the unreasonable-risk determination (or, for substances still under evaluation, a condition of
use within scope). Per-chemical EPA references are linked below; the scenario wording was
checked against those pages (June 2026; see *Reference verification*) and limited to what the
linked page supports.

| Chemical | Pathway | EPA-associated challenging scenario | EPA risk evaluation |
|----------|---------|-------------------------------------|---------------------|
| Trichloroethylene | Condition of use | Vapor degreasing (a primary industrial use); TCE presents unreasonable risk under its conditions of use | [TCE](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-trichloroethylene-tce-0) |
| Tetrachloroethylene (PCE) | Industry sector | Commercial dry cleaning, a condition of use; 60 of 61 conditions of use contribute to unreasonable risk | [PCE](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-perchloroethylene-pce) |
| Methylene chloride | Condition of use | Paint and coating removal; methylene chloride presents unreasonable risk under its conditions of use | [Methylene chloride](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-methylene-chloride) |
| 1-Bromopropane | Condition of use | Vapor degreasing and spray-adhesive use; 1-BP presents unreasonable risk under its conditions of use | [1-BP](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-1-bromopropane-1-bp) |
| Carbon tetrachloride | Condition of use | Industrial processing/use in chemical manufacturing; 13 of 15 conditions of use present unreasonable risk | [Carbon tetrachloride](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-carbon-tetrachloride) |
| 1,4-Dioxane | Condition of use | Worker exposure where 1,4-dioxane is generated as a byproduct; significantly contributes to unreasonable risk | [1,4-Dioxane](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/final-risk-evaluation-14-dioxane) |
| N-Methylpyrrolidone (NMP) | Condition of use | Industrial/commercial solvent use (dermal exposure); 29 of 37 conditions of use drive unreasonable risk | [NMP](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-n-methylpyrrolidone-nmp-0) |
| 1,3-Butadiene | Industry sector | Monomer in polymerization for synthetic rubber, plastic, and resin manufacturing (inhalation) | [1,3-Butadiene](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-13-butadiene) |
| Formaldehyde | Industry sector | Worker inhalation during manufacturing/processing (e.g., resin production) | [Formaldehyde](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-formaldehyde) |
| Tris(2-chloroethyl) phosphate (TCEP) | Condition of use | Flame-retardant use in paint and coating manufacturing and polymers; seven conditions of use present unreasonable worker risk | [TCEP](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-tris2-chloroethyl-phosphate-tcep) |
| Vinyl chloride | Industry sector | Manufacturing/processing as the monomer in PVC production (risk evaluation in progress) | [Vinyl chloride](https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-vinyl-chloride) |

#### Reference verification

All eleven EPA reference URLs were retrieved and checked (June 2026). Each resolves and confirms
the named chemical and its risk-evaluation status. Landing pages confirm the unreasonable-risk
determination and the use/sector cited above; chemical-specific condition-of-use counts that
appear on the page are quoted (PCE 60/61, carbon tetrachloride 13/15, TCEP seven). Two scope
clarifications identified during verification and reflected above: for carbon tetrachloride, the
*reactant-in-reactive-ion-etching* use is one of the two conditions of use that do **not** drive
risk (so the scenario is stated as general industrial manufacturing); for NMP, *consumer* paint-
and-coating removal is among the conditions of use that do **not** drive risk (so the scenario is
stated as industrial/commercial solvent use).

### Sources

- EPA, *Risk Evaluations for Existing Chemicals under TSCA* — https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluations-existing-chemicals-under-tsca
- EPA, *Scopes of the Risk Evaluations to be Conducted for the First Ten Chemical Substances under TSCA* — https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/scopes-risk-evaluations-be-conducted-first-ten-chemical
- EPA, *EPA Finalizes List of Next 20 Chemicals to Undergo Risk Evaluation under TSCA* — https://www.epa.gov/newsreleases/epa-finalizes-list-next-20-chemicals-undergo-risk-evaluation-under-tsca
- EPA, *Risk Evaluation for Vinyl Chloride* — https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/risk-evaluation-vinyl-chloride
- EPA, *Ongoing and Completed Chemical Risk Evaluations under TSCA* — https://www.epa.gov/assessing-and-managing-chemicals-under-tsca/ongoing-and-completed-chemical-risk-evaluations-under

### Structure verification

Every (CAS RN, SMILES) pair in the test-set table was verified against PubChem (June 2026): the
InChIKey computed from each SMILES with RDKit matched the InChIKey of the PubChem record
retrieved for the corresponding CAS Registry Number. All eleven matched. PubChem Compound IDs
(CIDs) for traceability:

| Chemical | CAS RN | PubChem CID | InChIKey |
|----------|--------|-------------|----------|
| Trichloroethylene | 79-01-6 | 6575 | XSTXAVWGXDQKEL-UHFFFAOYSA-N |
| Tetrachloroethylene (PCE) | 127-18-4 | 31373 | CYTYCFOTNPOANT-UHFFFAOYSA-N |
| Methylene chloride | 75-09-2 | 6344 | YMWUJEATGCHHMB-UHFFFAOYSA-N |
| 1-Bromopropane | 106-94-5 | 7840 | CYNYIHKIEHGYOZ-UHFFFAOYSA-N |
| Carbon tetrachloride | 56-23-5 | 5943 | VZGDMQKNWNREIO-UHFFFAOYSA-N |
| 1,4-Dioxane | 123-91-1 | 31275 | RYHBNJHYFVUHQT-UHFFFAOYSA-N |
| N-Methylpyrrolidone (NMP) | 872-50-4 | 13387 | SECXISVLQFMRJM-UHFFFAOYSA-N |
| 1,3-Butadiene | 106-99-0 | 7845 | KAKZBPTYRLMSJV-UHFFFAOYSA-N |
| Formaldehyde | 50-00-0 | 712 | WSFSSNUMVMOOMR-UHFFFAOYSA-N |
| TCEP | 115-96-8 | 8295 | HQUQLFOMPYWACS-UHFFFAOYSA-N |
| Vinyl chloride | 75-01-4 | 6338 | BZHJMEDXRYGGRV-UHFFFAOYSA-N |

Source: PubChem PUG REST (`https://pubchem.ncbi.nlm.nih.gov/rest/pug`). EPA URLs above verified
June 2026; page reorganizations may relocate them — confirm before final submission.

---

## 4. Recorded outputs

For each chemical the runner writes a per-case report to
`data/smoke_tests_response/NN_<chemical>.txt`, containing:

- the request (SMILES and conditioning context);
- the generated narrative answer;
- the **distinct source chemicals** contributing recommendations, with their SMILES-similarity
  scores — evidencing cross-chemical (analogue-based) recommendation; and
- the ranked, cross-chemical-interleaved source-reduction records.

Because these reports embed TRI-derived source-reduction records, the
`data/smoke_tests_response/` directory is excluded from version control in accordance with the
project's data-handling policy.

---

## 5. Reproducibility

The full stack must be running: Neo4j (with the chemical, NAICS-title, and Activity-name
vector indices populated), an Ollama server providing `llama3.1`, and the FastAPI service.

```bash
# bring up the stack (docker-compose)
docker compose up -d neo4j ollama app-dev
docker exec -it ollama ollama pull llama3.1     # ensure the generator model is present

# or run the API locally against the dockerized Neo4j/Ollama
uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

```bash
# execute the smoke validation (from the repository root)
python -m smoke_testing.smoke_test
python -m smoke_testing.smoke_test --base-url http://localhost:8000 --timeout 300
GRAPHRAG_API_URL=http://localhost:8000 python -m smoke_testing.smoke_test
```

The chemical definitions are declared in
[`epa_risk_eval_cases.py`](./epa_risk_eval_cases.py); the runner is
[`smoke_test.py`](./smoke_test.py).
