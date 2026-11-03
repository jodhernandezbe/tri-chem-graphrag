# Surrogate selection and P2 likelihood analysis (response to G. Ruiz review, Sept 2026)

Standalone scripts, independent of Neo4j and Ollama, that read the repo's TRI files.
Run from a scratch folder (they read `~/mnt/tri_chem_graphrag/data/raw` and `data/processed`; adjust the paths as needed).

1. `build.py <years>`: extracts every source reduction activity (code, description, estimated annual reduction R1 to R6) from `data/raw/tri_2a_<year>.csv` into `sr_<year>.csv`. `agg.py` concatenates them into `sr_all.pkl` and writes `code_catalog.csv` and `sr_by_year.csv`.
2. `ctx.py <years>`: caches processed 1a (SMILES, NAICS) and 1b (condition of use) files. `feat_kg.py` builds per-chemical NAICS and condition-of-use profiles for 2014 to 2024; `feat.py` does the same for all years.
3. `xwalk.py`: family-level mapping of the W-codes (1991 to 2020) and S-codes (2021 onward) to the five families F1 to F5. Verify it against EPA's official crosswalk.
4. `anal_kg.py` / `cases_kg.py`: **manuscript version**. Uses the same data window as the knowledge graph: 2014 to 2024 activities with an estimated reduction of R1 to R3, and chemicals with at least 3 such records (n = 88). It computes Spearman correlations and the surrogate-selection experiment, comparing structure, physicochemical descriptors, NAICS, condition of use and context, and reports the TSCA case analogues.
5. `anal.py` / `cases.py` / `rob.py`: all-years (1987 to 2020 W-code) sensitivity version, not used in the manuscript (n = 368).
6. The likelihood of at least a 25% reduction by family and activity (2014 to 2024, all reports with an estimate) is computed in `cases.py` and `cases_kg.py`.

Key finding: the TRI 2a estimated-annual-reduction field exists only from RY2014 onward. `Tri2aTransformer` keeps R1 to R3 only, so re-running the ETL for 1987 to 2013 adds no records to the graph.
