import glob
import os
import pickle
import sys

import pandas as pd

P = "/Users/josehernandez/Documents/tri_chem_graphrag/data/processed"
yrs = sys.argv[1:]
for y in yrs:
    a = pd.read_csv(
        f"{P}/tri_1a_{y}_transformed.csv",
        usecols=["document_control_number", "cas_number", "chemical_name", "primary_naics_code", "canonical_smiles", "trifid"],
        dtype=str,
    )
    b = pd.read_csv(f"{P}/tri_1b_{y}_transformed.csv", dtype=str)
    a.to_pickle(f"a_{y}.pkl")
    b.to_pickle(f"b_{y}.pkl")
    print(y, len(a), len(b), flush=True)
