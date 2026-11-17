import os
import sys

import pandas as pd

root = "/Users/josehernandez/Documents/tri_chem_graphrag/data/raw"
ords = ["first", "second", "third", "fourth"]
for y in sys.argv[1:]:
    f = f"{root}/tri_2a_{y}.csv"
    hdr = pd.read_csv(f, nrows=0, encoding="latin-1").columns
    base = [
        "reporting_year",
        "trifid",
        "document_control_number",
        "cas_number",
        "chemical_name",
        "primary_naics_code",
        "facility_state",
    ]
    cols = [c for c in hdr if any(c.startswith(o + "_source_reduction_activity_") for o in ords) and ("method" not in c)]
    df = pd.read_csv(f, usecols=base + cols, dtype=str, encoding="latin-1")
    parts = []
    for o in ords:
        c = f"{o}_source_reduction_activity_code"
        if c not in df:
            continue
        d = df[base].copy()
        d["code"] = df[c]
        d["desc"] = df.get(f"{o}_source_reduction_activity_description")
        d["est"] = df.get(f"{c}_est_annual_reduction_code")
        parts.append(d[d.code.notna()])
    out = pd.concat(parts)
    out.to_csv(f"sr_{y}.csv", index=False)
    print(y, len(out), flush=True)
