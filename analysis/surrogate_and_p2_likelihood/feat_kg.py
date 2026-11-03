import collections
import glob
import pickle

import pandas as pd

smi = {}
name = {}
naics = collections.defaultdict(collections.Counter)
cou = collections.defaultdict(collections.Counter)
nrep = collections.Counter()
for f in sorted(glob.glob("a_*.pkl")):
    if int(f[2:6]) < 2014:
        continue
    y = f[2:6]
    a = pd.read_pickle(f)
    b = pd.read_pickle(f"b_{y}.pkl")
    for c, s in a.dropna(subset=["canonical_smiles"]).drop_duplicates("cas_number")[["cas_number", "canonical_smiles"]].values:
        smi.setdefault(c, s)
    for c, n in a.drop_duplicates("cas_number")[["cas_number", "chemical_name"]].values:
        name.setdefault(c, n)
    a["n4"] = a.primary_naics_code.str[:4]
    for (c, n), k in a.groupby(["cas_number", "n4"]).size().items():
        naics[c][n] += k
    for c, k in a.groupby("cas_number").size().items():
        nrep[c] += k
    m = b.merge(
        a[["document_control_number", "cas_number"]].drop_duplicates("document_control_number"), on="document_control_number"
    )
    for (c, u), k in m.groupby(["cas_number", "chemical_activity"]).size().items():
        cou[c][u] += k
    del a, b, m
pickle.dump(dict(smi=smi, name=name, naics=dict(naics), cou=dict(cou), nrep=nrep), open("feat_kg.pkl", "wb"))
print(len(smi), len(name), len(naics), len(cou))
print(list(cou["79-01-6"].most_common(8)))
