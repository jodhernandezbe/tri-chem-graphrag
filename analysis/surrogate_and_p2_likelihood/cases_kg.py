import collections
import json
import pickle

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from xwalk import C2F, FAM

A = pickle.load(open("anal_kg.pkl", "rb"))
F = pickle.load(open("feat_kg.pkl", "rb"))
sr = pd.read_pickle("sr_all.pkl")
sr = sr[sr.year >= 2014]
sr = sr[sr.code.str.match(r"^[WS]\d\d$")].copy()
sr["fam"] = sr.code.map(C2F)
chems = A["chems"]
ix = {c: i for i, c in enumerate(chems)}
OP = A["OP"]


def top1(M, q):
    s = M[ix[q]].copy()
    s[ix[q]] = -np.inf
    return int(np.argmax(s))


st = [OP[ix[q], top1(A["T"], q)] for q in chems]
pc = [OP[ix[q], top1(A["PC"], q)] for q in chems]
cx = [OP[ix[q], top1(A["CTX"], q)] for q in chems]
print(
    "wilcoxon ctx vs struct",
    wilcoxon(cx, st).pvalue,
    "ctx vs pc",
    wilcoxon(cx, pc).pvalue,
    "share ctx better",
    np.mean(np.array(cx) > np.array(st)),
)
# how often structural top1 != context top1
print("different top1", np.mean([top1(A["T"], q) != top1(A["CTX"], q) for q in chems]))
# success likelihood 2014-2024
e = sr[(sr.year >= 2014) & sr.est.isin(["R1", "R2", "R3", "R4", "R5", "R6"])].copy()
e["hit"] = e.est.isin(["R1", "R2", "R3"])
fam = e.groupby("fam").agg(n=("hit", "size"), p=("hit", "mean"))
print(fam)
print("overall", len(e), e.hit.mean())
TS = {
    "79-01-6": "Trichloroethylene",
    "127-18-4": "Tetrachloroethylene",
    "75-09-2": "Methylene chloride",
    "106-94-5": "1-Bromopropane",
    "56-23-5": "Carbon tetrachloride",
    "123-91-1": "1,4-Dioxane",
    "872-50-4": "NMP",
    "106-99-0": "1,3-Butadiene",
    "50-00-0": "Formaldehyde",
    "115-96-8": "TCEP",
    "75-01-4": "Vinyl chloride",
}


def nm(c):
    return F["name"].get(c, c)


cases = {}
for c, n in TS.items():
    d = sr[sr.cas_number == c]
    ee = e[e.cas_number == c]
    r = dict(
        name=n,
        in_panel=c in ix,
        records=len(d),
        years=f"{d.year.min()}-{d.year.max()}" if len(d) else "",
        facilities=d.trifid.nunique(),
        top_acts=[(k, v) for k, v in d.desc.value_counts().head(5).items()],
        fam_share={k: round(v, 2) for k, v in d.fam.value_counts(normalize=True).items()},
        est_n=len(ee),
        est_hit=round(ee.hit.mean(), 2) if len(ee) else None,
        est_by_act=[(a, len(g), round(g.hit.mean(), 2)) for a, g in ee.groupby("desc") if len(g) >= 3],
    )
    if c in ix:
        for lab, M in [("struct", A["T"]), ("phys", A["PC"]), ("ctx", A["CTX"])]:
            s = M[ix[c]].copy()
            s[ix[c]] = -np.inf
            t = np.argsort(-s)[:3]
            r[lab] = [(nm(chems[j]), round(float(M[ix[c], j]), 2), round(float(OP[ix[c], j]), 2)) for j in t]
    cases[c] = r
json.dump(cases, open("cases_kg.json", "w"), indent=1, default=str)
[print(v["name"], v.get("records"), v.get("struct"), v.get("phys"), v.get("ctx")) for v in cases.values()]
