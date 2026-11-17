"""Top-3 analogue agreement (Table 3 column) from anal_kg.pkl; supplements anal_kg.py/cases_kg.py."""

import json
import pickle

import numpy as np
from scipy.stats import wilcoxon

A = pickle.load(open("anal_kg.pkl", "rb"))
ch = A["chems"]
n = len(ch)
OP = A["OP"]
R = np.random.default_rng(42).random((n, n))
crit = {"struct": A["T"], "phys": A["PC"], "sector": A["SN"], "cou": A["SU"], "ctx": A["CTX"], "random": R}


def topk(M, i, k):
    s = M[i].copy()
    s[i] = -np.inf
    return np.argsort(-s)[:k]


out = {}
for k in (1, 3, 5):
    per = {nm: np.array([OP[i, topk(M, i, k)].mean() for i in range(n)]) for nm, M in crit.items()}
    out[f"k{k}"] = {nm: float(v.mean()) for nm, v in per.items()}
    if k == 3:
        c, s, p = per["ctx"], per["struct"], per["phys"]
        out["k3_tests"] = dict(
            wilcoxon_ctx_vs_struct_p=float(wilcoxon(c, s).pvalue),
            wilcoxon_ctx_vs_phys_p=float(wilcoxon(c, p).pvalue),
            share_ctx_better_than_struct=float(np.mean(c > s)),
            share_ctx_ge_struct=float(np.mean(c >= s)),
            share_ctx_better_than_phys=float(np.mean(c > p)),
        )
out["k3_different_topset_struct_vs_ctx"] = float(
    np.mean([set(topk(A["T"], i, 3)) != set(topk(A["CTX"], i, 3)) for i in range(n)])
)
out["k3_no_overlap_struct_vs_ctx"] = float(
    np.mean([not (set(topk(A["T"], i, 3)) & set(topk(A["CTX"], i, 3))) for i in range(n)])
)
out["top1_different"] = float(np.mean([topk(A["T"], i, 1)[0] != topk(A["CTX"], i, 1)[0] for i in range(n)]))
c1 = np.array([OP[i, topk(A["CTX"], i, 1)[0]] for i in range(n)])
s1 = np.array([OP[i, topk(A["T"], i, 1)[0]] for i in range(n)])
out["top1_share_ctx_better"] = float(np.mean(c1 > s1))
out["top1_wilcoxon_p"] = float(wilcoxon(c1, s1).pvalue)
# 1-BP
i = ch.index("106-94-5")
out["bp_k3_mean_OP"] = {nm: float(OP[i, topk(M, i, 3)].mean()) for nm, M in crit.items()}
print(json.dumps(out, indent=1))
json.dump(out, open("top3_analysis.json", "w"), indent=1)
