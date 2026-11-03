import collections
import pickle

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from xwalk import C2F

A = pickle.load(open("anal.pkl", "rb"))
sr = pd.read_pickle("sr_all.pkl")
sr = sr[sr.code.str.match(r"^[WS]\d\d$")].copy()
sr["fam"] = sr.code.map(C2F)
chems = A["chems"]
n = len(chems)
ix = {c: i for i, c in enumerate(chems)}
w = sr[sr.code.str.startswith("W") & sr.cas_number.isin(chems)]
print("W records used", len(w), "years", w.year.min(), w.year.max(), "facilities", w.trifid.nunique())


def opsim(df, key):
    M = np.zeros((n, 0))
    keys = sorted(df[key].dropna().unique())
    kx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((n, len(keys)))
    for (c, k), v in df.groupby(["cas_number", key]).size().items():
        if c in ix:
            M[ix[c], kx[k]] = v
    dfq = (M > 0).sum(0)
    M = np.log1p(M) * (np.log((n + 1) / (dfq + 1)) + 1)
    nr = np.linalg.norm(M, axis=1, keepdims=True)
    ok = nr[:, 0] > 0
    M = M / (nr + 1e-12)
    return M @ M.T, ok


iu = np.triu_indices(n, 1)
for lab, df, key in [
    ("family all yrs", sr, "fam"),
    ("codes 2014-2024 (KG window)", sr[sr.year >= 2014], "code"),
    ("family 2014-2024", sr[sr.year >= 2014], "fam"),
]:
    OP, ok = opsim(df[df.cas_number.isin(chems)], key)
    m = ok[iu[0]] & ok[iu[1]]
    print(
        lab,
        "n chems",
        ok.sum(),
        {
            nm: round(spearmanr(M[iu][m], OP[iu][m]).correlation, 3)
            for nm, M in [("T", A["T"]), ("PC", A["PC"]), ("CTX", A["CTX"])]
        },
    )
