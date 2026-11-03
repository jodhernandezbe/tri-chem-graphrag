import collections
import json
import pickle

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from scipy.stats import spearmanr
from xwalk import C2F

RDLogger.DisableLog("rdApp.*")
F = pickle.load(open("feat.pkl", "rb"))
sr = pd.read_pickle("sr_all.pkl")
sr = sr[sr.code.str.match(r"^[WS]\d\d$")].copy()
sr["fam"] = sr.code.map(C2F)
MIN = 5
w = sr[sr.code.str.startswith("W")]  # native-code level, 1987-2020
cnt = w.groupby("cas_number").size()
chems = [c for c in cnt[cnt >= MIN].index if c in F["smi"]]
mols = {}
for c in chems:
    m = Chem.MolFromSmiles(F["smi"][c])
    if m is not None:
        mols[c] = m
chems = sorted(mols)
n = len(chems)
print("chemicals analysed", n)
fps = [AllChem.GetMorganFingerprintAsBitVect(mols[c], 2, 2048) for c in chems]
T = np.array([DataStructs.BulkTanimotoSimilarity(f, fps) for f in fps])


def desc(m):
    return [
        Descriptors.MolWt(m),
        Descriptors.MolLogP(m),
        Descriptors.TPSA(m),
        rdMolDescriptors.CalcNumHBD(m),
        rdMolDescriptors.CalcNumHBA(m),
        rdMolDescriptors.CalcNumRotatableBonds(m),
        m.GetNumHeavyAtoms(),
        rdMolDescriptors.CalcFractionCSP3(m),
        rdMolDescriptors.CalcNumAromaticRings(m),
        sum(a.GetSymbol() in ("F", "Cl", "Br", "I") for a in m.GetAtoms()),
        sum(a.GetSymbol() not in ("C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I") for a in m.GetAtoms()),
    ]


D = np.array([desc(mols[c]) for c in chems], float)
D = (D - D.mean(0)) / (D.std(0) + 1e-9)
dist = np.sqrt(((D[:, None, :] - D[None, :, :]) ** 2).sum(-1))
PC = 1 / (1 + dist)


def vec(dicts, keys):
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((n, len(keys)))
    for i, c in enumerate(chems):
        for k, v in dicts.get(c, {}).items():
            if k in idx:
                M[i, idx[k]] = v
    df = (M > 0).sum(0)
    idf = np.log((n + 1) / (df + 1)) + 1
    M = np.log1p(M) * idf
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M @ M.T, M


nk = sorted({k for c in chems for k in F["naics"].get(c, {})})
uk = sorted({k for c in chems for k in F["cou"].get(c, {})})
SN, _ = vec(F["naics"], nk)
SU, _ = vec(F["cou"], uk)
CTX = (SN + SU) / 2
prof = {c: collections.Counter(g) for c, g in w.groupby("cas_number").code if c in mols}
codes = sorted({k for p in prof.values() for k in p})
OP, OPM = vec(prof, codes)
iu = np.triu_indices(n, 1)
res = {}
for nm, M in [
    ("Structural (Morgan Tanimoto)", T),
    ("Physicochemical descriptors", PC),
    ("Industrial sector (NAICS)", SN),
    ("Condition of use (Form R Sec. 3)", SU),
    ("Context (NAICS+CoU)", CTX),
    ("Structural+Context", (T + CTX) / 2),
]:
    r = spearmanr(M[iu], OP[iu])
    res[nm] = (round(r.correlation, 3), r.pvalue)
print(json.dumps({k: [v[0], float(v[1])] for k, v in res.items()}, indent=1))
# surrogate selection experiment
sets = {c: set(prof[c]) for c in chems}
dfc = collections.Counter(k for c in chems for k in sets[c])
idf = {k: np.log(n / dfc[k]) for k in dfc}


def wcov(q, rec):
    tot = sum(idf[k] for k in sets[q])
    return sum(idf[k] for k in sets[q] if k in rec) / tot if tot > 0 else np.nan


def recommend(q, S, k_s=5, k=10):
    s = S[chems.index(q)].copy()
    s[chems.index(q)] = -np.inf
    top = np.argsort(-s)[:k_s]
    pool = collections.Counter()
    for j in top:
        for code, v in prof[chems[j]].items():
            pool[code] += v * max(s[j], 1e-6)
    return {c for c, _ in pool.most_common(k)}, [chems[j] for j in top]


rng = np.random.default_rng(42)
R = rng.random((n, n))
out = {}
for nm, M in [
    ("Structural", T),
    ("Physicochemical", PC),
    ("Sector (NAICS)", SN),
    ("Condition of use", SU),
    ("Context (NAICS+CoU)", CTX),
    ("Structural+Context", (T + CTX) / 2),
    ("Random", R),
    ("Oracle (true P2 profile)", OP),
]:
    for ks in (1, 5):
        v = [wcov(q, recommend(q, M, ks)[0]) for q in chems]
        out[(nm, ks)] = np.nanmean(v)
    v1 = [OP[chems.index(q)][chems.index(recommend(q, M, 1)[1][0])] for q in chems]
    out[(nm, "opsim_top1")] = np.mean(v1)
for k, v in out.items():
    print(k, round(v, 4))
pickle.dump(dict(chems=chems, T=T, PC=PC, SN=SN, SU=SU, CTX=CTX, OP=OP, prof=prof, res=res, out=out), open("anal.pkl", "wb"))
