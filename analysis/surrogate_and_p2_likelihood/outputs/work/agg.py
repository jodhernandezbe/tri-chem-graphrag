import glob

import pandas as pd

df = pd.concat([pd.read_csv(f, dtype=str) for f in sorted(glob.glob("sr_*.csv"))])
df["year"] = df.reporting_year.astype(int)
df.to_pickle("sr_all.pkl")
print(
    "total",
    len(df),
    "chems",
    df.cas_number.nunique(),
    "facilities",
    df.trifid.nunique(),
    "reports",
    df.document_control_number.nunique(),
)
df["sys"] = df.code.str[0]
print(df.groupby("sys").size())
p = df[df.year < 2014]
q = df[df.year >= 2014]
for n, d in [("1987-2013", p), ("2014-2024", q)]:
    print(n, len(d), d.cas_number.nunique(), d.trifid.nunique(), d.document_control_number.nunique())
q2 = q[q.est.isin(["R1", "R2", "R3"])]
print("2014-2024 R1-3", len(q2), q2.cas_number.nunique(), q2.trifid.nunique(), q2.document_control_number.nunique())
print("2014-2024 est present", q.est.notna().sum())
cat = (
    df.groupby(["code"])
    .agg(
        desc=("desc", lambda s: s.dropna().mode().iat[0] if s.notna().any() else ""),
        n=("code", "size"),
        y0=("year", "min"),
        y1=("year", "max"),
    )
    .reset_index()
)
cat.to_csv("code_catalog.csv", index=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)
pd.set_option("display.max_colwidth", 90)
print(cat[cat.n > 20])
