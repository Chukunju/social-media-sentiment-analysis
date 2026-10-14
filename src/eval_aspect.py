import sys
import numpy as np
import pandas as pd


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


df = pd.read_csv(sys.argv[1], dtype=str).fillna("")
d = df[df["aspect_ok"].isin(["y", "n", "x"])]
n = len(d)
y, w, x = [(d["aspect_ok"] == c).sum() for c in ("y", "n", "x")]
print(f"Labeled: {n}")
lo, hi = wilson(y, n)
print(f"Aspect right (of all labeled):        {y}/{n} = {y / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})")
if y + w:
    lo, hi = wilson(y, y + w)
    print(f"Aspect right (of on-topic comments):  {y}/{y + w} = {y / (y + w):.1%}  (95% CI {lo:.0%}-{hi:.0%})")
lo, hi = wilson(x, n)
print(f"Off-target (should have been dropped): {x}/{n} = {x / n:.1%}  (95% CI {lo:.0%}-{hi:.0%})")
print("\nBy assigned aspect (y / n / x):")
print(pd.crosstab(d["aspect"], d["aspect_ok"]))
print("\nWrong or off-target examples:")
for r in d[d["aspect_ok"].isin(["n", "x"])].itertuples():
    print(f"  [{r.aspect_ok}] ({r.aspect}) {r.text[:100]}")

