import re
import sys
import pandas as pd
from insights import prep, aspect_stats

sys.stdout.reconfigure(encoding="utf-8")
Q = re.compile(r"\?|^\s*(any idea|when|what|how|is|are|does|do|can|could|which|why|where|would|will|should)\b", re.I)


def isq(t):
    return bool(Q.search(str(t)))


lab = pd.read_csv("data/processed/aspect_check_mahindra_thar.csv", dtype=str).fillna("")
d = lab[lab["aspect_ok"].isin(["y", "n", "x"])].copy()
d["flag"] = d["text"].apply(isq)
bad = d["aspect_ok"] != "y"
print(f"Labeled rows: {len(d)}  wrong/off-target: {bad.sum()}  flagged as question: {d['flag'].sum()}")
print(f"Flagged and wrong/off-target (correctly dropped): {(d['flag'] & bad).sum()}")
print(f"Flagged but aspect was right (collateral):        {(d['flag'] & ~bad).sum()}")
rest = d[~d["flag"]]
print(f"After dropping flagged: {len(rest)} left, aspect right {(rest['aspect_ok'] == 'y').mean():.1%}, off-target {(rest['aspect_ok'] == 'x').mean():.1%}")
print("\nCollateral (flagged but right):")
for t in d[d["flag"] & ~bad]["text"]:
    print("  ", t[:100])
print("\nStill wrong after the filter:")
for r in rest[rest["aspect_ok"] != "y"].itertuples():
    print(f"  [{r.aspect_ok}] ({r.aspect}) {r.text[:100]}")

full = prep(pd.read_csv("data/processed/mahindra_thar_v2.csv"))
a = aspect_stats(full).set_index("aspect")
b = aspect_stats(full[~full["text"].apply(isq)]).set_index("aspect")
k = full[full["relevant"] & full["aspect"].notna()].copy()
k["q"] = k["text"].apply(isq)
share = (k.groupby("aspect")["q"].mean() * 100).round().rename("pct_questions")
out = a[["mentions", "net", "verdict"]].join(b[["mentions", "net", "verdict"]], rsuffix="_noQ").join(share)
print("\nAspect table before vs after dropping question-like comments:")
print(out.to_string())
