import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv(sys.argv[1], dtype=str).fillna("")
df["human_label"] = df["human_label"].str.strip().str.lower()
df = df[df["human_label"].isin(["positive", "neutral", "negative", "irrelevant"])]
df["kept"] = df["filter_kept"] == "True"
df["on"] = df["human_label"] != "irrelevant"
k, d = df[df["kept"]], df[~df["kept"]]

print(f"Labeled: {len(df)} (kept sample {len(k)}, dropped sample {len(d)})")
print(f"Off-target among comments the filter KEPT:    {(~k['on']).mean():.1%}  (n={len(k)})")
print(f"On-target among comments the filter DROPPED:  {d['on'].mean():.1%}  (n={len(d)})")

kk = k[k["on"]]
y, p = kk["human_label"], kk["model_pred"]
print(f"\nSentiment model on kept, on-target comments (n={len(kk)}):")
print(f"macro-F1 = {f1_score(y, p, average='macro'):.4f}   accuracy = {(y == p).mean():.3f}")
print(classification_report(y, p, digits=3, zero_division=0))
print(pd.crosstab(y, p, rownames=["human"], colnames=["model"]))

rng = np.random.default_rng(42)
yv, pv = y.to_numpy(), p.to_numpy()
boots = [f1_score(yv[i], pv[i], average="macro")
         for i in (rng.integers(0, len(yv), len(yv)) for _ in range(1000))]
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\n95% bootstrap CI for macro-F1: [{lo:.3f}, {hi:.3f}]")
