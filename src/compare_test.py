import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score

ref = pd.read_csv("data/processed/test_roberta_preds.csv")
mine = pd.read_csv("data/processed/mine_test_preds.csv")
assert len(ref) == len(mine), "length mismatch"
y = ref["label_name"].to_numpy()
a = ref["pred"].str.lower().to_numpy()   # pretrained cardiffnlp
b = mine["pred"].to_numpy()              # your model


def mf1(p, i):
    return f1_score(y[i], p[i], average="macro")


idx = np.arange(len(y))
print(f"macro-F1: mine {mf1(b, idx):.4f}  pretrained {mf1(a, idx):.4f}")
print(f"negative recall: mine {recall_score(y, b, labels=['negative'], average=None)[0]:.3f}  "
      f"pretrained {recall_score(y, a, labels=['negative'], average=None)[0]:.3f}")
rng = np.random.default_rng(42)
d = []
for _ in range(300):
    i = rng.integers(0, len(y), len(y))
    d.append(mf1(b, i) - mf1(a, i))
lo, hi = np.percentile(d, [2.5, 97.5])
print(f"difference (mine - pretrained): {mf1(b, idx) - mf1(a, idx):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]")
