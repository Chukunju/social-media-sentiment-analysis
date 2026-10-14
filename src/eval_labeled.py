import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, f1_score,
                             confusion_matrix)

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/label_me_nike_shoes.csv"
df = pd.read_csv(path)
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
df = df[df["human_label"].isin(["positive", "neutral", "negative", "irrelevant"])]
print(f"Labeled rows: {len(df)}")

irr = (df["human_label"] == "irrelevant").mean()
print(f"Irrelevant (survived keyword filter but off-target): {irr:.1%}")

d = df[df["human_label"] != "irrelevant"].copy()
labels = ["negative", "neutral", "positive"]
y, p = d["human_label"], d["model_pred"]

print(f"\nOn-target rows: {len(d)}")
print("Human label distribution:", y.value_counts(normalize=True).round(3).to_dict())
f1 = f1_score(y, p, average="macro")
print(f"\nYouTube macro-F1: {f1:.4f}   (TweetEval test: 0.7245)")
print(f"Accuracy: {(y == p).mean():.3f}")
print(classification_report(y, p, labels=labels, digits=3, zero_division=0))
print("Confusion matrix (rows=human, cols=model):")
print(pd.DataFrame(confusion_matrix(y, p, labels=labels),
                   index=labels, columns=labels))

rng = np.random.default_rng(42)
yv, pv = y.to_numpy(), p.to_numpy()
boots = []
for _ in range(1000):
    idx = rng.integers(0, len(yv), len(yv))
    boots.append(f1_score(yv[idx], pv[idx], average="macro"))
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"\n95% bootstrap CI for macro-F1: [{lo:.3f}, {hi:.3f}]")
