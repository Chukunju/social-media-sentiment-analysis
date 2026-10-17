import sys
import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics import f1_score, recall_score
from preprocess import normalize

sys.stdout.reconfigure(encoding="utf-8")
model_dir = sys.argv[1]
clf = pipeline("text-classification", model=model_dir, truncation=True, max_length=128, batch_size=32)
LABELS = ["positive", "neutral", "negative"]


def load(path, thar=False):
    df = pd.read_csv(path, dtype=str).fillna("")
    df["human_label"] = df["human_label"].str.strip().str.lower()
    df = df[df["human_label"].isin(LABELS)]
    if thar:
        df = df[df["filter_kept"] == "True"]
    return df.reset_index(drop=True)


def neg_recall(y, p):
    return recall_score(y, p, labels=["negative"], average=None, zero_division=0)[0]


def compare(name, df):
    y = df["human_label"].to_numpy()
    mine = np.array([o["label"].lower() for o in clf(df["text"].apply(normalize).tolist())])
    theirs = df["model_pred"].str.lower().to_numpy()

    def f1(p, i):
        return f1_score(y[i], p[i], average="macro", zero_division=0)

    all_i = np.arange(len(y))
    d0 = f1(mine, all_i) - f1(theirs, all_i)
    rng = np.random.default_rng(42)
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(f1(mine, i) - f1(theirs, i))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n{name} (n={len(y)})")
    print(f"  macro-F1:        mine {f1(mine, all_i):.4f}   pretrained {f1(theirs, all_i):.4f}")
    print(f"  negative recall: mine {neg_recall(y, mine):.2f}     pretrained {neg_recall(y, theirs):.2f}")
    print(f"  difference (mine - pretrained): {d0:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]")


compare("Nike", load("data/processed/label_me_nike_shoes.csv"))
compare("Mahindra Thar", load("data/processed/label_me_mahindra_thar_v2.csv", thar=True))
