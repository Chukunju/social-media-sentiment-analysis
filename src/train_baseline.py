import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (classification_report, f1_score,
                             recall_score, ConfusionMatrixDisplay)
from preprocess import normalize

os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

train = pd.read_csv("data/raw/train.csv").drop_duplicates(subset="text")
val = pd.read_csv("data/raw/validation.csv")
test = pd.read_csv("data/raw/test.csv")
for df in (train, val, test):
    df["clean"] = df["text"].apply(normalize)

labels = ["negative", "neutral", "positive"]
best = None
for C in [0.1, 0.3, 1, 3, 10]:
    pipe = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                        token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
        LogisticRegression(C=C, max_iter=1000, class_weight="balanced"),
    )
    pipe.fit(train["clean"], train["label_name"])
    f1 = f1_score(val["label_name"], pipe.predict(val["clean"]), average="macro")
    print(f"C={C:<4} val macro-F1={f1:.4f}")
    if best is None or f1 > best[0]:
        best = (f1, C, pipe)

f1_val, C, pipe = best
print(f"\nBest C={C} (val macro-F1={f1_val:.4f})")

pred = pipe.predict(test["clean"])
print("\nTEST macro-F1:    ", round(f1_score(test["label_name"], pred, average="macro"), 4))
print("TEST macro-recall:", round(recall_score(test["label_name"], pred, average="macro"), 4))
print(classification_report(test["label_name"], pred, labels=labels, digits=3))

ConfusionMatrixDisplay.from_predictions(
    test["label_name"], pred, labels=labels, normalize="true", cmap="Blues")
plt.title("Baseline (TF-IDF + LogReg), test set")
plt.savefig("reports/baseline_confusion.png", dpi=150, bbox_inches="tight")

joblib.dump(pipe, "models/baseline.joblib")
print("Saved models/baseline.joblib and reports/baseline_confusion.png")
