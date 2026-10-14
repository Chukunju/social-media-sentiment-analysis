import joblib
import pandas as pd
from sklearn.metrics import f1_score
from preprocess import normalize

df = pd.read_csv("data/processed/label_me_nike_shoes.csv")
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
d = df[df["human_label"].isin(["positive", "neutral", "negative"])].copy()
pipe = joblib.load("models/baseline.joblib")
pred = pipe.predict(d["text"].apply(normalize))
print(f"Baseline on YouTube (n={len(d)}): macro-F1 = {f1_score(d['human_label'], pred, average='macro'):.4f}")
print(f"RoBERTa on same rows: 0.6326")
