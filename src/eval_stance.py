import sys
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from stance import add_stance

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv("data/processed/nike_relevance_eval.csv")
df = df[df["human_label"].isin(["positive", "neutral", "negative"])]
df = df[df["relevant"].astype(str).eq("True") & df["aspect"].notna() & (df["aspect"] != "other")]
df = add_stance("Nike shoes", df)
df = df[df["stance"].isin(["positive", "neutral", "negative"])]   # drop off_topic / missing
y = df["human_label"]
print(f"n = {len(df)} rows with a stance")
for name, p in (("pretrained RoBERTa", df["model_pred"].str.lower()), ("LLM stance", df["stance"])):
    f1 = f1_score(y, p, average="macro", zero_division=0)
    nr = recall_score(y, p, labels=["negative"], average=None, zero_division=0)[0]
    print(f"{name:20s} macro-F1 {f1:.3f}   negative recall {nr:.2f}")
