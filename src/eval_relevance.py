import sys
import pandas as pd
from classify import classify

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv("data/processed/label_me_nike_shoes.csv")
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
df = df[df["human_label"].isin(["positive", "neutral", "negative", "irrelevant"])]
df = classify("Nike shoes", df)
df.to_csv("data/processed/nike_relevance_eval.csv", index=False)

truth = df["human_label"] != "irrelevant"
kept = df["relevant"]
print(f"\nRows: {len(df)}  (human-irrelevant: {(~truth).sum()})")
print(f"Old filter irrelevant rate:        {(~truth).mean():.1%}")
print(f"New filter irrelevant rate:        {(~truth[kept]).mean():.1%}  (kept {kept.sum()} rows)")
print(f"Irrelevant comments removed:       {(~kept[~truth]).mean():.1%}")
print(f"On-target comments kept (recall):  {kept[truth].mean():.1%}")
