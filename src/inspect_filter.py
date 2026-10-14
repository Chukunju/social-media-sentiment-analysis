import sys
import pandas as pd
from sklearn.metrics import f1_score

sys.stdout.reconfigure(encoding="utf-8")
df = pd.read_csv("data/processed/nike_relevance_eval.csv")
df["human_label"] = df["human_label"].astype(str).str.strip().str.lower()
on = df[df["human_label"] != "irrelevant"]
kept = on[on["relevant"]]
dropped = on[~on["relevant"]]

print("On-target comments by human label, kept vs dropped:")
print(pd.crosstab(on["relevant"].map({True: "kept", False: "dropped"}), on["human_label"]))
print("\nRoBERTa macro-F1:")
print(f"  all on-target rows (n={len(on)}):        {f1_score(on['human_label'], on['model_pred'], average='macro'):.4f}")
print(f"  rows the new filter keeps (n={len(kept)}): {f1_score(kept['human_label'], kept['model_pred'], average='macro'):.4f}")
print("\nDropped on-target comments (up to 12):")
for t, h in zip(dropped["text"].head(12), dropped["human_label"].head(12)):
    print(f"  [{h}] {str(t)[:110]}")
print("\nOff-target comments the new filter still KEPT:")
for t in df[(df["human_label"] == "irrelevant") & df["relevant"]]["text"]:
    print("  ", str(t)[:110])
