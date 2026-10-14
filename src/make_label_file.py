import sys
import pandas as pd
from pipeline import build_dataset, slugify

brand = " ".join(sys.argv[1:])
df = build_dataset(brand)
kept = df[df["relevant"]]
dropped = df[~df["relevant"]]
print(f"{len(df)} comments: {len(kept)} kept, {len(dropped)} dropped")

s = pd.concat([
    kept.sample(min(100, len(kept)), random_state=42).assign(filter_kept=True),
    dropped.sample(min(25, len(dropped)), random_state=42).assign(filter_kept=False),
]).sample(frac=1, random_state=7)
out = s[["text", "sentiment", "filter_kept"]].rename(columns={"sentiment": "model_pred"})
out["human_label"] = ""
path = f"data/processed/label_me_{slugify(brand)}_v2.csv"
out.to_csv(path, index=False)
print("Saved", path)
