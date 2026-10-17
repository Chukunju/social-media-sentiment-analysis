import sys
import pandas as pd
from classify import classify

sys.stdout.reconfigure(encoding="utf-8")
slug = sys.argv[1]
brand = " ".join(sys.argv[2:]) or slug.replace("_", " ")

df = pd.read_csv(f"data/processed/{slug}_scored.csv")
print("Rows in scored file:", len(df))
out = classify(brand, df)
out.to_csv(f"data/processed/{slug}_classified.csv", index=False)

rel = out[out["relevant"]]
print(f"\nRelevant: {len(rel)} of {len(out)}")
print(rel["aspect"].value_counts().to_string())

print("\n15 comments marked NOT relevant (check they really are):")
for t in out[~out["relevant"]]["text"].astype(str).str.slice(0, 140).head(15):
    print(" -", t)

print("\n10 comments assigned to 'other':")
for t in rel[rel["aspect"] == "other"]["text"].astype(str).str.slice(0, 140).head(10):
    print(" -", t)
