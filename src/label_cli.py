import sys
import pandas as pd

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/label_me_nike_shoes.csv"
df = pd.read_csv(path, dtype=str).fillna("")
keys = {"p": "positive", "n": "neutral", "g": "negative", "x": "irrelevant"}

todo = df.index[df["human_label"] == ""].tolist()
print(f"{len(todo)} left. Keys: p=positive  n=neutral  g=negative  x=irrelevant  s=skip  q=save+quit")
print("Judge sentiment toward the BRAND/PRODUCT only.\n")

for i in todo:
    print(f"[{i}] {df.at[i, 'text']}")
    while True:
        k = input("> ").strip().lower()
        if k in keys or k in ("s", "q"):
            break
    if k == "q":
        break
    if k in keys:
        df.at[i, "human_label"] = keys[k]
    df.to_csv(path, index=False)

df.to_csv(path, index=False)
print(f"Saved. Labeled so far: {(df['human_label'] != '').sum()} of {len(df)}")
