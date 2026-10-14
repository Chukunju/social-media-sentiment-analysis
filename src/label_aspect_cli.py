import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path, dtype=str).fillna("")
todo = df.index[df["aspect_ok"] == ""].tolist()
print(f"{len(todo)} left.  y = assigned aspect is right   n = wrong aspect   x = not an opinion about the product   s = skip   q = save and quit")
for i in todo:
    print(f"\n[{i}] ASPECT: {df.at[i, 'aspect']}\n    {df.at[i, 'text']}")
    while True:
        k = input("> ").strip().lower()
        if k in ("y", "n", "x", "s", "q"):
            break
    if k == "q":
        break
    if k != "s":
        df.at[i, "aspect_ok"] = k
    df.to_csv(path, index=False)
df.to_csv(path, index=False)
print("Saved.")
