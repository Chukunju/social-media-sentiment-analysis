import re
import sys
import pandas as pd

brand = " ".join(sys.argv[1:])
slug = re.sub(r"\W+", "_", brand.lower()).strip("_")
df = pd.read_csv(f"data/processed/{slug}_v2.csv")
df["relevant"] = df["relevant"].astype(str).str.lower().eq("true")
k = df[df["relevant"] & df["aspect"].notna()]
s = k.sample(min(60, len(k)), random_state=11)[["text", "aspect"]].copy()
s["aspect_ok"] = ""
path = f"data/processed/aspect_check_{slug}.csv"
s.to_csv(path, index=False)
print("Saved", path, "with", len(s), "rows")
