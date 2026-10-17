import json
import os
import threading
import time
import pandas as pd
from pipeline import build_dataset, slugify, sentiment_model
from insights import generate_report

CACHE_DIR = "live_cache"
threading.Thread(target=sentiment_model, daemon=True).start()   # warm-up


def _path(brand, strict=False):
    return os.path.join(CACHE_DIR, slugify(brand) + ("_strict" if strict else "") + ".json")


def _videos(df):
    if "video_title" not in df.columns:
        return []
    rel = (df["relevant"].astype(str).str.lower().eq("true")
           if "relevant" in df.columns else pd.Series(True, index=df.index))
    g = (df.assign(_rel=rel).groupby("video_title")
         .agg(comments=("text", "size"), relevant=("_rel", "sum"))
         .reset_index().sort_values("comments", ascending=False))
    return [{"title": str(r.video_title), "comments": int(r.comments), "relevant": int(r.relevant)}
            for r in g.itertuples()]


def load_cached(brand, strict=False):
    try:
        with open(_path(brand, strict), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def list_cached():
    names, seen = [], set()
    if os.path.isdir(CACHE_DIR):
        for fn in sorted(os.listdir(CACHE_DIR)):
            try:
                with open(os.path.join(CACHE_DIR, fn), encoding="utf-8") as f:
                    b = json.load(f)["brand"]
            except Exception:
                continue
            if b.lower() not in seen:
                seen.add(b.lower())
                names.append(b)
    return names


def run_brand(brand, progress=None, strict=False):
    for d in (CACHE_DIR, "data/processed", "data/cache", "data/raw", "reports"):
        os.makedirs(d, exist_ok=True)
    df = build_dataset(brand, progress=progress, strict=strict)
    if progress:
        progress("Writing the report")
    stats, qdf, text, notes = generate_report(brand, df)
    result = {"brand": brand, "strict": strict,
              "created": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
              "stats": stats.to_dict("records"), "text": text, "notes": notes,
              "videos": _videos(df)}
    with open(_path(brand, strict), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result
