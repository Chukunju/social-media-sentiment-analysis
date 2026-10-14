import os
import re
import pandas as pd
from transformers import pipeline as hf_pipeline
from collector import fetch_youtube_comments
from classify import classify
from preprocess import normalize

_clf = None


def sentiment_model():
    global _clf
    if _clf is None:
        _clf = hf_pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            truncation=True, max_length=128, batch_size=32)
    return _clf


def slugify(brand):
    return re.sub(r"\W+", "_", brand.lower()).strip("_")


def build_dataset(brand):
    """All collected comments with relevant, aspect, sentiment, confidence. Cached on disk."""
    os.makedirs("data/processed", exist_ok=True)
    path = f"data/processed/{slugify(brand)}_v2.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    df = classify(brand, fetch_youtube_comments(brand))
    df["sentiment"] = None
    df["confidence"] = None
    rel = df["relevant"]
    if rel.any():
        out = sentiment_model()(df.loc[rel, "text"].apply(normalize).tolist())
        df.loc[rel, "sentiment"] = [o["label"].lower() for o in out]
        df.loc[rel, "confidence"] = [round(o["score"], 3) for o in out]
    df.to_csv(path, index=False)
    print(f"{len(df)} comments collected, {int(rel.sum())} kept by the relevance classifier")
    return df
