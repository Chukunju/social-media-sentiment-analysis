import os
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from collector import fetch_youtube_comments
from preprocess import normalize

ASPECTS = ["battery", "camera", "price", "screen", "display", "performance",
           "quality", "design", "software", "update", "charger", "charging",
           "sound", "lag", "worth", "build", "feature", "value", "cheap",
           "expensive", "broke", "durable", "comfortable", "fit", "size",
           "warranty", "service", "refund"]


def relevant(text, brand_tokens):
    t = text.lower()
    return any(b in t for b in brand_tokens) or any(a in t for a in ASPECTS)


def analyze(brand):
    slug = re.sub(r"\W+", "_", brand.lower()).strip("_")
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    df = fetch_youtube_comments(brand)
    n_all = len(df)
    tokens = [w.lower() for w in brand.split() if len(w) >= 3]
    df = df[df["text"].apply(lambda x: relevant(x, tokens))].copy()
    print(f"Relevance filter kept {len(df)} of {n_all} comments")

    clf = pipeline("text-classification",
                   model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                   truncation=True, max_length=128, batch_size=32)
    out = clf(df["text"].apply(normalize).tolist())
    df["sentiment"] = [o["label"].lower() for o in out]
    df["confidence"] = [round(o["score"], 3) for o in out]
    df.to_csv(f"data/processed/{slug}_scored.csv", index=False)

    share = df["sentiment"].value_counts(normalize=True).reindex(
        ["positive", "neutral", "negative"]).fillna(0).mul(100).round(1)
    print(f"\n=== {brand}: sentiment share (n={len(df)}) ===")
    print(share.to_string())

    for lab in ["positive", "negative"]:
        top = df[df["sentiment"] == lab].sort_values(
            ["confidence", "likes"], ascending=False).head(3)
        print(f"\nTop {lab} comments:")
        for _, r in top.iterrows():
            print(f"  [{r['confidence']}] {r['text'][:120]}")

    ax = share.plot(kind="bar", color=["#2e7d32", "#9e9e9e", "#c62828"])
    ax.set_title(f"{brand}: YouTube comment sentiment (n={len(df)})")
    ax.set_ylabel("% of comments")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"reports/{slug}_sentiment.png", dpi=150)
    plt.close()

    sample = df.sample(min(200, len(df)), random_state=42)[["text", "sentiment"]]
    sample = sample.rename(columns={"sentiment": "model_pred"})
    sample["human_label"] = ""
    sample.to_csv(f"data/processed/label_me_{slug}.csv", index=False)
    print(f"\nSaved scored data, chart, and label_me_{slug}.csv")


if __name__ == "__main__":
    analyze(" ".join(sys.argv[1:]) or "Samsung Galaxy")
