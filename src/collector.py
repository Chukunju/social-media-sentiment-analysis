import os
import re
import sys
import html
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from langdetect import detect_langs, DetectorFactory
DetectorFactory.seed = 0


def is_english(text):
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return False
    if sum(ch.isascii() for ch in letters) / len(letters) < 0.9:
        return False  # drops Devanagari and other scripts
    try:
        top = detect_langs(text)[0]
        return top.lang == "en" and top.prob > 0.8
    except Exception:
        return False

load_dotenv()
yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))


def clean_comments(df):
    df = df.copy()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True).str.strip()
    df = df[~df["text"].str.contains(r"https?://|www\.", regex=True, na=False)]
    df = df[df["text"].str.split().str.len() >= 4]
    df = df.drop_duplicates(subset="text")
    df = df[df["text"].apply(is_english)]
    return df.reset_index(drop=True)


def fetch_youtube_comments(brand, n_videos=8, per_video=100,
                           query_suffix="review", use_cache=True):
    os.makedirs("data/raw", exist_ok=True)
    slug = re.sub(r"\W+", "_", brand.lower()).strip("_")
    path = f"data/raw/youtube_{slug}.csv"
    if use_cache and os.path.exists(path):
        print(f"Loaded cached data for '{brand}'")
        return pd.read_csv(path)

    videos = yt.search().list(
        q=f"{brand} {query_suffix}", part="snippet", type="video",
        maxResults=n_videos, relevanceLanguage="en",regionCode="US", order="relevance",
    ).execute()["items"]

    rows = []
    for v in videos:
        vid = v["id"]["videoId"]
        title = html.unescape(v["snippet"]["title"])
        try:
            resp = yt.commentThreads().list(
                part="snippet", videoId=vid, maxResults=min(per_video, 100),
                order="relevance", textFormat="plainText",
            ).execute()
        except HttpError as e:
            print(f"Skipping {vid} (HTTP {e.resp.status}, comments likely disabled)")
            continue
        for c in resp["items"]:
            s = c["snippet"]["topLevelComment"]["snippet"]
            rows.append({
                "brand": brand,
                "source": "youtube",
                "text": s["textDisplay"],
                "date": s["publishedAt"],
                "likes": s["likeCount"],
                "video_title": title,
            })

    if not rows:
        raise ValueError(f"No comments could be collected for '{brand}'. Check the spelling or try a more popular name.")
    df = clean_comments(pd.DataFrame(rows))
    df.to_csv(path, index=False)
    print(f"Collected {len(df)} clean comments for '{brand}' from {len(videos)} videos")
    return df


if __name__ == "__main__":
    brand = " ".join(sys.argv[1:]) or "iPhone 17"
    df = fetch_youtube_comments(brand)
    print(df[["text", "likes"]].head(10).to_string())
