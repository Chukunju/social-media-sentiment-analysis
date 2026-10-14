import os
from dotenv import load_dotenv
from googleapiclient.discovery import build

load_dotenv()
yt = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

# Find videos about a brand
videos = yt.search().list(
    q="iPhone review", part="snippet", type="video", maxResults=3
).execute()

for v in videos["items"]:
    vid = v["id"]["videoId"]
    print("VIDEO:", v["snippet"]["title"])
    comments = yt.commentThreads().list(
        part="snippet", videoId=vid, maxResults=3, textFormat="plainText"
    ).execute()
    for c in comments["items"]:
        print("  -", c["snippet"]["topLevelComment"]["snippet"]["textDisplay"][:100])
