import re


def normalize(text):
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    return re.sub(r"\s+", " ", text).strip()
