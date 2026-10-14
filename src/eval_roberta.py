import os
import pandas as pd
import torch
from transformers import pipeline
from sklearn.metrics import classification_report, f1_score, recall_score
from preprocess import normalize

os.makedirs("reports", exist_ok=True)
test = pd.read_csv("data/raw/test.csv")
test["clean"] = test["text"].apply(normalize)

device = 0 if torch.cuda.is_available() else -1
print("Device:", "GPU" if device == 0 else "CPU (expect ~10-20 min)")

clf = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    device=device, truncation=True, max_length=64, batch_size=32,
)
out = clf(test["clean"].tolist())
test["pred"] = [o["label"].lower() for o in out]
test.to_csv("data/processed/test_roberta_preds.csv", index=False)

labels = ["negative", "neutral", "positive"]
print("\nTEST macro-F1:    ", round(f1_score(test["label_name"], test["pred"], average="macro"), 4))
print("TEST macro-recall:", round(recall_score(test["label_name"], test["pred"], average="macro"), 4))
print(classification_report(test["label_name"], test["pred"], labels=labels, digits=3))
