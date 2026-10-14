from datasets import load_dataset

ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
label_map = {0: "negative", 1: "neutral", 2: "positive"}

for split in ["train", "validation", "test"]:
    df = ds[split].to_pandas()
    df["label_name"] = df["label"].map(label_map)
    df.to_csv(f"data/raw/{split}.csv", index=False)
    print(split, df.shape)
    print(df["label_name"].value_counts(normalize=True).round(3).to_dict())
