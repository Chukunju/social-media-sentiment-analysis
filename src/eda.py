import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("reports", exist_ok=True)
train = pd.read_csv("data/raw/train.csv")
val = pd.read_csv("data/raw/validation.csv")
test = pd.read_csv("data/raw/test.csv")

for name, df in [("train", train), ("validation", val), ("test", test)]:
    print(f"\n{name}: {df.shape}")
    print(df["label_name"].value_counts(normalize=True).round(3).to_dict())

train["n_words"] = train["text"].str.split().str.len()
print("\nWords per tweet:")
print(train["n_words"].describe().round(1))
print("\nDuplicates in train:", train["text"].duplicated().sum())
print("Tweets with @user:", train["text"].str.contains("@user").mean().round(3))
print("Tweets with http:", train["text"].str.contains("http").mean().round(3))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.countplot(data=train, x="label_name",
              order=["negative", "neutral", "positive"], ax=ax[0])
ax[0].set_title("Class balance (train)")
sns.histplot(data=train, x="n_words", hue="label_name", bins=30,
             element="step", ax=ax[1])
ax[1].set_title("Tweet length by class")
plt.tight_layout()
plt.savefig("reports/eda_tweeteval.png", dpi=150)
print("\nSaved reports/eda_tweeteval.png")
