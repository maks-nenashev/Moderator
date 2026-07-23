import pandas as pd

# =========================
# Load Jigsaw dataset
# =========================
JIGSAW_PATH = "data/raw/jigsaw_train.csv"
OUT_PATH = "data/processed/dataset_v1.csv"

print("Loading Jigsaw dataset...")
df = pd.read_csv(JIGSAW_PATH)

# =========================
# Normalize text column
# =========================
df["text"] = df["comment_text"].astype(str)

# =========================
# Map Jigsaw labels → our labels
# =========================
df["toxicity"] = ((df["toxic"] == 1) | (df["insult"] == 1)).astype(int)
df["hate"] = (df["identity_hate"] == 1).astype(int)
df["sexual"] = (df["obscene"] == 1).astype(int)
df["violence"] = (df["threat"] == 1).astype(int)

# Not present in Jigsaw
df["scam"] = 0
df["spam"] = 0

# =========================
# Select final columns
# =========================
final_cols = [
    "text",
    "toxicity",
    "hate",
    "sexual",
    "violence",
    "scam",
    "spam"
]

df_final = df[final_cols]

# =========================
# Basic sanity checks
# =========================
print("Dataset shape:", df_final.shape)
print("Label distribution:")
print(df_final[final_cols[1:]].mean())

# =========================
# Save processed dataset
# =========================
df_final.to_csv(OUT_PATH, index=False)
print(f"Saved processed dataset to {OUT_PATH}")
