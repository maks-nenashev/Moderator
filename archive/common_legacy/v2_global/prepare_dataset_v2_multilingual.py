import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
RAW_PATH = Path("data/raw/jigsaw_multilingual.csv")
OUT_PATH = Path("data/processed/dataset_v2_multilingual.csv")

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# =========================
# Load dataset
# =========================
print("Loading multilingual dataset...")
df = pd.read_csv(RAW_PATH)

print("Original shape:", df.shape)
print("Columns:", list(df.columns))

# =========================
# Expected columns (Jigsaw multilingual)
# =========================
# text column usually named: "comment_text" or "text"
TEXT_COL = "comment_text"

# These are standard Jigsaw labels
LABEL_MAP = {
    "toxicity": "toxic",
    "hate": "hate",
    "sexual": "sexual",
    "violence": "violence",
    # scam / spam отсутствуют — оставим нулями
}

# =========================
# Normalize column names
# =========================
df = df.rename(columns=str.lower)

if TEXT_COL not in df.columns:
    raise ValueError(f"Text column '{TEXT_COL}' not found in dataset")

# =========================
# Build unified dataset
# =========================
out = pd.DataFrame()
out["text"] = df[TEXT_COL].astype(str)

for target, source in LABEL_MAP.items():
    if source in df.columns:
        out[target] = df[source].fillna(0).astype(int)
    else:
        out[target] = 0

# Missing labels — explicitly set to 0
out["scam"] = 0
out["spam"] = 0

# =========================
# Basic cleanup
# =========================
out = out.dropna(subset=["text"])
out = out[out["text"].str.len() > 5]

# =========================
# Stats
# =========================
print("\nFinal dataset shape:", out.shape)
print("\nLabel distribution:")
print(out[["toxicity", "hate", "sexual", "violence", "scam", "spam"]].mean())

# =========================
# Save
# =========================
out.to_csv(OUT_PATH, index=False)
print(f"\nSaved multilingual dataset to {OUT_PATH}")
