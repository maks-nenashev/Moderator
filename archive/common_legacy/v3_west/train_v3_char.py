# 
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# ======================================================
# Paths
# ======================================================
DATA_PATH = Path("data/processed/dataset_v3_west.csv")

ARTIFACTS_DIR = Path("artifacts/v3")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"

# ======================================================
# Load dataset
# ======================================================
print("[v3] Loading dataset...")
df = pd.read_csv(DATA_PATH)

required_cols = {"text", "label", "language", "country_group"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

texts = df["text"].astype(str).tolist()
y = df["label"].astype(int).values

print(f"[v3] Samples: {len(texts)} | Positives: {y.sum()}")

# ======================================================
# Train / validation split
# ======================================================
X_train, X_val, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ======================================================
# Vectorizer (CHAR N-GRAMS — CRITICAL)
# ======================================================
print("[v3] Fitting char-level TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=2,
    lowercase=True,
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

print(f"[v3] Vectorizer vocab size: {len(vectorizer.vocabulary_)}")

# ======================================================
# Model
# ======================================================
print("[v3] Training LogisticRegression...")

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train_vec, y_train)

# ======================================================
# Evaluation
# ======================================================
print("[v3] Evaluating...")

y_val_pred = model.predict(X_val_vec)
y_val_prob = model.predict_proba(X_val_vec)[:, 1]

print(classification_report(y_val, y_val_pred, digits=3))

try:
    auc = roc_auc_score(y_val, y_val_prob)
    print(f"[v3] ROC-AUC: {auc:.4f}")
except ValueError:
    print("[v3] ROC-AUC not computable (single class in validation)")

# ======================================================
# Save artifacts
# ======================================================
print("[v3] Saving artifacts...")

joblib.dump(model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

meta = {
    "model_version": "v3.1.0",
    "task": "sexual_intent",
    "language_group": "WEST",
    "model_type": "char_tfidf_logreg",
    "ngram_range": [3, 5],
    "analyzer": "char_wb",
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("[v3] DONE.")
print(f"[v3] Model saved to: {MODEL_PATH}")
print(f"[v3] Vectorizer saved to: {VECTORIZER_PATH}")
print(f"[v3] Meta saved to: {META_PATH}")
