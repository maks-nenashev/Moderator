import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump

# -------- CONFIG & ARGS --------

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="4.0", help="Model version: 4.0 (base) or 4.1 (slang)")
    return parser.parse_args()

def train():
    args = get_args()
    
    # Динамические пути в зависимости от версии
    suffix = "4_1" if args.version == "4.1" else "4"
    DATASET_PATH = Path(f"data/processed/dataset_v4_{suffix}_cee.csv")
    ARTIFACTS_DIR = Path(f"artifacts/v{args.version}")

    print(f"🚀 Training Session: v{args.version}")
    print(f"📂 Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}. Run preparation first!")

    # -------- LOAD DATA --------
    df = pd.read_csv(DATASET_PATH)
    X = df["text"].astype(str)
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # -------- VECTORIZATION (Logic split) --------
    # Если это 4.1 — включаем посимвольный анализ для сленга
    if args.version == "4.1":
        print("🛠 Using CHAR-WB vectorizer for Slang detection...")
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(2, 5), # Ловим корни от 2 до 5 символов
            min_df=3,
            max_df=0.9,
            strip_accents="unicode",
            lowercase=True
        )
    else:
        print("🛠 Using WORD vectorizer for Base intent...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True
        )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # -------- TRAINING --------
    print(f"Training Logistic Regression (Samples: {len(df)})...")
    model = LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1, C=1.0)
    model.fit(X_train_vec, y_train)

    # -------- EVALUATION --------
    y_val_proba = model.predict_proba(X_val_vec)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    print(f"✅ AUC: {auc:.4f}")

    # -------- SAVE --------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    dump(model, ARTIFACTS_DIR / "model.joblib")

    meta = {
        "model_version": f"v{args.version}",
        "task": "sexual_intent",
        "language_group": "CEE",
        "model_type": "tfidf_logreg_char" if args.version == "4.1" else "tfidf_logreg_word",
        "auc": round(float(auc), 4),
        "samples": len(df),
    }

    with open(ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"🔥 Model v{args.version} saved to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train()