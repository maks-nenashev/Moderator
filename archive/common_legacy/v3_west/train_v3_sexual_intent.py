import json
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from joblib import dump

# -------- CONFIG & ARGS --------

def get_args():
    """
    Standard argument parser for model versioning.
    v3.0: Base intent (word-level)
    v3.1: Hardcore slang & obfuscation (char-level)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="3.0", help="Model version: 3.0 or 3.1")
    return parser.parse_args()

def train():
    args = get_args()
    
    # 1. Формируем чистый путь версии (3.1 -> 3_1)
    v_name = args.version.replace(".", "_")
    # 2. Используем v_name напрямую в названии файла
    # Результат будет: data/processed/dataset_v3_1_west.csv
    DATASET_PATH = Path(f"data/processed/dataset_v{v_name}_west.csv")
   # 3. Директория артефактов остается с точкой (как в конфигах API)
    # Результат: artifacts/v3.1
    ARTIFACTS_DIR = Path(f"artifacts/v{args.version}")

    print(f"Training Session: v{args.version} (WEST BLOCK: EN, DE, FR, ES, NL, IT, PT)")
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}. Run dataset preparation first!")

    # -------- DATA LOADING --------
    df = pd.read_csv(DATASET_PATH)
    # Filling NaNs to prevent TfidfVectorizer from crashing on empty strings
    X = df["text"].astype(str).fillna("")
    y = df["label"].astype(int)

    # Stratified split to maintain class balance in both sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # -------- VECTORIZATION STRATEGY --------
    if args.version == "3.1":
        # CHAR-WB is critical for detecting obfuscated slang (e.g., "s-e-n-d n-u-d-e-s")
        # ngram_range (2, 6) captures short roots common in Germanic/Romance slang
        print("🛠 Using CHAR-WB vectorizer for Slang & Obfuscation detection...")
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(2, 6), 
            min_df=3,
            max_df=0.9,
            strip_accents="unicode",
            lowercase=True
        )
    else:
        # WORD-level for standard intent (v3.0 base model)
        print("🛠 Using WORD vectorizer for Base intent detection...")
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            strip_accents="unicode",
            lowercase=True
        )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    # -------- MODEL TRAINING --------
    print(f"Training Logistic Regression (Samples: {len(df)})...")
    # Logistic Regression with 'balanced' weights to handle any class distribution drift
    model = LogisticRegression(
        class_weight="balanced", 
        max_iter=1000, 
        n_jobs=-1, 
        C=1.0, 
        solver='liblinear' # Robust solver for smaller/medium datasets
    )
    model.fit(X_train_vec, y_train)

    # -------- PERFORMANCE METRICS --------
    y_val_proba = model.predict_proba(X_val_vec)[:, 1]
    auc = roc_auc_score(y_val, y_val_proba)
    print(f"✅ AUC Score: {auc:.4f}")

    # -------- EXPORTING ARTIFACTS --------
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    dump(model, ARTIFACTS_DIR / "model.joblib")

    # Automated threshold estimation: 98th percentile of negative class scores
    y_train_proba = model.predict_proba(X_train_vec)[:, 1]
    neg_scores = y_train_proba[y_train == 0]
    estimated_thr = float(pd.Series(neg_scores).quantile(0.98))

    # Meta information for reproducibility and policy layers
    meta = {
        "model_version": f"v{args.version}",
        "task": "sexual_intent_west",
        "model_type": "tfidf_logreg_char" if args.version == "3.1" else "tfidf_logreg_word",
        "auc": round(float(auc), 4),
        "estimated_98_percentile_threshold": round(estimated_thr, 4),
        "samples_total": len(df),
    }

    with open(ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Main policy threshold for deployment (Standardized at 0.70)
    with open(ARTIFACTS_DIR / "thresholds.json", "w") as f:
        json.dump({"review_threshold": 0.70}, f, indent=2)

    print(f"🔥 Model artifacts saved to {ARTIFACTS_DIR}")

if __name__ == "__main__":
    train()