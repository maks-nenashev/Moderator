# train/train_v3_sexual_intent.py

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump


# -------- CONFIG --------

DATASET_PATH = Path("data/processed/dataset_v3_west.csv")
ARTIFACTS_DIR = Path("artifacts/v3")

RANDOM_STATE = 42


# -------- LOAD DATA --------

def load_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_cols = {"text", "label", "language", "country_group"}
    if not required_cols.issubset(df.columns):
        raise ValueError("Dataset missing required columns")

    return df


# -------- TRAINING --------

def train():
    print("Loading dataset...")
    df = load_dataset()

    X = df["text"]
    y = df["label"]

    print(f"Total samples: {len(df)}")
    print("Label distribution:")
    print(y.value_counts(normalize=True))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nVectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print("Training Logistic Regression...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )

    model.fit(X_train_vec, y_train)

    print("\nEvaluating model...")
    y_val_pred = model.predict(X_val_vec)
    y_val_proba = model.predict_proba(X_val_vec)[:, 1]

    print("\nClassification report:")
    print(classification_report(y_val, y_val_pred, digits=4))

    auc = roc_auc_score(y_val, y_val_proba)
    print(f"AUC: {auc:.4f}")

    # -------- SAVE ARTIFACTS --------

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    dump(vectorizer, ARTIFACTS_DIR / "vectorizer.joblib")
    dump(model, ARTIFACTS_DIR / "model.joblib")

    meta = {
        "version": "v3.0.0",
        "task": "sexual_intent",
        "language_group": "WEST",
        "model_type": "tfidf_logreg",
        "auc": round(float(auc), 4),
        "samples": len(df),
    }

    with open(ARTIFACTS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nArtifacts saved:")
    print(f"- {ARTIFACTS_DIR / 'vectorizer.joblib'}")
    print(f"- {ARTIFACTS_DIR / 'model.joblib'}")
    print(f"- {ARTIFACTS_DIR / 'meta.json'}")


if __name__ == "__main__":
    train()
