# train/calibrate_v3_thresholds.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load
from sklearn.metrics import precision_recall_fscore_support


# -------- CONFIG --------

DATASET_PATH = Path("data/processed/dataset_v3_west.csv")
ARTIFACTS_DIR = Path("artifacts/v3")
THRESHOLDS_PATH = ARTIFACTS_DIR / "thresholds.json"

THRESHOLD_GRID = np.linspace(0.1, 0.9, 17)  # 0.10, 0.15, ..., 0.90


# -------- LOAD --------

def load_assets():
    vectorizer = load(ARTIFACTS_DIR / "vectorizer.joblib")
    model = load(ARTIFACTS_DIR / "model.joblib")
    return vectorizer, model


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    return pd.read_csv(DATASET_PATH)


# -------- CALIBRATION --------

def calibrate():
    print("Loading dataset and model...")
    df = load_dataset()
    vectorizer, model = load_assets()

    X = df["text"]
    y_true = df["label"].values

    print("Vectorizing and predicting probabilities...")
    X_vec = vectorizer.transform(X)
    y_score = model.predict_proba(X_vec)[:, 1]

    results = []

    print("\nEvaluating thresholds:")
    for t in THRESHOLD_GRID:
        y_pred = (y_score >= t).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )

        results.append({
            "threshold": round(float(t), 2),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1": round(float(f1), 4),
        })

        print(
            f"t={t:.2f} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f}"
        )

    # -------- CHOOSE THRESHOLDS --------
    # review_threshold: balance precision/recall
    # strong_signal_threshold: high precision

    df_res = pd.DataFrame(results)

    review_row = df_res.sort_values(
        by=["f1", "precision"], ascending=False
    ).iloc[0]

    strong_row = df_res[df_res["precision"] >= 0.85]
    if not strong_row.empty:
        strong_row = strong_row.sort_values(
            by=["precision", "threshold"], ascending=[False, True]
        ).iloc[0]
    else:
        strong_row = df_res.sort_values(
            by=["precision", "threshold"], ascending=[False, True]
        ).iloc[0]

    thresholds = {
        "review_threshold": review_row["threshold"],
        "strong_signal_threshold": strong_row["threshold"],
        "metrics": {
            "review": {
                "precision": review_row["precision"],
                "recall": review_row["recall"],
                "f1": review_row["f1"],
            },
            "strong_signal": {
                "precision": strong_row["precision"],
                "recall": strong_row["recall"],
                "f1": strong_row["f1"],
            },
        },
        "notes": "v3 is a signal-only model; thresholds used for policy severity",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(THRESHOLDS_PATH, "w") as f:
        json.dump(thresholds, f, indent=2)

    print("\nChosen thresholds:")
    print(json.dumps(thresholds, indent=2))
    print(f"\nSaved thresholds to: {THRESHOLDS_PATH.resolve()}")


if __name__ == "__main__":
    calibrate()
