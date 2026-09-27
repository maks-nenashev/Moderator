#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dynamic Threshold Recalibration & True Hold-Out RQ1 Generator
Path: scripts/recalibrate_rq1_thresholds.py
Author: Maksym Nenashev (Systems Engineer)

Recalibrates thresholds on TRAIN-negatives (FPR_train <= 0.01) to prevent FPR=1.0000.
Re-evaluates 22 engines strictly on 20% Hold-out Test split.
"""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"

DATASET_MAPPING = [
    {
        "cluster": "Human Trafficking",
        "engine_base": "v1",
        "file": "trafficking_v1.csv",
    },
    {
        "cluster": "Western European",
        "engine_base": "v3",
        "file": "west_v3_base.csv",
    },
    {
        "cluster": "Central & Eastern Europe",
        "engine_base": "v4",
        "file": "cee_v4_base.csv",
    },
    {"cluster": "Baltic", "engine_base": "v5", "file": "baltic_v5_base.csv"},
    {"cluster": "CIS", "engine_base": "v6", "file": "cis_v6_base.csv"},
    {"cluster": "Nordic", "engine_base": "v7", "file": "nordic_v7_base.csv"},
    {"cluster": "Balkan", "engine_base": "v8", "file": "balkan_v8_base.csv"},
    {"cluster": "Caucasus", "engine_base": "v9", "file": "caucasus_v9_base.csv"},
]

ALL_ENGINES = [
    "v1",
    "v3",
    "v3.1",
    "v3.2",
    "v4",
    "v4.1",
    "v4.2",
    "v5",
    "v5.1",
    "v5.2",
    "v6",
    "v6.1",
    "v6.2",
    "v7",
    "v7.1",
    "v7.2",
    "v8",
    "v8.1",
    "v8.2",
    "v9",
    "v9.1",
    "v9.2",
]


def wilson_ci(p, n, z=1.96):
  if n == 0:
    return 0.0, 0.0
  denom = 1 + z**2 / n
  centre = p + z**2 / (2 * n)
  err = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
  return max(0.0, (centre - err) / denom), min(1.0, (centre + err) / denom)


def main():
  print("==================================================================")
  print("🛠️ RECALIBRATING THRESHOLDS & GENERATING VALID RQ1 PROOFS")
  print("==================================================================")

  rq1_proofs = []

  for eng in ALL_ENGINES:
    base_key = eng.split(".")[0]
    csv_info = next(
        (x for x in DATASET_MAPPING if x["engine_base"] == base_key), None
    )
    if not csv_info:
      continue

    fpath = DATA_DIR / csv_info["file"]
    if not fpath.exists():
      continue

    df_raw = pd.read_csv(fpath, on_bad_lines="skip", engine="python")
    df_clean = df_raw.dropna(subset=["text", "label"]).copy()
    df_clean["text"] = df_clean["text"].astype(str).str.strip()
    df_clean = df_clean.drop_duplicates(subset=["text"])

    y = df_clean["label"].values.astype(int)
    train_df, test_df = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=y
    )

    m_path = ARTIFACTS_DIR / eng / "model.joblib"
    v_path = ARTIFACTS_DIR / eng / "vectorizer.joblib"
    t_path = ARTIFACTS_DIR / eng / "thresholds.json"

    if not (m_path.exists() and v_path.exists()):
      continue

    model = joblib.load(m_path)
    vec = joblib.load(v_path)

    # 1. Калибровка порога tau на TRAIN
    X_train = vec.transform(train_df["text"])
    y_train = train_df["label"].values.astype(int)
    probs_train_neg = model.predict_proba(X_train[y_train == 0])[:, 1]

    if len(probs_train_neg) > 0:
      calibrated_tau = float(np.percentile(probs_train_neg, 99.5))
      # Гарантируем рабочий диапазон порога
      calibrated_tau = max(0.50, min(0.95, calibrated_tau))
    else:
      calibrated_tau = 0.85

    # Перезапись thresholds.json
    with open(t_path, "w") as f:
      json.dump({"review_threshold": calibrated_tau}, f, indent=2)

    # 2. Оценка на HOLD-OUT TEST
    X_test = vec.transform(test_df["text"])
    probs_test = model.predict_proba(X_test)[:, 1]
    preds_test = (probs_test >= calibrated_tau).astype(int)
    y_test = test_df["label"].values.astype(int)

    tp = int(np.sum((y_test == 1) & (preds_test == 1)))
    tn = int(np.sum((y_test == 0) & (preds_test == 0)))
    fp = int(np.sum((y_test == 0) & (preds_test == 1)))
    fn = int(np.sum((y_test == 1) & (preds_test == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    rec_low, rec_high = wilson_ci(recall, tp + fn)
    fpr_low, fpr_high = wilson_ci(fpr, fp + tn)

    rq1_proofs.append({
        "engine": eng,
        "threshold": round(calibrated_tau, 4),
        "test_n": len(test_df),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "recall": round(recall, 4),
        "recall_95ci": [round(rec_low, 4), round(rec_high, 4)],
        "fpr": round(fpr, 4),
        "fpr_95ci": [round(fpr_low, 4), round(fpr_high, 4)],
    })

  out_file = ARTIFACTS_DIR / "statistical_proofs.json"
  with open(out_file, "w", encoding="utf-8") as f:
    json.dump(rq1_proofs, f, indent=2)

  print(
      "✅ Thresholds recalibrated and updated in"
      f" {out_file.relative_to(BASE_DIR)}"
  )


if __name__ == "__main__":
  main()