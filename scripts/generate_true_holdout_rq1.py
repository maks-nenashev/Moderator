#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
True Hold-Out RQ1 Evaluation Engine (N = 3,629)
Path: scripts/generate_true_holdout_rq1.py
Author: Maksym Nenashev (Systems Engineer)

Evaluates 22 frozen models STRICTLY on the 20% hold-out test split (seed=42).
Replaces diagnostic metrics in artifacts/statistical_proofs.json.
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
  print("🎯 EVALUATING TRUE HOLD-OUT METRICS FOR RQ1 (N = 3,629)")
  print("==================================================================")

  rq1_proofs = []

  for eng in ALL_ENGINES:
    base_key = eng.split(".")[0]
    csv_info = next(
        (item for item in DATASET_MAPPING if item["engine_base"] == base_key),
        None,
    )

    if not csv_info:
      continue

    fpath = DATA_DIR / csv_info["file"]
    if not fpath.exists():
      continue

    # 1. Clean & Deduplicate
    df_raw = pd.read_csv(fpath, on_bad_lines="skip", engine="python")
    df_clean = df_raw.dropna(subset=["text", "label"]).copy()
    df_clean["text"] = df_clean["text"].astype(str).str.strip()
    df_clean = df_clean.drop_duplicates(subset=["text"])

    # 2. Extract ONLY 20% Hold-out Split (Seed=42)
    y = df_clean["label"].values.astype(int)
    _, test_df = train_test_split(
        df_clean, test_size=0.20, random_state=42, stratify=y
    )

    m_path = ARTIFACTS_DIR / eng / "model.joblib"
    v_path = ARTIFACTS_DIR / eng / "vectorizer.joblib"
    t_path = ARTIFACTS_DIR / eng / "thresholds.json"

    if not (m_path.exists() and v_path.exists() and t_path.exists()):
      print(f"❌ Missing model artifacts for {eng}")
      continue

    with open(t_path) as f:
      t_data = json.load(f)
      thresh = t_data.get("review_threshold", t_data.get("threshold", 0.85))

    model = joblib.load(m_path)
    vec = joblib.load(v_path)

    # 3. Vectorization and Inference STRICTLY on Hold-out Test
    X_test = vec.transform(test_df["text"])
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= thresh).astype(int)
    y_true = test_df["label"].values.astype(int)

    tp = int(np.sum((y_true == 1) & (preds == 1)))
    tn = int(np.sum((y_true == 0) & (preds == 0)))
    fp = int(np.sum((y_true == 0) & (preds == 1)))
    fn = int(np.sum((y_true == 1) & (preds == 0)))

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    rec_low, rec_high = wilson_ci(recall, tp + fn)
    fpr_low, fpr_high = wilson_ci(fpr, fp + tn)

    rq1_proofs.append({
        "engine": eng,
        "threshold": thresh,
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
      "\n✅ True Hold-Out RQ1 evaluation complete. Saved to"
      f" {out_file.relative_to(BASE_DIR)}"
  )


if __name__ == "__main__":
  main()