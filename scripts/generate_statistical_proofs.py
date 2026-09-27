#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical Proofs Generator: 95% CI, McNemar Test, and Raw Confusion Matrices
Author: Maksym Nenashev (Systems Engineer)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import confusion_matrix
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DATASET_MAP = {
    "v1": "trafficking_v1.csv",
    "v3": "west_v3_base.csv", "v3.1": "west_v3_1_slang.csv", "v3.2": "west_v3_2_context.csv",
    "v4": "cee_v4_base.csv", "v4.1": "cee_v4_1_slang.csv", "v4.2": "cee_v4_2_context.csv",
    "v5": "baltic_v5_base.csv", "v5.1": "baltic_v5_1_slang.csv", "v5.2": "baltic_v5_2_context.csv",
    "v6": "cis_v6_base.csv", "v6.1": "cis_v6_1_slang.csv", "v6.2": "cis_v6_2_context.csv",
    "v7": "nordic_v7_base.csv", "v7.1": "nordic_v7_1_slang.csv", "v7.2": "nordic_v7_2_context.csv",
    "v8": "balkan_v8_base.csv", "v8.1": "balkan_v8_1_slang.csv", "v8.2": "balkan_v8_2_context.csv",
    "v9": "caucasus_v9_base.csv", "v9.1": "caucasus_v9_1_slang.csv", "v9.2": "caucasus_v9_2_context.csv"
}

def wilson_ci(p, n, z=1.96):
    """Вычисление 95% доверительного интервала Уилсона для пропорций."""
    if n == 0:
        return 0.0, 0.0
    denominator = 1 + z**2 / n
    centre_adjusted_probability = p + z**2 / (2 * n)
    adjusted_standard_error = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower_bound = (centre_adjusted_probability - adjusted_standard_error) / denominator
    upper_bound = (centre_adjusted_probability + adjusted_standard_error) / denominator
    return max(0.0, lower_bound), min(1.0, upper_bound)

def run_mcnemar_test(y_true, y_pred_base, y_pred_sub):
    """Тест Мак-Немара для сравнения Base vs .1/.2 слоев."""
    # b: Base ошибался, Sub прав
    # c: Base прав, Sub ошибался
    b = np.sum((y_pred_base != y_true) & (y_pred_sub == y_true))
    c = np.sum((y_pred_base == y_true) & (y_pred_sub != y_true))
    
    if (b + c) == 0:
        return 0.0, 1.0 # Нет различий
    
    stat = ((abs(b - c) - 1)**2) / (b + c)
    p_value = chi2.sf(stat, 1)
    return float(stat), float(p_value)

def main():
    print("==================================================")
    print("🔬 GENERATING ACADEMIC STATISTICAL PROOFS (Q2/Q1)")
    print("==================================================")
    
    detailed_results = []

    for ver, csv_file in MODEL_DATASET_MAP.items():
        m_path = MODELS_DIR / ver / "model.joblib"
        v_path = MODELS_DIR / ver / "vectorizer.joblib"
        t_path = MODELS_DIR / ver / "thresholds.json"
        d_path = DATA_DIR / csv_file

        if not all([m_path.exists(), v_path.exists(), t_path.exists(), d_path.exists()]):
            continue

        model = joblib.load(m_path)
        vectorizer = joblib.load(v_path)
        with open(t_path) as f:
            thresh = json.load(f).get("review_threshold", 0.5)

        df = pd.read_csv(d_path, on_bad_lines="skip", engine="python").dropna(subset=["text", "label"])
        texts, y_true = df["text"].astype(str).tolist(), df["label"].values.astype(int)

        X = vectorizer.transform(texts)
        probs = model.predict_proba(X)[:, 1]
        y_pred = (probs >= thresh).astype(int)

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        n = len(y_true)

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        fpr_low, fpr_high = wilson_ci(fpr, fp + tn)

        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        rec_low, rec_high = wilson_ci(recall, tp + fn)

        detailed_results.append({
            "version": ver,
            "n_samples": n,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "threshold": thresh,
            "fpr": fpr, "fpr_95ci": [round(fpr_low, 4), round(fpr_high, 4)],
            "recall": recall, "recall_95ci": [round(rec_low, 4), round(rec_high, 4)],
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist()
        })

    # Сохранение матриц и 95% CI
    out_file = BASE_DIR / "artifacts" / "statistical_proofs.json"
    with open(out_file, "w") as f:
        json.dump(detailed_results, f, indent=2)

    print(f"✅ Saved Raw CMs and 95% CIs to {out_file.relative_to(BASE_DIR)}")

    # Таблица для статьи
    print("\n--- STATISTICAL VALIDATION TABLE FOR PAPER ---")
    print("| Engine | Test N | TP | TN | FP | FN | FPR (95% CI) | Recall (95% CI) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in detailed_results:
        fpr_str = f"{r['fpr']:.4f} [{r['fpr_95ci'][0]:.4f}-{r['fpr_95ci'][1]:.4f}]"
        rec_str = f"{r['recall']:.4f} [{r['recall_95ci'][0]:.4f}-{r['recall_95ci'][1]:.4f}]"
        print(f"| `{r['version']}` | {r['n_samples']} | {r['tp']} | {r['tn']} | {r['fp']} | {r['fn']} | {fpr_str} | {rec_str} |")

if __name__ == "__main__":
    main()