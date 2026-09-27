#!/usr/bin/env python3
# -*- coding: utf-8 -*-                   python3 scripts/generate_paper_metrics.py

"""
Comprehensive Metrics Generator for Paper (Precision, Recall, F1, FPR, FNR, CM)
Author: Maksym Nenashev (Systems Engineer)
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"

MODEL_DATASET_MAP = {
    "v1": "trafficking_v1.csv",
    "v3": "west_v3_base.csv",
    "v3.1": "west_v3_1_slang.csv",
    "v3.2": "west_v3_2_context.csv",
    "v4": "cee_v4_base.csv",
    "v4.1": "cee_v4_1_slang.csv",
    "v4.2": "cee_v4_2_context.csv",
    "v5": "baltic_v5_base.csv",
    "v5.1": "baltic_v5_1_slang.csv",
    "v5.2": "baltic_v5_2_context.csv",
    "v6": "cis_v6_base.csv",
    "v6.1": "cis_v6_1_slang.csv",
    "v6.2": "cis_v6_2_context.csv",
    "v7": "nordic_v7_base.csv",
    "v7.1": "nordic_v7_1_slang.csv",
    "v7.2": "nordic_v7_2_context.csv",
    "v8": "balkan_v8_base.csv",
    "v8.1": "balkan_v8_1_slang.csv",
    "v8.2": "balkan_v8_2_context.csv",
    "v9": "caucasus_v9_base.csv",
    "v9.1": "caucasus_v9_1_slang.csv",
    "v9.2": "caucasus_v9_2_context.csv"
}

def evaluate_engine(version: str, csv_filename: str):
    model_path = MODELS_DIR / version / "model.joblib"
    vec_path = MODELS_DIR / version / "vectorizer.joblib"
    thresh_path = MODELS_DIR / version / "thresholds.json"
    data_path = DATA_DIR / csv_filename
    
    if not all([model_path.exists(), vec_path.exists(), thresh_path.exists(), data_path.exists()]):
        print(f"⚠️ Skipping [{version}]: Missing artifacts or dataset.")
        return None

    # Загрузка артефактов
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    with open(thresh_path, "r", encoding="utf-8") as f:
        threshold_data = json.load(f)
    
    threshold = threshold_data.get("review_threshold", threshold_data.get("threshold", 0.5))

    # Загрузка датасета с защитой от синтаксических ошибок в CSV
    try:
        df = pd.read_csv(data_path, on_bad_lines="skip", engine="python")
    except Exception as e:
        print(f"⚠️ Failed to parse dataset {csv_filename}: {e}")
        return None

    if "text" not in df.columns or "label" not in df.columns:
        print(f"⚠️ Invalid columns in {csv_filename}")
        return None

    texts = df["text"].astype(str).tolist()
    y_true = df["label"].values

    # Инференс
    X = vectorizer.transform(texts)
    y_scores = model.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    # Метрики через sklearn
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    # Confusion Matrix (TN, FP, FN, TP)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # False Positive Rate (FPR) & False Negative Rate (FNR)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "version": version,
        "threshold": threshold,
        "samples": len(y_true),
        "precision_class_0": precision[0],
        "precision_class_1": precision[1] if len(precision) > 1 else 0.0,
        "recall_class_0": recall[0],
        "recall_class_1": recall[1] if len(recall) > 1 else 0.0,
        "f1_class_0": f1[0],
        "f1_class_1": f1[1] if len(f1) > 1 else 0.0,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "fpr": fpr,
        "fnr": fnr,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    }

def main():
    print("==================================================")
    print("📊 DATA SENTINEL — ACADEMIC METRICS GENERATOR")
    print("==================================================")
    
    results = []
    for version, csv_file in MODEL_DATASET_MAP.items():
        res = evaluate_engine(version, csv_file)
        if res:
            results.append(res)
            print(f"✅ Processed engine [{version}]: FPR={res['fpr']:.4f}, Macro-F1={res['macro_f1']:.4f}")

    # Сохранение полного отчета
    report_path = BASE_DIR / "artifacts" / "paper_metrics_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n📄 Full evaluation report saved to: {report_path}")

    # Генерация Markdown таблицы для статьи
    print("\n--- MARKDOWN TABLE FOR PAPER ---")
    print("| Engine | Threshold | Macro-F1 | Precision (1) | Recall (1) | FPR | FNR |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        print(f"| `{r['version']}` | {r['threshold']:.2f} | **{r['macro_f1']:.4f}** | {r['precision_class_1']:.4f} | {r['recall_class_1']:.4f} | {r['fpr']:.4f} | {r['fnr']:.4f} |")

if __name__ == "__main__":
    main()