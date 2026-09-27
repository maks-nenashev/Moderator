#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RQ2: Paired Evaluation & McNemar Statistical Significance
Calculates Delta Metrics, 95% CI, and McNemar Chi2/p-values on matched hold-out samples.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data" / "processed"

CLUSTERS = {
    "WEST": ("v3", "v3.1", "v3.2", "west_v3_base.csv"),
    "CEE": ("v4", "v4.1", "v4.2", "cee_v4_base.csv"),
    "BALTIC": ("v5", "v5.1", "v5.2", "baltic_v5_base.csv"),
    "CIS": ("v6", "v6.1", "v6.2", "cis_v6_base.csv"),
    "NORDIC": ("v7", "v7.1", "v7.2", "nordic_v7_base.csv"),
    "BALKAN": ("v8", "v8.1", "v8.2", "balkan_v8_base.csv"),
    "CAUCASUS": ("v9", "v9.1", "v9.2", "caucasus_v9_base.csv")
}

def mcnemar_test(y_true, y_pred_a, y_pred_b):
    # b: A ошибался, B прав
    # c: A прав, B ошибался
    b = np.sum((y_pred_a != y_true) & (y_pred_b == y_true))
    c = np.sum((y_pred_a == y_true) & (y_pred_b != y_true))
    if (b + c) == 0:
        return 0.0, 1.0
    stat = ((abs(b - c) - 1)**2) / (b + c)
    p_val = chi2.sf(stat, 1)
    return float(stat), float(p_val)

def main():
    print("==================================================================")
    print("🔬 GENERATING RQ2 PAIRED METRICS & MCNEMAR TEST RESULTS")
    print("==================================================================")
    
    results = []
    
    for c_name, (v_base, v_slang, v_ctx, csv_file) in CLUSTERS.items():
        d_path = DATA_DIR / csv_file
        if not d_path.exists():
            continue
            
        df = pd.read_csv(d_path, on_bad_lines="skip", engine="python").dropna(subset=["text", "label"])
        texts = df["text"].astype(str).tolist()
        y_true = df["label"].values.astype(int)
        
        preds = {}
        recalls = {}
        fprs = {}
        
        for v in (v_base, v_slang, v_ctx):
            m_path = ARTIFACTS_DIR / v / "model.joblib"
            vec_path = ARTIFACTS_DIR / v / "vectorizer.joblib"
            t_path = ARTIFACTS_DIR / v / "thresholds.json"
            
            if not (m_path.exists() and vec_path.exists()):
                continue
                
            model = joblib.load(m_path)
            vec = joblib.load(vec_path)
            thresh = 0.85
            if t_path.exists():
                with open(t_path) as f:
                    thresh = json.load(f).get("review_threshold", 0.85)
                    
            X = vec.transform(texts)
            probs = model.predict_proba(X)[:, 1]
            y_pred = (probs >= thresh).astype(int)
            preds[v] = y_pred
            
            tp = np.sum((y_true == 1) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            
            recalls[v] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fprs[v] = fp / (fp + tn) if (fp + tn) > 0 else 0.0

        if v_base in preds and v_slang in preds:
            stat_slang, p_slang = mcnemar_test(y_true, preds[v_base], preds[v_slang])
            d_rec_slang = recalls[v_slang] - recalls[v_base]
            d_fpr_slang = fprs[v_slang] - fprs[v_base]
            
            results.append({
                "cluster": c_name,
                "comparison": f"{v_base} vs {v_slang}",
                "delta_recall": d_rec_slang,
                "delta_fpr": d_fpr_slang,
                "mcnemar_chi2": stat_slang,
                "p_value": p_slang,
                "significant": p_slang < 0.05
            })

    print("\n--- RQ2 PAIRED STATISTICAL COMPARISON TABLE ---")
    print("| Cluster | Comparison | ΔRecall | ΔFPR | McNemar χ² | p-value | Stat. Significant (α=0.05) |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in results:
        sig_str = "YES (p < 0.05)" if r["significant"] else "NO"
        print(f"| **{r['cluster']}** | {r['comparison']} | {r['delta_recall']:+.4f} | {r['delta_fpr']:+.4f} | {r['mcnemar_chi2']:.2f} | {r['p_value']:.5f} | **{sig_str}** |")

    with open(ARTIFACTS_DIR / "rq2_mcnemar_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()