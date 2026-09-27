import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Добавляем корень проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.s3_sync import upload_version_artifacts


def main():
    parser = argparse.ArgumentParser(
        description="Paper-Grade Moderation Model Trainer"
    )
    parser.add_argument("--version", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--analyzer", type=str, default="word")
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)
    parser.add_argument("--test_size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File {args.input} not found!")
        return

    print(f"🚀 Starting Paper-Grade Training for [{args.version}] using {args.input}")

    # 1. Загрузка и принудительная дедупликация
    df = pd.read_csv(input_path, on_bad_lines="skip", engine="python")
    initial_len = len(df)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    print(f"📦 Loaded {initial_len} rows | Clean Deduplicated: {len(df)} rows")

    if len(df) < 10:
        print("❌ Error: Insufficient data for train/test split!")
        return

    # 2. Стратифицированное разделение на Train (80%) и Hold-out Test (20%)
    X_raw = df["text"].astype("U").values
    y_raw = df["label"].values.astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw,
        y_raw,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_raw,
    )

    print(
        f"✂️ Split: Train = {len(X_train_raw)} rows | Hold-out Test = {len(X_test_raw)} rows"
    )

    mlflow.set_experiment("findway_moderation_models")

    with mlflow.start_run(run_name=f"train_{args.version}"):
        # 3. Векторизация: Fit ТОЛЬКО на Train
        ngram_range = (args.ngram_min, args.ngram_max)
        vectorizer = TfidfVectorizer(
            analyzer=args.analyzer,
            ngram_range=ngram_range,
        )

        X_train = vectorizer.fit_transform(X_train_raw)
        X_test = vectorizer.transform(X_test_raw)  # Исключительно Transform!

        # 4. Обучение модели
        model = LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=args.seed
        )
        model.fit(X_train, y_train)

        # 5. Калибровка порога СТРОГО на Train-выборке
        probs_train = model.predict_proba(X_train)[:, 1]
        train_negatives = probs_train[y_train == 0]

        if len(train_negatives) > 0:
            review_threshold = float(np.percentile(train_negatives, 95))
        else:
            review_threshold = 0.5

        # 6. Валидация на НЕТРОНУТОМ Hold-out Test
        probs_test = model.predict_proba(X_test)[:, 1]
        preds_test = (probs_test >= review_threshold).astype(int)

        # Вычисление метрик для статьи
        auc_test = (
            float(roc_auc_score(y_test, probs_test))
            if len(np.unique(y_test)) > 1
            else 0.5
        )
        precision_test = float(
            precision_score(y_test, preds_test, zero_division=0)
        )
        recall_test = float(recall_score(y_test, preds_test, zero_division=0))
        f1_test = float(f1_score(y_test, preds_test, zero_division=0))

        cm = confusion_matrix(y_test, preds_test, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        fpr_test = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr_test = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

        # 7. Логирование в MLflow
        mlflow.log_params(
            {
                "version": args.version,
                "analyzer": args.analyzer,
                "ngram_range": str(ngram_range),
                "dataset_rows_raw": initial_len,
                "train_rows": len(X_train_raw),
                "test_rows": len(X_test_raw),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "seed": args.seed,
            }
        )

        mlflow.log_metrics(
            {
                "holdout_roc_auc": round(auc_test, 4),
                "holdout_precision": round(precision_test, 4),
                "holdout_recall": round(recall_test, 4),
                "holdout_f1": round(f1_test, 4),
                "holdout_fpr": round(fpr_test, 4),
                "holdout_fnr": round(fnr_test, 4),
                "calibrated_threshold": round(review_threshold, 4),
            }
        )

        s3_bucket = os.getenv("S3_BUCKET_NAME", "findway-ml-artifacts")
        mlflow.set_tag(
            "s3_artifact_uri", f"s3://{s3_bucket}/models/{args.version}/"
        )

        # 8. Сохранение локальных артефактов
        out_dir = Path(f"artifacts/{args.version}")
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, out_dir / "model.joblib")
        joblib.dump(vectorizer, out_dir / "vectorizer.joblib")

        thresholds = {
            "review_threshold": round(review_threshold, 4),
            "block_threshold": 0.9,
            "version": args.version,
        }

        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=4)

        # Сохранение полного академического отчета для генератора метрик
        paper_eval = {
            "version": args.version,
            "threshold": round(review_threshold, 4),
            "holdout_samples": len(y_test),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
            "metrics": {
                "roc_auc": round(auc_test, 4),
                "precision": round(precision_test, 4),
                "recall": round(recall_test, 4),
                "f1": round(f1_test, 4),
                "fpr": round(fpr_test, 4),
                "fnr": round(fnr_test, 4),
            },
        }

        with open(out_dir / "paper_eval.json", "w") as f:
            json.dump(paper_eval, f, indent=4)

        print(f"✅ Saved paper-grade artifacts to {out_dir}")
        print(
            f"📊 HOLDOUT RESULTS -> ROC-AUC: {auc_test:.4f} | Precision: {precision_test:.4f} | Recall: {recall_test:.4f} | F1: {f1_test:.4f}"
        )
        print(
            f"🎯 FPR: {fpr_test:.4f} | FNR: {fnr_test:.4f} | CM: TP={tp}, TN={tn}, FP={fp}, FN={fn}"
        )

        # 9. Выгрузка в S3
        upload_version_artifacts(args.version)


if __name__ == "__main__":
    main()