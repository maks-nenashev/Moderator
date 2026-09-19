import argparse
import pandas as pd
import joblib
import json
import numpy as np
import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import mlflow

# Импортируем нашу утилиту загрузки в S3
import sys
from pathlib import Path

# Добавляем корень проекта в sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from train.s3_sync import upload_version_artifacts

def main():
    parser = argparse.ArgumentParser(description="Moderation Model Trainer")
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--analyzer', type=str, default='word')
    parser.add_argument('--ngram_min', type=int, default=1)
    parser.add_argument('--ngram_max', type=int, default=2)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Error: File {args.input} not found!")
        return

    print(f"🚀 Starting training for [{args.version}] using {args.input}")
    
    # 1. Загрузка данных
    df = pd.read_csv(input_path, on_bad_lines='skip')
    print(f"📦 Loaded {len(df)} rows")

    # Указываем имя эксперимента в MLflow
    mlflow.set_experiment("findway_moderation_models")

    with mlflow.start_run(run_name=f"train_{args.version}"):
        # 2. Векторизация
        ngram_range = (args.ngram_min, args.ngram_max)
        vectorizer = TfidfVectorizer(
            analyzer=args.analyzer, 
            ngram_range=ngram_range
        )
        X = vectorizer.fit_transform(df['text'].values.astype('U'))
        y = df['label'].values

        # 3. Обучение
        model = LogisticRegression(class_weight='balanced', max_iter=1000)
        model.fit(X, y)

        # 4. Расчет вероятностей и метрик
        probs = model.predict_proba(X)[:, 1]
        
        # 95-й перцентиль на чистых данных (y == 0) для снижения False Positives
        review_threshold = float(np.percentile(probs[y == 0], 95))
        preds = (probs >= review_threshold).astype(int)

        # Метрики качества
        auc = float(roc_auc_score(y, probs))
        precision = float(precision_score(y, preds, zero_division=0))
        recall = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))

        # 5. Логирование гиперпараметров и метрик в MLflow
        mlflow.log_params({
            "version": args.version,
            "analyzer": args.analyzer,
            "ngram_range": str(ngram_range),
            "dataset_rows": len(df),
            "vocabulary_size": len(vectorizer.vocabulary_)
        })

        mlflow.log_metrics({
            "roc_auc": round(auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "review_threshold": round(review_threshold, 4)
        })

        s3_bucket = os.getenv("S3_BUCKET_NAME", "findway-ml-artifacts")
        mlflow.set_tag("s3_artifact_uri", f"s3://{s3_bucket}/models/{args.version}/")

        # 6. Сохранение локальных артефактов
        out_dir = Path(f"artifacts/{args.version}")
        out_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, out_dir / "model.joblib")
        joblib.dump(vectorizer, out_dir / "vectorizer.joblib")
        
        thresholds = {
            "review_threshold": round(review_threshold, 4),
            "block_threshold": 0.9,
            "version": args.version
        }
        
        with open(out_dir / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=4)

        print(f"✅ Saved local artifacts for [{args.version}] to {out_dir}")
        print(f"📊 Metrics -> ROC-AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")
        print(f"🎯 Review Threshold: {review_threshold:.4f}")

        # 7. Автоматическая выгрузка весов в S3
        upload_version_artifacts(args.version)

if __name__ == "__main__":
    main()