import argparse
import pandas as pd
import joblib
import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    parser = argparse.ArgumentParser(description="Moderation Model Trainer")
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--analyzer', type=str, default='word')
    parser.add_argument('--ngram_min', type=int, default=1)
    parser.add_argument('--ngram_max', type=int, default=2)
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"❌ Error: File {args.input} not found!")
        return

    print(f"🚀 Starting training for {args.version} using {args.input}")
    
    # 1. Загрузка данных с защитой от "битых" строк
    df = pd.read_csv(args.input, on_bad_lines='skip')
    print(f"📦 Loaded {len(df)} rows")

    # 2. Векторизация
    vectorizer = TfidfVectorizer(
        analyzer=args.analyzer, 
        ngram_range=(args.ngram_min, args.ngram_max)
    )
    X = vectorizer.fit_transform(df['text'].values.astype('U'))
    y = df['label'].values

    # 3. Обучение
    model = LogisticRegression(class_weight='balanced', max_iter=1000)
    model.fit(X, y)

    # 4. Расчет порога (Review Threshold)
    probs = model.predict_proba(X)[:, 1]
    # Используем 95-й перцентиль на чистых данных, чтобы снизить False Positives
    review_threshold = float(np.percentile(probs[y == 0], 95)) 

    # 5. Сохранение артефактов
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

    print(f"✅ Saved artifacts for {args.version} to {out_dir}")
    print(f"📊 Suggested Review Threshold: {review_threshold:.4f}")

if __name__ == "__main__":
    main()