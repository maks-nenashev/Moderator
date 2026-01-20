import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from pathlib import Path
import argparse

def train(version):
    root_dir = Path(__file__).parent.parent
    data_path = root_dir / "data" / "processed" / f"dataset_v{version.replace('.', '_')}_west.csv"
    out_dir = root_dir / "artifacts" / f"v{version}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"❌ Ошибка: Файл {data_path} не найден! Сначала запусти prepare_dataset.py")
        return

    df = pd.read_csv(data_path)
    X_text = df['text'].values.astype('U')
    y = df['label'].values

    # Логика: если версия со сленгом (.1), используем символьные n-граммы
    if ".1" in version:
        print(f"🛠 Режим: SLANG/CHAR-WB (v{version})")
        vec = TfidfVectorizer(ngram_range=(2, 5), analyzer='char_wb', max_features=15000)
    else:
        print(f"🛠 Режим: BASE/WORD (v{version})")
        vec = TfidfVectorizer(ngram_range=(1, 2), analyzer='word', max_features=10000)

    X = vec.fit_transform(X_text)
    model = LogisticRegression(C=1.0, solver='liblinear', n_jobs=1)
    model.fit(X, y)

    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(vec, out_dir / "vectorizer.joblib")
    
    print(f"✅ Модель v{version} обучена и сохранена в {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    train(args.version)
