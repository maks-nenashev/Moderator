#!/bin/bash

# Протокол: Stability & Reproducibility
# Автор: Maksym Nenashev (Systems Engineer)

echo "🚀 Starting FindWay Build Pipeline..."

# 1. Активация окружения
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated."
else
    echo "❌ Error: venv not found. Run 'python3 -m venv venv' first."
    exit 1
fi

# 2. Функция обучения (Targeted Build)
train_model() {
    local version=$1
    local input=$2
    local analyzer=$3
    
    echo "--- 🧠 Training $version ($analyzer) ---"
    python train/train_engine.py --version "$version" --input "$input" --analyzer "$analyzer" \
        --ngram_min 2 --ngram_max 5
    
    echo "--- 📊 Calibrating $version ---"
    # Авто-калибровка (98-й квантиль)
    python train/common/evaluate.py --data "$input" --model "artifacts/$version/model.joblib" \
        --vectorizer "artifacts/$version/vectorizer.joblib" --out "artifacts/$version/scores.csv"
    
    python3 -c "
import pandas as pd, json, numpy as np
df = pd.read_csv('artifacts/$version/scores.csv')
neg = df[df['label'] == 0]['score']
thr = float(np.quantile(neg, 0.98))
with open('artifacts/$version/thresholds.json', 'w') as f:
    json.dump({'review_threshold': round(thr, 4), 'version': '$version'}, f)
"
    echo "✅ $version ready and calibrated."
}

# 3. Диспетчер задач
case "$1" in
    
    "west")
        train_model "v3" "data/processed/dataset_v3_1_west.csv" "word"
        train_model "v3.1" "data/processed/dataset_v3_3_1_west.csv" "char"
        ;;
    "cee")
        # Центральная и Восточная Европа (PL, CZ, SK)
        train_model "v4" "data/processed/dataset_v4_4_1_cee.csv" "word"
        train_model "v4.1" "data/processed/dataset_v4_4_1_cee.csv" "char"
        ;;
    "baltic")
        train_model "v5" "data/raw/baltic_v5_basic_raw.csv" "word"
        train_model "v5.1" "data/raw/baltic_v5_1_slang_raw.csv" "char"
        ;;
    "cis")
        # Обучение единого блока РФ/Украина
        train_model "v6" "data/raw/cis_v6_base.csv" "word"
        train_model "v6.1" "data/raw/cis_v6_1_slang.csv" "char"
        ;;
    "full")
        # Полная сборка всех векторов системы FindWay
        echo "🌍 Running FULL pipeline build..."
        # West
        train_model "v3" "data/processed/dataset_v3_1_west.csv" "word"
        train_model "v3.1" "data/processed/dataset_v3_3_1_west.csv" "char"
        # CEE
        train_model "v4" "data/processed/dataset_v4_4_1_cee.csv" "word"
        train_model "v4.1" "data/processed/dataset_v4_4_1_cee.csv" "char"
        # Baltic
        train_model "v5" "data/raw/baltic_v5_basic_raw.csv" "word"
        train_model "v5.1" "data/raw/baltic_v5_1_slang_raw.csv" "char"
        # CIS
        train_model "v6" "data/raw/cis_v6_base.csv" "word"
        train_model "v6.1" "data/raw/cis_v6_1_slang.csv" "char"
        ;;
    *)
        echo "Usage: $0 {west|cee|baltic|cis|full}"
        exit 1
        ;;
esac

echo "🏁 All tasks completed."
