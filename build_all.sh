#!/bin/bash

# Протокол: Stability & Reproducibility
# Автор: Maksym Nenashev (Systems Engineer)

set -e # Остановка при любой ошибке скрипта

echo "🚀 Starting FindWay Build Pipeline..."

# 1. Определение активного питона
if [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_BIN="$VIRTUAL_ENV/bin/python3"
    echo "✅ Active environment detected: $VIRTUAL_ENV"
elif [ -f ".venv/bin/python3" ]; then
    PYTHON_BIN="$(pwd)/.venv/bin/python3"
    echo "✅ Using environment at .venv"
elif [ -f "venv/bin/python3" ]; then
    PYTHON_BIN="$(pwd)/venv/bin/python3"
    echo "✅ Using environment at venv"
else
    echo "❌ Error: No virtual environment found."
    exit 1
fi

# 2. Функция обучения и калибровки (Targeted Build)
train_model() {
    local version=$1
    local input=$2
    local analyzer=$3
    
    if [ ! -f "$input" ]; then
        echo "❌ Critical Error: Dataset $input not found!"
        exit 1
    fi

    echo "--- 🧠 Training $version ($analyzer) using $input ---"
    
    "$PYTHON_BIN" train/train_engine.py --version "$version" --input "$input" --analyzer "$analyzer" \
        --ngram_min 2 --ngram_max 5

    echo "--- 📊 Calibrating $version ---"
    "$PYTHON_BIN" -c "
import pandas as pd, numpy as np, json, joblib
from pathlib import Path

data_path = Path('$input')
model_path = Path('artifacts/$version/model.joblib')
vec_path = Path('artifacts/$version/vectorizer.joblib')

# Чтение с защитой от поврежденных строк и неправильных кавычек
try:
    df = pd.read_csv(data_path, on_bad_lines='skip', engine='python')
except Exception as e:
    df = pd.read_csv(data_path, on_bad_lines='skip')

model = joblib.load(model_path)
vec = joblib.load(vec_path)

# Определяем колонку с текстом
text_col = 'text' if 'text' in df.columns else df.columns[0]
label_col = 'label' if 'label' in df.columns else df.columns[-1]

X = vec.transform(df[text_col].fillna('').astype(str))
probs = model.predict_proba(X)[:, 1]

df['score'] = probs
neg = df[df[label_col] == 0]['score']

thr = float(np.quantile(neg, 0.98)) if len(neg) > 0 else 0.5

out_dir = Path('artifacts/$version')
out_dir.mkdir(parents=True, exist_ok=True)

with open(out_dir / 'thresholds.json', 'w') as f:
    json.dump({'review_threshold': round(thr, 4), 'version': '$version'}, f)
"
    echo "✅ $version ready and calibrated."
}

# 3. The main case statement to handle different build targets
case "$1" in
    "trafficking")
        train_model "v1" "data/processed/trafficking_v1.csv" "char_wb"
        ;;
    "west")
        train_model "v3" "data/processed/west_v3_base.csv" "word"
        train_model "v3.1" "data/processed/west_v3_1_slang.csv" "char"
        train_model "v3.2" "data/processed/west_v3_2_context.csv" "word"
        ;;
    "cee")
        train_model "v4" "data/processed/cee_v4_base.csv" "word"
        train_model "v4.1" "data/processed/cee_v4_1_slang.csv" "char"
        train_model "v4.2" "data/processed/cee_v4_2_context.csv" "word"
        ;;
    "baltic")
        train_model "v5" "data/processed/baltic_v5_base.csv" "word"
        train_model "v5.1" "data/processed/baltic_v5_1_slang.csv" "char"
        train_model "v5.2" "data/processed/baltic_v5_2_context.csv" "word"
        ;;
    "cis")
        train_model "v6" "data/processed/cis_v6_base.csv" "word"
        train_model "v6.1" "data/processed/cis_v6_1_slang.csv" "char"
        train_model "v6.2" "data/processed/cis_v6_2_context.csv" "word"
        ;;
    "full")
        echo "🌍 Running FULL pipeline build..."
        $0 west
        $0 cee
        $0 baltic
        $0 cis
        ;;
    *)
        echo "Usage: $0 {west|cee|baltic|cis|full}"
        exit 1
        ;;
esac

echo "🏁 All tasks completed successfully."

#   ./build_all.sh baltic