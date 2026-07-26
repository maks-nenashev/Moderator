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

    echo "✅ $version ready, calibrated, and synced to S3."
}

# 3. Main case statement
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
        $0 trafficking
        $0 west
        $0 cee
        $0 baltic
        $0 cis
        ;;
    *)
        echo "Usage: $0 {trafficking|west|cee|baltic|cis|full}"
        exit 1
        ;;
esac

echo "🏁 All tasks completed successfully."