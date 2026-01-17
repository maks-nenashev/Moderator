import json
import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, precision_recall_curve
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

# =========================
# Paths
# =========================
DATA_PATH = "data/processed/dataset_v1.csv"
ARTIFACTS_DIR = "artifacts"
MODEL_PATH = f"{ARTIFACTS_DIR}/model_v1_real.keras"

# =========================
# Load artifacts
# =========================
with open(f"{ARTIFACTS_DIR}/config.json", "r") as f:
    config = json.load(f)

with open(f"{ARTIFACTS_DIR}/tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

with open(f"{ARTIFACTS_DIR}/labels.json", "r") as f:
    labels_cfg = json.load(f)

labels = labels_cfg["labels"]
max_len = config["sequence"]["max_len"]

# =========================
# Load dataset
# =========================
df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()
y_true = df[labels].values.astype("int32")

# =========================
# Tokenize
# =========================
X_seq = tokenizer.texts_to_sequences(texts)
X = pad_sequences(
    X_seq,
    maxlen=max_len,
    padding=config["sequence"]["padding"],
    truncating=config["sequence"]["truncating"]
)

# =========================
# Load model & predict
# =========================
model = load_model(MODEL_PATH)
y_pred = model.predict(X, batch_size=256)

# =========================
# Threshold calibration
# =========================
thresholds = {}

print("\n=== THRESHOLD CALIBRATION (BALANCED) ===")

for i, label in enumerate(labels):
    if y_true[:, i].sum() == 0:
        print(f"{label:10s} | no positives → skipped")
        thresholds[label] = {
            "review": None,
            "block": None
        }
        continue

    precision, recall, thresh = precision_recall_curve(y_true[:, i], y_pred[:, i])
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    best_idx = np.nanargmax(f1_scores)
    best_threshold = thresh[best_idx] if best_idx < len(thresh) else 0.5

    # Conservative block threshold (higher confidence)
    thresholds[label] = {
        "review": round(float(best_threshold), 3),
        "block": None
    }

    print(
        f"{label:10s} | "
        f"review ≥ {thresholds[label]['review']:<5} "
        f"block ≥ {thresholds[label]['block']}"
    )

# =========================
# Save thresholds
# =========================
OUT_PATH = f"{ARTIFACTS_DIR}/policy_thresholds.json"
with open(OUT_PATH, "w") as f:
    json.dump(thresholds, f, indent=2)

print(f"\nSaved policy thresholds to {OUT_PATH}")
