import json
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from sklearn.metrics import precision_recall_curve
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

from app.ml.preprocess import normalize_batch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
# =========================
# Paths
# =========================
DATA_PATH = Path("data/processed/dataset_v2_multilingual.csv")
ARTIFACTS = Path("artifacts")

MODEL_PATH = ARTIFACTS / "model_v2_multilingual.keras"
TOKENIZER_PATH = ARTIFACTS / "tokenizer_v2_multilingual.json"
CONFIG_PATH = ARTIFACTS / "config.json"

OUT_PATH = ARTIFACTS / "policy_thresholds_v2.json"

# =========================
# Load config
# =========================
with open(CONFIG_PATH) as f:
    config = json.load(f)

MAX_LEN = config["sequence"]["max_len"]

# =========================
# Load tokenizer
# =========================
with open(TOKENIZER_PATH) as f:
    tokenizer = tokenizer_from_json(f.read())

# =========================
# Load dataset
# =========================
print("Loading multilingual dataset...")
df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
y_true = df["toxicity"].values.astype("int32")

# =========================
# Preprocess
# =========================
texts = normalize_batch(texts)

seqs = tokenizer.texts_to_sequences(texts)
X = pad_sequences(seqs, maxlen=MAX_LEN)

# =========================
# Predict
# =========================
print("Running inference...")
model = load_model(MODEL_PATH)
y_pred = model.predict(X, batch_size=256)

# гарантируем форму (N,)
if y_pred.ndim > 1:
    y_pred = y_pred[:, 0]

# =========================
# Calibration
# =========================
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)

# F1 score
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_idx = np.argmax(f1_scores)

review_threshold = thresholds[best_idx]
block_threshold = min(review_threshold + 0.25, 0.95)

print("\n=== V2 TOXICITY THRESHOLDS ===")
print(f"review ≥ {review_threshold:.3f}")
print(f"block  ≥ {block_threshold:.3f}")

# =========================
# Save
# =========================
policy = {
    "model": "v2_multilingual",
    "toxicity": {
        "review": round(float(review_threshold), 3),
        "block": round(float(block_threshold), 3)
    }
}

with open(OUT_PATH, "w") as f:
    json.dump(policy, f, indent=2)

print(f"\nSaved thresholds to {OUT_PATH}")
