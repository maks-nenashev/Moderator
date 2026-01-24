import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam

# =========================
# Paths
# =========================
DATA_PATH = Path("data/processed/dataset_v2_multilingual.csv")
ARTIFACTS = Path("artifacts")

MODEL_PATH = ARTIFACTS / "model_v2_multilingual.keras"
TOKENIZER_PATH = ARTIFACTS / "tokenizer_v2_multilingual.json"
LABELS_PATH = ARTIFACTS / "labels_v2.json"

ARTIFACTS.mkdir(exist_ok=True)

# =========================
# Config (reuse v1)
# =========================
with open(ARTIFACTS / "config.json", "r") as f:
    config = json.load(f)

MAX_WORDS = config["tokenizer"]["max_words"]
MAX_LEN = config["sequence"]["max_len"]

LABELS = ["toxicity", "hate", "sexual", "violence", "scam", "spam"]

# =========================
# Load dataset
# =========================
print("Loading multilingual dataset...")
df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
y = df[LABELS].values.astype("float32")

# =========================
# Train / validation split
# =========================
X_train, X_val, y_train, y_val = train_test_split(
    texts, y,
    test_size=0.2,
    random_state=42
)

# =========================
# Tokenizer (NEW)
# =========================
print("Training tokenizer (multilingual)...")

tokenizer = Tokenizer(
    num_words=MAX_WORDS,
    oov_token="<OOV>"
)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
X_val_pad = pad_sequences(X_val_seq, maxlen=MAX_LEN)

# =========================
# Model architecture
# =========================
print("Building model...")

model = Sequential([
    Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(len(LABELS), activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# Train
# =========================
print("Training multilingual model...")

model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=6,
    batch_size=128
)

# =========================
# Evaluation
# =========================
print("\n=== EVALUATION (v2 multilingual) ===")

y_pred = model.predict(X_val_pad)

for i, label in enumerate(LABELS):
    if y_val[:, i].sum() == 0:
        print(f"{label:10s} | no positive samples")
        continue

    auc = roc_auc_score(y_val[:, i], y_pred[:, i])
    f1 = f1_score(
        y_val[:, i],
        (y_pred[:, i] > 0.5).astype(int)
    )

    print(f"{label:10s} | AUC: {auc:.4f} | F1: {f1:.4f}")

# =========================
# Save artifacts
# =========================
model.save(MODEL_PATH)
print(f"\nSaved model to {MODEL_PATH}")

with open(TOKENIZER_PATH, "w") as f:
    f.write(tokenizer.to_json())

with open(LABELS_PATH, "w") as f:
    json.dump(
        {
            "labels": LABELS,
            "version": "v2_multilingual"
        },
        f,
        indent=2
    )

print("Saved tokenizer and labels")
