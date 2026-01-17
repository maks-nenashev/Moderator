import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# =========================
# Paths
# =========================
DATA_PATH = "data/processed/dataset_v1.csv"
ARTIFACTS_DIR = "artifacts"

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
num_labels = len(labels)

max_len = config["sequence"]["max_len"]
num_words = config["tokenizer"]["num_words"]

# =========================
# Load dataset
# =========================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

texts = df["text"].astype(str).tolist()
y = df[labels].values.astype("float32")

# =========================
# Train / Validation split
# =========================
X_train_texts, X_val_texts, y_train, y_val = train_test_split(
    texts,
    y,
    test_size=0.2,
    random_state=42,
    stratify=df["toxicity"]  # базовая стратификация
)

# =========================
# Tokenize + pad
# =========================
X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
X_val_seq = tokenizer.texts_to_sequences(X_val_texts)

X_train = pad_sequences(
    X_train_seq,
    maxlen=max_len,
    padding=config["sequence"]["padding"],
    truncating=config["sequence"]["truncating"]
)

X_val = pad_sequences(
    X_val_seq,
    maxlen=max_len,
    padding=config["sequence"]["padding"],
    truncating=config["sequence"]["truncating"]
)

# =========================
# Class weights (per label)
# =========================
# weight = (N - positives) / positives
class_weights = {}
for i, label in enumerate(labels):
    positives = y_train[:, i].sum()
    negatives = len(y_train) - positives
    if positives > 0:
        class_weights[i] = negatives / positives
    else:
        class_weights[i] = 1.0

print("Class weights:")
for i, label in enumerate(labels):
    print(label, "→", class_weights[i])

# =========================
# Model definition
# =========================
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64, return_sequences=False)),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dense(num_labels, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy"
)

model.summary()

# =========================
# Training
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=128,
    callbacks=[early_stop],
    verbose=1
)

# =========================
# Evaluation
# =========================
print("\n=== EVALUATION ===")
y_val_pred = model.predict(X_val)

for i, label in enumerate(labels):
    if y_val[:, i].sum() > 0:
        auc = roc_auc_score(y_val[:, i], y_val_pred[:, i])
        f1 = f1_score(y_val[:, i], y_val_pred[:, i] > 0.5)
        print(f"{label:10s} | AUC: {auc:.4f} | F1: {f1:.4f}")
    else:
        print(f"{label:10s} | no positive samples")

# =========================
# Save model (v1 real)
# =========================
model.save(f"{ARTIFACTS_DIR}/model_v1_real.keras")
print("Saved model to artifacts/model_v1_real.keras")
