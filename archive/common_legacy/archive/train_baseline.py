import json
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Bidirectional, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json
from keras.optimizers import Adam

# -------------------------
# Load configs & artifacts
# -------------------------
with open("artifacts/config.json", "r") as f:
    config = json.load(f)

with open("artifacts/tokenizer.json", "r") as f:
    tokenizer = tokenizer_from_json(f.read())

with open("artifacts/labels.json", "r") as f:
    labels_cfg = json.load(f)

labels = labels_cfg["labels"]
num_labels = len(labels)

max_len = config["sequence"]["max_len"]
num_words = config["tokenizer"]["num_words"]

# -------------------------
# Dummy training data
# (заменим реальным датасетом позже)
# -------------------------
texts = [
    "You are stupid",
    "Free money click now",
    "I will kill you",
    "Hot girls near you",
    "Normal neutral message"
]

# multi-label targets (toxicity, hate, sexual, violence, scam, spam)
y = np.array([
    [1, 0, 0, 0, 0, 0],  # toxicity
    [0, 0, 0, 0, 1, 1],  # scam + spam
    [0, 0, 0, 1, 0, 0],  # violence
    [0, 0, 1, 0, 0, 1],  # sexual + spam
    [0, 0, 0, 0, 0, 0],  # clean
], dtype="float32")

# -------------------------
# Text -> sequences
# -------------------------
sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(
    sequences,
    maxlen=max_len,
    padding=config["sequence"]["padding"],
    truncating=config["sequence"]["truncating"]
)

# -------------------------
# Model definition
# -------------------------
model = Sequential([
    Embedding(input_dim=num_words, output_dim=128, input_length=max_len),
    Bidirectional(LSTM(64)),
    Dense(64, activation="relu"),
    Dense(num_labels, activation="sigmoid")
])

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Train (baseline)
# -------------------------
model.fit(
    X, y,
    epochs=3,
    batch_size=2,
    verbose=1
)

# -------------------------
# Save model
# -------------------------
model.save("artifacts/model.keras")
print("Model saved to artifacts/model.keras")
