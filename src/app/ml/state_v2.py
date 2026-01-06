import json
from typing import List
from pathlib import Path

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

from app.ml.preprocess import normalize_batch


class ModelLoaderV2:
    """
    Multilingual toxicity detector (v2).
    Returns ONE score per text.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.max_len = None
        self.model_version = "v2_multilingual"

    def load(self):
        with open("artifacts/config.json") as f:
            config = json.load(f)
            self.max_len = config["sequence"]["max_len"]

        with open("artifacts/tokenizer_v2_multilingual.json") as f:
            self.tokenizer = tokenizer_from_json(f.read())

        self.model = load_model("artifacts/model_v2_multilingual.keras")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, texts: List[str]) -> List[float]:
        assert self.is_loaded(), "v2 model not loaded"

        texts = normalize_batch(texts)
        seqs = self.tokenizer.texts_to_sequences(texts)

        X = pad_sequences(seqs, maxlen=self.max_len)

        preds = self.model.predict(X, batch_size=256)

        # гарантируем форму (N,)
        if preds.ndim > 1:
            preds = preds[:, 0]

        return preds.tolist()


# singleton
model_loader_v2 = ModelLoaderV2()
