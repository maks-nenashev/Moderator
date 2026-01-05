import json
from typing import List, Dict
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import tokenizer_from_json

from app.ml.preprocess import normalize_batch


class ModelLoader:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.labels: List[str] = []
        self.config: Dict = {}
        self.model_version: str = "none"

    def load(self):
        # load config
        with open("artifacts/config.json", "r") as f:
            self.config = json.load(f)

        # load labels
        with open("artifacts/labels.json", "r") as f:
            labels_cfg = json.load(f)
            self.labels = labels_cfg["labels"]
            self.model_version = labels_cfg.get("version", "v1")

        # load tokenizer
        with open("artifacts/tokenizer.json", "r") as f:
            self.tokenizer = tokenizer_from_json(f.read())

        # load model
        self.model = load_model("artifacts/model.keras")

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        assert self.is_loaded(), "Model not loaded"

        texts = normalize_batch(texts)
        seqs = self.tokenizer.texts_to_sequences(texts)

        X = pad_sequences(
            seqs,
            maxlen=self.config["sequence"]["max_len"],
            padding=self.config["sequence"]["padding"],
            truncating=self.config["sequence"]["truncating"]
        )

        preds = self.model.predict(X)

        results = []
        for row in preds:
            results.append({
                label: float(score)
                for label, score in zip(self.labels, row)
            })

        return results

