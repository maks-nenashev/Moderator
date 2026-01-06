import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

# ======================================================
# Paths
# ======================================================
ARTIFACTS_DIR = Path("artifacts/v3")

MODEL_PATH = ARTIFACTS_DIR / "model.joblib"
VECTORIZER_PATH = ARTIFACTS_DIR / "vectorizer.joblib"
META_PATH = ARTIFACTS_DIR / "meta.json"

# ======================================================
# Model Loader v3 (Sexual Intent)
# ======================================================
class ModelLoaderV3:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.meta = {}
        self.loaded = False

        self._load()

    # --------------------------
    def _load(self):
        try:
            if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
                print("[v3] Artifacts not found — v3 disabled")
                return

            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)

            if META_PATH.exists():
                with open(META_PATH, "r") as f:
                    self.meta = json.load(f)

            self.loaded = True
            print("[v3] Sexual intent model loaded")

        except Exception as e:
            print(f"[v3] Failed to load model: {e}")
            self.loaded = False

    # --------------------------
    def is_loaded(self) -> bool:
        return self.loaded

    # --------------------------
    def predict(self, texts: List[str]) -> List[Optional[dict]]:
        """
        Returns:
        [
          {
            "score": float,
            "model_version": "v3.0.0",
            "task": "sexual_intent",
            "language_group": "WEST"
          }
        ]
        """
        if not self.loaded:
            return [None] * len(texts)

        X = self.vectorizer.transform(texts)
        scores = self.model.predict_proba(X)[:, 1]

        results = []
        for score in scores:
            results.append({
                "score": float(score),
                "model_version": self.meta.get("model_version", "v3.0.0"),
                "task": "sexual_intent",
                "language_group": self.meta.get("language_group", "WEST"),
                "model_type": self.meta.get("model_type", "tfidf_logreg"),
            })

        return results


# ======================================================
# Singleton (IMPORTANT)
# ======================================================
model_loader_v3 = ModelLoaderV3()
