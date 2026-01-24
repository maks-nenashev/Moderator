import json
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np

# ======================================================
# Paths for V4 (CEE - Central Eastern Europe)
# ======================================================
# Указываем на папку v4, которую ты подготовил
ARTIFACTS_DIR_V4 = Path("artifacts/v4")

MODEL_PATH_V4 = ARTIFACTS_DIR_V4 / "model.joblib"
VECTORIZER_PATH_V4 = ARTIFACTS_DIR_V4 / "vectorizer.joblib"
META_PATH_V4 = ARTIFACTS_DIR_V4 / "meta.json"

# ======================================================
# Model Loader v4 (Sexual Intent - CEE) 
# ======================================================
class ModelLoaderV4:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.meta = {}
        self.loaded = False

        self._load()

    # --------------------------
    def _load(self):
        try:
            if not MODEL_PATH_V4.exists() or not VECTORIZER_PATH_V4.exists():
                print("[v4] CEE Artifacts not found — v4 disabled")
                return

            self.model = joblib.load(MODEL_PATH_V4)
            self.vectorizer = joblib.load(VECTORIZER_PATH_V4)

            if META_PATH_V4.exists():
                with open(META_PATH_V4, "r") as f:
                    self.meta = json.load(f)

            self.loaded = True
            print("[v4] Sexual intent model (CEE) loaded")

        except Exception as e:
            print(f"[v4] Failed to load CEE model: {e}")
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
            "model_version": "v4.0.0",
            "task": "sexual_intent",
            "language_group": "CEE"
          }
        ]
        """
        if not self.loaded:
            return [None] * len(texts)

        # Превращаем текст в вектор через чешско-польский векторизатор
        X = self.vectorizer.transform(texts)
        scores = self.model.predict_proba(X)[:, 1]

        results = []
        for score in scores:
            results.append({
                "score": float(score),
                # Берем версию из мета-файла, по дефолту v4.0.0
                "model_version": self.meta.get("model_version", "v4.0.0"),
                "task": "sexual_intent",
                "language_group": self.meta.get("language_group", "CEE"),
                "model_type": self.meta.get("model_type", "tfidf_logreg"),
            })

        return results


# ======================================================
# Singleton for V4
# ======================================================
model_loader_v4 = ModelLoaderV4()