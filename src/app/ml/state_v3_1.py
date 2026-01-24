import joblib
import json
from pathlib import Path
from typing import List

class ModelLoaderV31:
    """Менеджер для модели v3.1 (Hardcore WEST Slang: EN, DE, FR, ES, NL, IT, PT)"""
    def __init__(self):
        self.base_path = Path("artifacts/v3.1")
        self.model = None
        self.vectorizer = None
        self.threshold = 0.65 

        self.load_artifacts()

    def load_artifacts(self):
        model_path = self.base_path / "model.joblib"
        vec_path = self.base_path / "vectorizer.joblib"
        thr_path = self.base_path / "thresholds.json"

        if model_path.exists() and vec_path.exists():
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
            
            if thr_path.exists():
                with open(thr_path, "r", encoding="utf-8") as f:
                    self.threshold = json.load(f).get("review_threshold", self.threshold)
            
            # ИСПРАВЛЕНО: Лог теперь соответствует версии и региону
            print(f"[v3.1] Hardcore WEST model loaded (Threshold: {self.threshold})")
        else:
            print(f"⚠️ [v3.1] Warning: West artifacts not found in {self.base_path}")

    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def predict(self, texts: List[str]) -> List[float]:
        if not self.is_loaded():
            return [0.0] * len(texts)
        
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1].tolist()

# Singleton инстанс
model_loader_v3_1 = ModelLoaderV31()