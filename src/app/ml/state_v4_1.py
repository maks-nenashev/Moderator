import joblib
import json
from pathlib import Path

class ModelLoaderV41:
    """Менеджер для модели v4.1 (Hardcore CEE Slang)"""
    def __init__(self):
        self.base_path = Path("artifacts/v4.1")
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
                with open(thr_path, "r") as f:
                    self.threshold = json.load(f).get("review_threshold", self.threshold)
            
            print(f"[v4.1] Hardcore CEE model loaded (Threshold: {self.threshold})")
        else:
            print(f"⚠️ [v4.1] Warning: Artifacts not found in {self.base_path}")

    # --- ОБЯЗАТЕЛЬНО: Проверка загрузки для API ---
    def is_loaded(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    # --- ОБЯЗАТЕЛЬНО: Поддержка пакетной обработки (списков) ---
    def predict(self, texts: list) -> list:
        """Принимает список строк, возвращает список скоров"""
        if not self.is_loaded():
            return [0.0] * len(texts)
        
        # Обрабатываем весь список сразу через векторизатор
        X = self.vectorizer.transform(texts)
        # Превращаем в список float
        return self.model.predict_proba(X)[:, 1].tolist()

# Создаем синглтон
model_loader_v4_1 = ModelLoaderV41()