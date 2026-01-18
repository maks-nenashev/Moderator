import json
import joblib
from pathlib import Path
from typing import List, Optional, Dict

# ======================================================
# UNIFIED CEE MODEL MANAGER (v4 & v4.1 Hardcore)
# ======================================================
class CEEModelManager:
    def __init__(self):
        # Хранилище для всех моделей и их настроек
        self.registry: Dict[str, Dict] = {}
        
        # Конфигурация путей (согласно твоей новой структуре)
        self.version_map = {
            "v4": Path("artifacts/v4"),
            "v4.1": Path("artifacts/v4.1")
        }

        self._initialize_registry()

    def _initialize_registry(self):
        """Автоматическая загрузка всех доступных версий"""
        for version, folder in self.version_map.items():
            try:
                model_p = folder / "model.joblib"
                vec_p = folder / "vectorizer.joblib"
                meta_p = folder / "meta.json"
                thr_p = folder / "thresholds.json"

                if model_p.exists() and vec_p.exists():
                    # Загружаем артефакты
                    model = joblib.load(model_p)
                    vec = joblib.load(vec_p)
                    
                    # Загружаем мета-данные
                    meta = json.load(open(meta_p)) if meta_p.exists() else {}
                    # Загружаем порог из калибровки
                    thr = json.load(open(thr_p)).get("review_threshold", 0.5) if thr_p.exists() else 0.5

                    self.registry[version] = {
                        "model": model,
                        "vectorizer": vec,
                        "threshold": thr,
                        "meta": meta
                    }
                    print(f"✅ [Registry] {version} loaded (Threshold: {thr})")
                else:
                    print(f"⚠️ [Registry] {version} skipped — missing artifacts in {folder}")

            except Exception as e:
                print(f"❌ [Registry] Failed to load {version}: {e}")

    def get_score(self, text: str, version: str = "v4") -> float:
        """Получить «сырой» скор от конкретной версии"""
        if version not in self.registry:
            return 0.0
        
        reg = self.registry[version]
        X = reg["vectorizer"].transform([text])
        return float(reg["model"].predict_proba(X)[:, 1][0])

    def moderate(self, text: str) -> dict:
        """
        ГЛАВНЫЙ МЕТОД: Опрашивает базу и сленг, выдает финальное решение.
        """
        # 1. Получаем сигналы
        score_v4 = self.get_score(text, "v4")
        score_v4_1 = self.get_score(text, "v4.1")

        # 2. Логика Decision Engine (Политика важнее ML)
        # Мы берем максимум, но v4.1 (сленг) имеет приоритет на коротких фразах
        final_score = max(score_v4, score_v4_1)
        
        # Определяем сработавшую версию
        source = "v4.1_hardcore" if score_v4_1 >= score_v4 else "v4_base"
        
        # Берем порог из v4 (базовый)
        threshold = self.registry.get("v4", {}).get("threshold", 0.45)

        return {
            "text": text,
            "final_score": round(final_score, 4),
            "decision": "REVIEW" if final_score >= threshold else "ALLOW",
            "source_model": source,
            "details": {
                "v4_score": round(score_v4, 4),
                "v4.1_score": round(score_v4_1, 4)
            }
        }

# ======================================================
# SINGLETON INSTANCE
# ======================================================
cee_manager = CEEModelManager()