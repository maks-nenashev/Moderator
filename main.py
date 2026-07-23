import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import joblib
from fastapi import FastAPI
from pydantic import BaseModel

# Определяем базовый путь к проекту относительно текущего файла
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Глобальный кэш моделей в RAM
MODELS: Dict[str, Dict[str, Any]] = {}

ENGINES = ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Pre-loading контур: прогрев всех моделей в оперативную память
    при запуске FastAPI сервиса.
    """
    for eng in ENGINES:
        eng_path = ARTIFACTS_DIR / eng
        model_file = eng_path / "model.joblib"
        vec_file = eng_path / "vectorizer.joblib"
        thr_file = eng_path / "thresholds.json"

        if eng_path.exists() and model_file.exists() and vec_file.exists():
            threshold_data = json.loads(thr_file.read_text())
            
            MODELS[eng] = {
                "model": joblib.load(model_file),
                "vectorizer": joblib.load(vec_file),
                "threshold": threshold_data.get("review_threshold", 0.5)
            }
            print(f"[LOADER] Engine {eng} cached successfully in RAM.")
        else:
            print(f"[WARNING] Engine {eng} artifacts missing in {eng_path}")
            
    yield
    
    # Очистка ресурсов при остановке
    MODELS.clear()
    print("[SHUTDOWN] All ML engines purged from RAM.")


app = FastAPI(
    title="FindWay Data Sentinel NLP",
    version="1.0.0",
    lifespan=lifespan
)


class ModerationRequest(BaseModel):
    text: str


@app.post("/predict")
async def predict(data: ModerationRequest):
    """
    Инференс векторных оценок из RAM без обращения к диску.
    """
    if not data.text or not data.text.strip():
        return {
            "decision": "ALLOW",
            "scores": {},
            "text": data.text
        }

    scores = {}
    detected_engines = []

    # Прогон текста по всем cached моделям
    for eng, assets in MODELS.items():
        vec = assets["vectorizer"]
        model = assets["model"]
        threshold = assets["threshold"]

        # Трансформация и расчет вероятности
        X = vec.transform([data.text])
        prob = float(model.predict_proba(X)[0][1])
        
        scores[eng] = prob
        
        if prob >= threshold:
            detected_engines.append(eng)

    is_toxic = len(detected_engines) > 0

    return {
        "decision": "REVIEW" if is_toxic else "ALLOW",
        "detected_by": detected_engines,
        "scores": scores,
        "text": data.text
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "loaded_engines": list(MODELS.keys())
    }