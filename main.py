import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Базовый путь к проекту
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Глобальный кэш моделей в RAM
MODELS: Dict[str, Dict[str, Any]] = {}
ENGINES = ["v1", "v3", "v3.1", "v3.2", "v4", "v4.1", "v4.2", "v5", "v5.1", "v5.2", "v6", "v6.1", "v6.2"]  


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
            threshold_data = {}
            if thr_file.exists():
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
    
    MODELS.clear()
    print("[SHUTDOWN] All ML engines purged from RAM.")


app = FastAPI(
    title="FindWay Data Sentinel NLP",
    version="1.0.0",
    lifespan=lifespan
)


class ModerationRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None


def process_single_text(text: str) -> Dict[str, Any]:
    """Вспомогательная функция инференса для одного фрагмента текста"""
    if not text or not text.strip():
        return {
            "decision": "ALLOW",
            "detected_by": [],
            "scores": {},
            "text": text
        }

    scores = {}
    detected_engines = []

    for eng, assets in MODELS.items():
        vec = assets["vectorizer"]
        model = assets["model"]
        threshold = assets["threshold"]

        X = vec.transform([text])
        prob = float(model.predict_proba(X)[0][1])
        
        scores[eng] = prob
        
        if prob >= threshold:
            detected_engines.append(eng)

    is_toxic = len(detected_engines) > 0

    return {
        "decision": "REVIEW" if is_toxic else "ALLOW",
        "detected_by": detected_engines,
        "scores": scores,
        "text": text
    }


# ======================================================
# РОУТЫ СЕРВИСА
# ======================================================

@app.post("/predict")
async def predict(data: ModerationRequest):
    """
    Универсальный инференс: принимает либо "text", либо "texts".
    Возвращает массив результатов [{decision, scores, ...}],
    чтобы полностью соответствовать логике Faraday / Rails Client.
    """
    input_texts = [data.text] if data.text else (data.texts or [])
    
    if not input_texts:
        raise HTTPException(status_code=400, detail="No text provided")

    results = [process_single_text(t) for t in input_texts]
    return results


@app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root_fallback():
    """
    Заглушка для корневых запросов.
    Полностью устраняет 405 Method Not Allowed в логах Uvicorn.
    """
    return {
        "status": "ready",
        "service": "FindWay Data Sentinel NLP",
        "loaded_engines": list(MODELS.keys())
    }


@app.get("/health")
def health():
    return {
        "status": "ready",
        "loaded_engines": list(MODELS.keys())
    }