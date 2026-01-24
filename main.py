from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from pathlib import Path
from contextlib import asynccontextmanager

# 1. Глобальный кэш моделей (в RAM)
MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ЗАГРУЗКА ПРИ СТАРТЕ (Pre-loading)
    engines = ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"]
    for eng in engines:
        path = Path(f"artifacts/{eng}")
        if path.exists():
            MODELS[eng] = {
                "model": joblib.load(path / "model.joblib"),
                "vectorizer": joblib.load(path / "vectorizer.joblib"),
                "threshold": json.loads((path / "thresholds.json").read_text())["review_threshold"]
            }
            print(f"[LOADER] Engine {eng} cached in RAM.")
    yield
    MODELS.clear()

app = FastAPI(lifespan=lifespan)

class RequestData(BaseModel):
    text: str

@app.post("/predict") # Синхронизируем с твоим Rails-клиентом
async def predict(data: RequestData):
    results = {}
    
    for eng, assets in MODELS.items():
        # Быстрый инференс из памяти
        vec = assets["vectorizer"]
        model = assets["model"]
        
        # transform ожидает список строк
        X = vec.transform([data.text])
        prob = float(model.predict_proba(X)[0][1])
        
        results[eng] = prob # Твой Policy ожидает плоский список скоров

    is_any_toxic = any(prob >= MODELS[eng]["threshold"] for eng, prob in results.items())
    
    return {
        "decision": "🔴 REVIEW" if is_any_toxic else "✅ ALLOW",
        "scores": results,
        "text": data.text
    }