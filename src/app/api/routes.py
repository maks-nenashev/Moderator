from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional

# --- Импорты моделей и логики ---
from app.ml.policy import apply_policy
from app.ml.state import model_loader               # v1
from app.ml.state_v2 import model_loader_v2         # v2 (aggression)
from app.ml.state_v3 import model_loader_v3         # v3 (sexual intent West)
from app.ml.state_v3_1 import model_loader_v3_1     # v3.1 (Hardcore West Slang)
from app.ml.state_v4 import model_loader_v4         # v4 (sexual intent CEE)
from app.ml.state_v4_1 import model_loader_v4_1     # v4.1 (Hardcore CEE Slang)

# --- Определение схем данных (обязательно до использования!) ---
class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    country: Optional[str] = None
    user_id: Optional[str] = None

router = APIRouter()

# ======================================================
# Health 
# ======================================================
@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_v1_loaded": model_loader.is_loaded(),
        "model_v2_loaded": model_loader_v2.is_loaded(),
        "model_v3_loaded": model_loader_v3.is_loaded(),
        "model_v3_1_loaded": model_loader_v3_1.is_loaded(),
        "model_v4_loaded": model_loader_v4.is_loaded(),
        "model_v4_1_loaded": model_loader_v4_1.is_loaded()
    }

# ======================================================
# Meta
# ======================================================
@router.get("/meta")
def meta():
    return {
        "model_v1_loaded": model_loader.is_loaded(),
        "model_v2_loaded": model_loader_v2.is_loaded(),
        "model_v3_loaded": model_loader_v3.is_loaded(),
        "model_v3_1_loaded": model_loader_v3_1.is_loaded(),
        "model_v4_loaded": model_loader_v4.is_loaded(),
        "model_v4_1_loaded": model_loader_v4_1.is_loaded(),
        "model_version": model_loader.model_version,
        "labels": model_loader.labels,
    }

# ======================================================
# Predict
# ======================================================
@router.post("/predict")
def predict(payload: PredictRequest):
    # Проверка загрузки базовой модели
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="v1 model not loaded")

    # ---- unify input ----
    texts = [payload.text] if payload.text else (payload.texts or [])
    if not texts:
        raise HTTPException(status_code=400, detail="No text(s) provided")

    # ---- Collection of signals (ML Authority) ----
    predictions_v1 = model_loader.predict(texts)
    scores_v2 = model_loader_v2.predict(texts) if model_loader_v2.is_loaded() else [None] * len(texts)
    scores_v3 = model_loader_v3.predict(texts) if model_loader_v3.is_loaded() else [None] * len(texts)
    scores_v3_1 = model_loader_v3_1.predict(texts) if model_loader_v3_1.is_loaded() else [None] * len(texts)
    scores_v4 = model_loader_v4.predict(texts) if model_loader_v4.is_loaded() else [None] * len(texts)
    scores_v4_1 = model_loader_v4_1.predict(texts) if model_loader_v4_1.is_loaded() else [None] * len(texts)

    # ---- policy orchestration (Decision Authority) ----
    responses = []

    # Проходим по всем текстам и собираем вердикты
    for s_v1, s_v2, s_v3, s_v3_1, s_v4, s_v4_1 in zip(
        predictions_v1, scores_v2, scores_v3, scores_v3_1, scores_v4, scores_v4_1
    ):
        # Передаем сигналы в политику (убедись, что apply_policy принимает v4_1!)
        policy_result = apply_policy(
            scores_v1=s_v1,
            score_v2=s_v2,
            v3=s_v3,
            v3_1={"score": s_v3_1},  # <--- Оборачиваем float в dict
            v4=s_v4,
            #v4_1=s_v4_1
            v4_1={"score": s_v4_1}  # <--- Оборачиваем float в dict
        )

        responses.append({
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "scores": {
                "v1": s_v1,
                "v2": s_v2,
                "v3": s_v3,
                "v3.1": s_v3_1,
                "v4": s_v4,
                "v4.1": s_v4_1,
            },
            "model_version": model_loader.model_version,
        })

    return responses