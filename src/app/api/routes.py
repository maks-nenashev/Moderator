from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# --- ИМПОРТИРУЕМ ТОЛЬКО РОУТЕР ---
from app.core.moderation_router import ModerationRouter

router = APIRouter()

# Инициализируем роутер один раз при запуске
# Он сам найдет все папки в artifacts (v3, v3.1, v4, v4.1...)
moderator = ModerationRouter()

class PredictRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

# ======================================================
# Health (Теперь универсальный!)
# ======================================================
@router.get("/health")
def health():
    return {
        "status": "healthy",
        "active_engines": list(moderator.engines.keys())
    }

# ======================================================
# Predict (Чистая логика)
# ======================================================
@router.post("/predict")
def predict(payload: PredictRequest):
    # 1. Unify input
    texts = [payload.text] if payload.text else (payload.texts or [])
    if not texts:
        raise HTTPException(status_code=400, detail="No text provided")

    # 2. Process
    responses = []
    for text in texts:
        # Вся магия, все скоры и все пороги теперь ВНУТРИ метода predict роутера
        result = moderator.predict(text)
        
        responses.append({
            "decision": result["decision"],
            "reason": result["reason"],
            "scores": result["raw_scores"]
        })

    return responses