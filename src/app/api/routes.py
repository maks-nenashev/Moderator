from fastapi import APIRouter, HTTPException

from app.api.schemas import PredictRequest
from app.ml.policy import apply_policy

from app.ml.state import model_loader               # v1
from app.ml.state_v2 import model_loader_v2         # v2 (aggression)
from app.ml.state_v3 import model_loader_v3         # v3 (sexual intent West)
from app.ml.state_v4 import model_loader_v4         # v4 (sexual intent CEE)

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
        "model_v4_loaded": model_loader_v4.is_loaded(),
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
        "model_v4_loaded": model_loader_v4.is_loaded(),
        "model_version": model_loader.model_version,
        "labels": model_loader.labels,
    }

# ======================================================
# Predict
# ======================================================
@router.post("/predict")
def predict(payload: PredictRequest):
    # ---- safety check ----
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="v1 model not loaded")

    # ---- unify input ----
    if payload.text:
        texts = [payload.text]
    elif payload.texts:
        texts = payload.texts
    else:
        raise HTTPException(status_code=400, detail="No text(s) provided")

    # ---- v1 predictions (content signals) ----
    predictions_v1 = model_loader.predict(texts)

    # ---- v2 predictions (aggression / block gate) ----
    scores_v2 = model_loader_v2.predict(texts) if model_loader_v2.is_loaded() else [None] * len(texts)

    # ---- v3 predictions (sexual intent West) ----
    scores_v3 = model_loader_v3.predict(texts) if model_loader_v3.is_loaded() else [None] * len(texts)

    # ---- v4 predictions (sexual intent CEE) ----
    scores_v4 = model_loader_v4.predict(texts) if model_loader_v4.is_loaded() else [None] * len(texts)

    # ---- policy orchestration ----
    responses = []

    # Используем zip для прохода по результатам всех четырех моделей одновременно
    for s_v1, s_v2, s_v3, s_v4 in zip(
        predictions_v1, scores_v2, scores_v3, scores_v4
    ):
        # Передаем всё в policy.py, который мы до этого обновили
        policy_result = apply_policy(
            scores_v1=s_v1,
            score_v2=s_v2,
            v3=s_v3,
            v4=s_v4
        )

        responses.append({
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "scores": {
                "v1": s_v1,
                "v2": s_v2,
                "v3": s_v3,
                "v4": s_v4,
            },
            "model_version": model_loader.model_version,
        })

    return responses