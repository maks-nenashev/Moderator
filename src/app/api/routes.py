from fastapi import APIRouter, HTTPException

from app.api.schemas import PredictRequest
from app.ml.policy import apply_policy

from app.ml.state import model_loader               # v1
from app.ml.state_v2 import model_loader_v2         # v2 (aggression)
from app.ml.state_v3 import model_loader_v3         # v3 (sexual intent)

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
    if model_loader_v2.is_loaded():
        scores_v2 = model_loader_v2.predict(texts)
    else:
        scores_v2 = [None] * len(texts)

    # ---- v3 predictions (sexual intent signal) ----
    if model_loader_v3.is_loaded():
        scores_v3 = model_loader_v3.predict(texts)
    else:
        scores_v3 = [None] * len(texts)

    # ---- policy orchestration ----
    responses = []

    for scores_v1, score_v2, score_v3 in zip(
        predictions_v1, scores_v2, scores_v3
    ):
        policy_result = apply_policy(
            scores_v1=scores_v1,
            score_v2=score_v2,
            v3=score_v3,
        )

        responses.append({
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "scores": {
                "v1": scores_v1,
                "v2": score_v2,
                "v3": score_v3,
            },
            "model_version": model_loader.model_version,
        })

    return responses
