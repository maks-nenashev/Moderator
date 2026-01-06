from fastapi import APIRouter, HTTPException
from app.ml.state import model_loader
from app.ml.state_v2 import model_loader_v2  # second model
from app.ml.policy import apply_policy
from app.api.schemas import PredictRequest

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded(),
        "model_v2_loaded": model_loader_v2.is_loaded()
    }


@router.get("/meta")
def meta():
    return {
        "model_loaded": model_loader.is_loaded(),
        "model_v2_loaded": model_loader_v2.is_loaded(),
        "model_version": model_loader.model_version,
        "labels": model_loader.labels
    }


@router.post("/predict")
def predict(payload: PredictRequest):
    # ---- safety check ----
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    # ---- unify input ----
    if payload.text:
        texts = [payload.text]
    elif payload.texts:
        texts = payload.texts
    else:
        raise HTTPException(status_code=400, detail="No text(s) provided")

    # ---- model predictions ----
    predictions_v1 = model_loader.predict(texts)

    if model_loader_v2.is_loaded():
        scores_v2 = model_loader_v2.predict(texts)
    else:
        scores_v2 = [None] * len(texts)

    # ---- policy orchestration ----
    responses = []

    for scores_v1, score_v2 in zip(predictions_v1, scores_v2):
        policy_result = apply_policy(
            scores_v1=scores_v1,
            score_v2=score_v2
        )

        responses.append({
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "scores": scores_v1,
            "v2_score": score_v2,
            "model_version": model_loader.model_version
        })

    return responses
