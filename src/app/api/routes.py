from fastapi import APIRouter, HTTPException
from app.ml.state import model_loader
from app.ml.policy import apply_policy
from app.api.schemas import PredictRequest

router = APIRouter()


@router.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model_loader.is_loaded()
    }


@router.get("/meta")
def meta():
    return {
        "model_loaded": model_loader.is_loaded(),
        "model_version": model_loader.model_version,
        "labels": model_loader.labels
    }


@router.post("/predict")
def predict(payload: PredictRequest):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded")

    if payload.text:
        texts = [payload.text]
    elif payload.texts:
        texts = payload.texts
    else:
        raise HTTPException(status_code=400, detail="No text(s) provided")

    predictions = model_loader.predict(texts)

    responses = []
    for scores in predictions:
        policy_result = apply_policy(scores)

        responses.append({
            "decision": policy_result["decision"],
            "reasons": policy_result["reasons"],
            "scores": scores,
            "model_version": model_loader.model_version
        })

    return responses

