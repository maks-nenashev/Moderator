import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

try:
    from train.s3_sync import download_version_artifacts
except ModuleNotFoundError:
    download_version_artifacts = None

ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODELS: Dict[str, Dict[str, Any]] = {}
ENGINES = ["v1", "v3", "v3.1", "v3.2", "v4", "v4.1", "v4.2", "v5", "v5.1", "v5.2", "v6", "v6.1", "v6.2"]  


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей в RAM."""
    for eng in ENGINES:
        eng_path = ARTIFACTS_DIR / eng
        model_file = eng_path / "model.joblib"
        vec_file = eng_path / "vectorizer.joblib"
        thr_file = eng_path / "thresholds.json"

        if not (eng_path.exists() and model_file.exists() and vec_file.exists()):
            if download_version_artifacts:
                print(f"⚠️ Local artifacts for [{eng}] missing. Pulling from S3...")
                try:
                    download_version_artifacts(eng)
                except Exception as e:
                    print(f"❌ Failed to download [{eng}] from S3: {e}")

        if eng_path.exists() and model_file.exists() and vec_file.exists():
            threshold_data = {}
            if thr_file.exists():
                threshold_data = json.loads(thr_file.read_text())
            
            MODELS[eng] = {
                "model": joblib.load(model_file),
                "vectorizer": joblib.load(vec_file),
                "threshold": threshold_data.get("review_threshold", 0.5)
            }
            print(f"[LOADER] Engine {eng} cached in RAM.")

    print(f"✅ Successfully loaded {len(MODELS)} engines: {list(MODELS.keys())}")
            
    yield
    MODELS.clear()


app = FastAPI(
    title="FindWay Data Sentinel NLP",
    version="1.0.0",
    lifespan=lifespan
)


class ModerationRequest(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None
    content: Optional[str] = None

    def get_input_texts(self) -> List[str]:
        if self.text:
            return [self.text]
        if self.content:
            return [self.content]
        if self.texts:
            return self.texts
        return []


def process_single_text(text: str) -> Dict[str, Any]:
    if not text or not text.strip():
        return {"decision": "ALLOW", "detected_by": [], "scores": {}, "text": text}

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


@app.post("/predict")
@app.post("/moderate")
async def predict(data: ModerationRequest):
    input_texts = data.get_input_texts()
    if not input_texts:
        raise HTTPException(status_code=400, detail="No text provided")
    return [process_single_text(t) for t in input_texts]


@app.api_route("/", methods=["GET", "POST", "HEAD"])
async def root_fallback():
    return {"status": "ready", "service": "FindWay Data Sentinel NLP", "loaded_engines": list(MODELS.keys())}


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ready", "loaded_engines": list(MODELS.keys())}


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)