import json
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, List, Optional

import joblib
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST, Counter, Histogram

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

# ==============================================================================
# PROMETHEUS METRICS DECLARATION
# ==============================================================================
NLP_ENGINE_INFERENCE_SECONDS = Histogram(
    "nlp_engine_inference_seconds",
    "Time spent in inference for a single NLP engine",
    ["engine_id"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
)

NLP_ENGINE_PREDICTIONS_TOTAL = Counter(
    "nlp_engine_predictions_total",
    "Total predictions count by engine and decision threshold result",
    ["engine_id", "triggered"]
)

NLP_REQUESTS_TOTAL = Counter(
    "nlp_requests_total",
    "Total moderation requests handled by overall system decision",
    ["decision"]
)


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
        NLP_REQUESTS_TOTAL.labels(decision="ALLOW").inc()
        return {"decision": "ALLOW", "detected_by": [], "scores": {}, "text": text}

    scores = {}
    detected_engines = []

    for eng, assets in MODELS.items():
        start_time = time.perf_counter()
        
        vec = assets["vectorizer"]
        model = assets["model"]
        threshold = assets["threshold"]

        X = vec.transform([text])
        prob = float(model.predict_proba(X)[0][1])
        scores[eng] = prob
        
        elapsed = time.perf_counter() - start_time
        
        # Запись метрики задержки для конкретной модели
        NLP_ENGINE_INFERENCE_SECONDS.labels(engine_id=eng).observe(elapsed)
        
        is_triggered = prob >= threshold
        if is_triggered:
            detected_engines.append(eng)
            
        # Запись счетчика срабатываний по моделям
        NLP_ENGINE_PREDICTIONS_TOTAL.labels(
            engine_id=eng, 
            triggered="true" if is_triggered else "false"
        ).inc()

    is_toxic = len(detected_engines) > 0
    final_decision = "REVIEW" if is_toxic else "ALLOW"
    
    # Запись итогового решения по запросу
    NLP_REQUESTS_TOTAL.labels(decision=final_decision).inc()

    return {
        "decision": final_decision,
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