from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import json
from pathlib import Path

app = FastAPI()

def get_threshold(version: str):
    path = Path(f"artifacts/{version}/thresholds.json")
    with open(path, "r") as f:
        return json.load(f)["review_threshold"]

class RequestData(BaseModel):
    text: str

@app.post("/moderate")
async def moderate(data: RequestData):
    results = {}
    engines = ["v3", "v3.1", "v4", "v4.1", "v5", "v5.1", "v6", "v6.1"]
    
    for eng in engines:
        path = Path(f"artifacts/{eng}")
        if not path.exists(): continue
        
        model = joblib.load(path / "model.joblib")
        vec = joblib.load(path / "vectorizer.joblib")
        thr = get_threshold(eng)
        
        prob = float(model.predict_proba(vec.transform([data.text]))[0][1])
        results[eng] = {
            "score": prob,
            "threshold": thr,
            "is_toxic": prob >= thr
        }
    
    is_any_toxic = any(res["is_toxic"] for res in results.values())
    return {
        "text": data.text,
        "decision": "review" if is_any_toxic else "allow",
        "engines": results
    }

@app.get("/health")
def health():
    return {"status": "ready"}