from typing import List, Dict
from pydantic import BaseModel


class PredictRequest(BaseModel):
    text: str | None = None
    texts: List[str] | None = None


class PredictResponse(BaseModel):
    label: str
    scores: Dict[str, float]
    model_version: str
