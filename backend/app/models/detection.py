from pydantic import BaseModel
from typing import List, Dict, Any

class DetectionRequest(BaseModel):
    features: List[float]
    meta: Dict[str, Any] | None = None

class DetectionResponse(BaseModel):
    prediction: int
    score: float | None = None
    meta: Dict[str, Any] | None = None
