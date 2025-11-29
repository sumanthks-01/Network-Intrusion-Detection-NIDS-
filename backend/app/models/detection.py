from pydantic import BaseModel
from typing import List, Dict, Any, Union

class DetectionRequest(BaseModel):
    features: List[float]
    meta: Dict[str, Any] | None = None

class DetectionResponse(BaseModel):
    prediction: Union[int, str]
    score: float | None = None
    meta: Dict[str, Any] | None = None
