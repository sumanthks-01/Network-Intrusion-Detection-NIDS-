from fastapi import APIRouter, HTTPException
from app.models.detection import DetectionRequest, DetectionResponse
from app.services.detector import detect
from app.services.supabase_client import save_detection

router = APIRouter()

@router.post("/predict", response_model=DetectionResponse)
async def predict(req: DetectionRequest):
    result = detect(req.features, req.meta)

    # Save into Supabase
    save_detection({
        "prediction": result["prediction"],
        "score": result["score"],
        "meta": result["meta"]
    })

    return DetectionResponse(
        prediction=result["prediction"],
        score=result["score"],
        meta=result["meta"]
    )
