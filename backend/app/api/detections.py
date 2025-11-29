from fastapi import APIRouter, HTTPException
from app.models.detection import DetectionRequest, DetectionResponse
from app.services.detector import detect
from app.services.supabase_client import save_detection

router = APIRouter()

@router.post("/predict", response_model=DetectionResponse)
async def predict(req: DetectionRequest):
    from loguru import logger
    logger.info(f"Received prediction request: {req.meta}")
    
    # If meta contains attack info, use it directly
    if req.meta and "attack_type" in req.meta:
        attack_type = req.meta["attack_type"]
        confidence = req.meta.get("confidence", 0.0)
        
        detection_data = {
            "prediction": attack_type,
            "score": confidence,
            "meta": req.meta
        }
        
        logger.info(f"Saving detection: {detection_data}")
        save_detection(detection_data)
        
        return DetectionResponse(
            prediction=attack_type,
            score=confidence,
            meta=req.meta
        )
    
    # Fallback to model prediction
    result = detect(req.features, req.meta)
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
