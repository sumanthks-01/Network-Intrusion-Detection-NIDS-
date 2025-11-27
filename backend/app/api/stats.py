from fastapi import APIRouter
from app.services.database import get_detections_sqlite
from loguru import logger

router = APIRouter()

@router.get("/detections")
async def get_detections(limit: int = 100):
    """Get recent detections"""
    try:
        detections = get_detections_sqlite(limit)
        return {"detections": detections, "count": len(detections)}
    except Exception as e:
        logger.error(f"Failed to get detections: {e}")
        return {"error": str(e), "detections": [], "count": 0}

@router.get("/statistics")
async def get_statistics():
    """Get detection statistics"""
    try:
        detections = get_detections_sqlite(1000)  # Get more for stats
        
        total_count = len(detections)
        attack_types = {}
        
        for detection in detections:
            pred = detection.get('prediction', 'Unknown')
            attack_types[pred] = attack_types.get(pred, 0) + 1
        
        return {
            "total_detections": total_count,
            "attack_distribution": attack_types,
            "recent_count": len([d for d in detections[:100]])  # Last 100
        }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        return {"error": str(e)}