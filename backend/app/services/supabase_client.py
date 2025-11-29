from app.core.db import get_supabase
from app.core.config import settings
from app.services.database import save_detection_sqlite
from loguru import logger

TABLE = settings.SUPABASE_TABLE_DETECTIONS

def save_detection(data: dict):
    # Try Supabase only
    supa = get_supabase()
    if not supa:
        logger.error("Supabase not configured")
        return None
    
    try:
        resp = supa.table(TABLE).insert(data).execute()
        logger.info(f"Detection saved to Supabase: {data.get('prediction', 'Unknown')}")
        return resp.data if hasattr(resp, "data") else None
    except Exception as e:
        logger.error(f"Supabase save failed: {e}")
        return None
