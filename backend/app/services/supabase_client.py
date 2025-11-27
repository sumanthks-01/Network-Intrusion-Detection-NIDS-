from app.core.db import get_supabase
from app.core.config import settings
from app.services.database import save_detection_sqlite
from loguru import logger

TABLE = settings.SUPABASE_TABLE_DETECTIONS

def save_detection(data: dict):
    # Try Supabase first
    supa = get_supabase()
    if supa:
        try:
            resp = supa.table(TABLE).insert(data).execute()
            logger.info("Detection saved to Supabase")
            return resp.data if hasattr(resp, "data") else None
        except Exception as e:
            logger.error(f"Supabase save failed: {e}")
    
    # Fallback to SQLite
    logger.info("Using SQLite fallback")
    save_detection_sqlite(data)
    return data
