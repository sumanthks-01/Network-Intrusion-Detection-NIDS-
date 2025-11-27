from app.core.config import settings
from loguru import logger

try:
    from supabase import create_client
except:
    create_client = None

_supabase = None

def get_supabase():
    global _supabase

    if _supabase:
        return _supabase

    if not settings.SUPABASE_URL:
        logger.warning("Supabase not configured!")
        return None

    if create_client is None:
        logger.warning("supabase-py not installed")
        return None

    key = settings.SUPABASE_SERVICE_ROLE_KEY or settings.SUPABASE_ANON_KEY
    _supabase = create_client(settings.SUPABASE_URL, key)
    logger.info("Supabase client initialized")

    return _supabase
