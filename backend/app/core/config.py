from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    APP_NAME = os.getenv("APP_NAME", "NIDS Backend")
    ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    SUPABASE_TABLE_DETECTIONS = os.getenv("SUPABASE_TABLE_DETECTIONS", "detections")

    MODEL_PATH = os.getenv("MODEL_PATH", "ids_model.pkl")

settings = Settings()
