from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings

from app.core.logger import configure_logging
from app.api.health import router as health_router
from app.api.detections import router as detections_router
from app.api.auth import router as auth_router
from app.api.stats import router as stats_router
from app.api.frontend import router as frontend_router

configure_logging()
app = FastAPI(title=settings.APP_NAME)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(health_router, prefix="/api/health", tags=["health"])
app.include_router(detections_router, prefix="/api/detections", tags=["detections"])
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(stats_router, prefix="/api/stats", tags=["stats"])
app.include_router(frontend_router, tags=["frontend"])




@app.get("/")
async def root():
    return {"status": "ok", "app": settings.APP_NAME}
