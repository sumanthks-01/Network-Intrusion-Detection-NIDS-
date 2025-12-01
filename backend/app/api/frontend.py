from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Get the correct path to frontend files
# Go up from backend/app/api/ to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

# Serve frontend files
@router.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

@router.get("/dashboard")
async def serve_dashboard():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

@router.get("/dashboard.html")
async def serve_dashboard_html():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.html"))

@router.get("/style.css")
async def serve_css():
    response = FileResponse(os.path.join(FRONTEND_DIR, "style.css"))
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response

@router.get("/auth.js")
async def serve_auth_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "auth.js"))

@router.get("/dashboard.js")
async def serve_dashboard_js():
    return FileResponse(os.path.join(FRONTEND_DIR, "dashboard.js"))