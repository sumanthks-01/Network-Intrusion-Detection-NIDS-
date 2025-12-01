from fastapi import APIRouter
from fastapi.responses import FileResponse
import os

router = APIRouter()

# Serve frontend files
@router.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@router.get("/dashboard")
async def serve_dashboard():
    return FileResponse("frontend/dashboard.html")

@router.get("/dashboard.html")
async def serve_dashboard_html():
    return FileResponse("frontend/dashboard.html")

@router.get("/style.css")
async def serve_css():
    return FileResponse("frontend/style.css")

@router.get("/auth.js")
async def serve_auth_js():
    return FileResponse("frontend/auth.js")

@router.get("/dashboard.js")
async def serve_dashboard_js():
    return FileResponse("frontend/dashboard.js")