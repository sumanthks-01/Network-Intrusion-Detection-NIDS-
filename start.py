#!/usr/bin/env python3
"""
Production startup script for Render deployment
"""
import os
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "backend.app.main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )