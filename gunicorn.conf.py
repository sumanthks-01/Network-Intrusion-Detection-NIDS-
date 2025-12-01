# Gunicorn configuration for Render deployment
import os

bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2