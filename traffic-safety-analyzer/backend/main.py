"""
FastAPI application entry point.
"""
from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Ensure project root is in sys.path whether executed from project root or backend/
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

from backend.config.settings import settings
from backend.database.db import init_db
from backend.detection.detector import load_detector

from backend.api.camera_routes import router as camera_router
from backend.api.monitoring_routes import router as monitoring_router
from backend.api.incident_routes import router as incident_router
from backend.api.dashboard_routes import router as dashboard_router
from backend.api.evaluation_routes import router as evaluation_router
from backend.api.settings_routes import router as settings_router
from backend.api.ws_routes import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Traffic Safety Analyzer backend …")
    settings.ensure_dirs()
    await init_db()
    logger.info(f"Database ready: {settings.database_url}")
    load_detector()
    yield
    # Shutdown
    from backend.pipeline import get_pipeline
    from backend.camera.capture import get_camera
    pipeline = get_pipeline()
    if pipeline and pipeline._running:
        await pipeline.stop()
    cam = get_camera()
    if cam and cam.status.running:
        cam.stop()
    logger.info("Backend shut down cleanly.")


app = FastAPI(
    title="AI Traffic Safety Analyzer",
    version="1.0.0",
    description="AI-powered traffic violation detection system.",
    lifespan=lifespan,
)

# CORS — allow the Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_url, "http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount evidence images as static files (for viewing in the frontend)
evidence_dir = Path(settings.evidence_dir)
evidence_dir.mkdir(parents=True, exist_ok=True)
app.mount("/evidence", StaticFiles(directory=str(evidence_dir)), name="evidence")

# Register routers
app.include_router(ws_router)
app.include_router(camera_router,     prefix="/api/v1")
app.include_router(monitoring_router, prefix="/api/v1")
app.include_router(incident_router,   prefix="/api/v1")
app.include_router(dashboard_router,  prefix="/api/v1")
app.include_router(evaluation_router, prefix="/api/v1")
app.include_router(settings_router,   prefix="/api/v1")


@app.get("/")
async def root() -> dict:
    return {"message": "Traffic Safety Analyzer API", "docs": "/docs"}


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": "1.0.0"}

