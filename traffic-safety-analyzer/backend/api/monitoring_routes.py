"""
Monitoring start/stop API routes.
"""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException

from backend.camera.capture import get_camera, create_camera
from backend.config.settings import settings
from backend.detection.detector import get_detector, load_detector
from backend.pipeline import get_pipeline, create_pipeline

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.post("/start")
async def start_monitoring() -> dict:
    detector = get_detector()
    if detector is None or not detector.is_loaded:
        detector = load_detector()
        if not detector.is_loaded:
            raise HTTPException(status_code=503, detail="AI model failed to load.")

    loop = asyncio.get_event_loop()
    cam = get_camera()
    if cam is None:
        cam = create_camera(source=settings.camera_source_parsed, loop=loop)

    if not cam.status.running:
        ok = cam.start()
        if not ok:
            raise HTTPException(
                status_code=503,
                detail=f"Cannot open camera: {cam.status.last_error}",
            )

    pipeline = get_pipeline()
    if pipeline is None or not pipeline._running:
        # Import broadcast function lazily to avoid circular import
        from backend.api.ws_routes import broadcast_ws
        pipeline = create_pipeline(
            detector=detector,
            frame_queue=cam.queue,
            broadcast_fn=broadcast_ws,
        )
        pipeline.start()

    return {"message": "Monitoring started."}


@router.post("/stop")
async def stop_monitoring() -> dict:
    pipeline = get_pipeline()
    if pipeline:
        await pipeline.stop()

    cam = get_camera()
    if cam and cam.status.running:
        cam.stop()

    return {"message": "Monitoring stopped."}


@router.get("/status")
async def monitoring_status() -> dict:
    pipeline = get_pipeline()
    cam = get_camera()

    cam_running = cam.status.running if cam else False
    pipe_running = pipeline._running if pipeline else False

    metrics = pipeline.metrics.to_dict() if pipeline else {}
    frame_b64 = pipeline.latest_frame_b64 if pipeline else None

    return {
        "camera_running": cam_running,
        "pipeline_running": pipe_running,
        "metrics": metrics,
        "has_frame": frame_b64 is not None,
    }


@router.get("/frame")
async def get_latest_frame() -> dict:
    pipeline = get_pipeline()
    if pipeline is None or not pipeline._running:
        raise HTTPException(status_code=404, detail="No active monitoring session.")
    frame_b64 = pipeline.latest_frame_b64
    if frame_b64 is None:
        raise HTTPException(status_code=503, detail="No frame available yet.")
    return {"frame": frame_b64, "format": "jpeg_base64"}

