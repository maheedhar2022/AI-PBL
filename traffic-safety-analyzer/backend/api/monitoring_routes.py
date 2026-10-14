"""
Monitoring start/stop and recording API routes.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

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
    is_rec = pipeline.is_recording if pipeline else False

    metrics = pipeline.metrics.to_dict() if pipeline else {}
    frame_b64 = pipeline.latest_frame_b64 if pipeline else None

    return {
        "camera_running": cam_running,
        "pipeline_running": pipe_running,
        "is_recording": is_rec,
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


@router.post("/record/start")
async def start_recording() -> dict:
    pipeline = get_pipeline()
    if pipeline is None or not pipeline._running:
        raise HTTPException(status_code=400, detail="Start monitoring before recording.")

    filename = pipeline.start_recording()
    return {"message": "Recording started.", "filename": filename}


@router.post("/record/stop")
async def stop_recording() -> dict:
    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=400, detail="Pipeline not initialized.")

    info = pipeline.stop_recording()
    if not info:
        raise HTTPException(status_code=400, detail="No active recording was running.")
    return {"message": "Recording stopped.", "recording": info}


@router.get("/recordings")
async def list_recordings() -> dict:
    rec_dir = settings.evidence_dir / "recordings"
    rec_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for f in sorted(rec_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append({
            "filename": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"recordings": items}


@router.get("/recordings/{filename}")
async def get_recording_file(filename: str) -> FileResponse:
    file_path = settings.evidence_dir / "recordings" / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Recording not found.")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@router.delete("/recordings/{filename}")
async def delete_recording(filename: str) -> dict:
    file_path = settings.evidence_dir / "recordings" / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Recording not found.")
    file_path.unlink()
    return {"message": "Recording deleted.", "filename": filename}


@router.post("/snapshot")
async def take_snapshot() -> dict:
    pipeline = get_pipeline()
    if pipeline and pipeline._running:
        info = pipeline.save_snapshot()
        if info:
            return {"message": "Snapshot captured.", "snapshot": info}

    cam = get_camera()
    if cam and cam.status.running:
        try:
            if cam._cap:
                ret, frame = cam._cap.read()
                if ret and frame is not None:
                    snapshot_dir = settings.evidence_dir / "snapshots"
                    snapshot_dir.mkdir(parents=True, exist_ok=True)
                    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    filename = f"snapshot_{ts}.jpg"
                    filepath = snapshot_dir / filename
                    cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    stat = filepath.stat()
                    h, w = frame.shape[:2]
                    return {
                        "message": "Snapshot captured from camera.",
                        "snapshot": {
                            "filename": filename,
                            "path": str(filepath),
                            "resolution": f"{w}x{h}",
                            "size_kb": round(stat.st_size / 1024, 1),
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    }
        except Exception:
            pass

    if pipeline is None or not pipeline._running:
        raise HTTPException(status_code=400, detail="Start monitoring before taking snapshot.")
    raise HTTPException(status_code=503, detail="No frame available to snapshot yet. Please wait a moment.")


@router.get("/snapshots")
async def list_snapshots() -> dict:
    snap_dir = settings.evidence_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)

    items = []
    for f in sorted(snap_dir.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True):
        stat = f.stat()
        items.append({
            "filename": f.name,
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return {"snapshots": items}


@router.get("/snapshots/{filename}")
async def get_snapshot_file(filename: str) -> FileResponse:
    file_path = settings.evidence_dir / "snapshots" / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return FileResponse(file_path, media_type="image/jpeg", filename=filename)


@router.delete("/snapshots/{filename}")
async def delete_snapshot(filename: str) -> dict:
    file_path = settings.evidence_dir / "snapshots" / filename
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    file_path.unlink()
    return {"message": "Snapshot deleted.", "filename": filename}

