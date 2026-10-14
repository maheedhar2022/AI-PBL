"""
Camera management API routes.
"""
from __future__ import annotations

import asyncio
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from backend.camera.capture import get_camera, create_camera, CameraStatus
from backend.config.settings import settings

router = APIRouter(prefix="/camera", tags=["camera"])


class CameraConfigRequest(BaseModel):
    source: str  # "0", "/path/to/video.mp4", "rtsp://..."
    fps_target: int = 10


class CameraStatusResponse(BaseModel):
    running: bool
    connected: bool
    source: str
    capture_fps: float
    frame_count: int
    last_error: str


@router.get("/status", response_model=CameraStatusResponse)
async def get_camera_status() -> Any:
    cam = get_camera()
    if cam is None:
        return CameraStatusResponse(
            running=False, connected=False, source="",
            capture_fps=0.0, frame_count=0, last_error="Camera not initialised.",
        )
    st = cam.status
    return CameraStatusResponse(
        running=st.running,
        connected=st.connected,
        source=st.source,
        capture_fps=st.capture_fps,
        frame_count=st.frame_count,
        last_error=st.last_error,
    )


@router.post("/configure")
async def configure_camera(body: CameraConfigRequest) -> dict:
    loop = asyncio.get_event_loop()
    cam = get_camera()
    if cam and cam.status.running:
        raise HTTPException(
            status_code=409,
            detail="Stop monitoring before changing camera source.",
        )
    create_camera(source=body.source, loop=loop)
    return {"message": "Camera configured.", "source": body.source}


@router.post("/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    valid_exts = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    ext = Path(file.filename or "").suffix.lower()
    if ext not in valid_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed formats: {sorted(list(valid_exts))}",
        )

    upload_dir = Path("./uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = file.filename or f"video_{uuid.uuid4().hex[:8]}{ext}"
    target_path = upload_dir / filename

    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    loop = asyncio.get_event_loop()
    create_camera(source=str(target_path), loop=loop)

    return {
        "message": "Video uploaded and camera configured.",
        "filename": filename,
        "source": str(target_path),
        "size_bytes": target_path.stat().st_size,
    }
