"""
Settings API routes.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database.db import get_db
from backend.database.models import AppSettings

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingUpdate(BaseModel):
    value: str


@router.get("/")
async def get_settings() -> dict:
    return {
        "model_name": settings.model_name,
        "confidence_threshold": settings.confidence_threshold,
        "iou_threshold": settings.iou_threshold,
        "camera_source": settings.camera_source,
        "camera_fps_target": settings.camera_fps_target,
        "inference_width": settings.inference_width,
        "inference_height": settings.inference_height,
        "evidence_dir": str(settings.evidence_dir),
        "helmet_frames_required": settings.helmet_frames_required,
        "triple_frames_required": settings.triple_frames_required,
        "dedup_seconds": settings.dedup_seconds,
    }


@router.get("/model-info")
async def model_info() -> dict:
    from backend.detection.detector import get_detector
    detector = get_detector()
    if detector is None or not detector.is_loaded:
        return {"loaded": False, "model_name": settings.model_name, "classes": {}}
    return {
        "loaded": True,
        "model_name": detector.model_name,
        "classes": detector.supported_classes,
    }

