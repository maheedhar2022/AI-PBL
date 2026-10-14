"""
Settings API routes.
"""
from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import settings
from backend.database.db import get_db
from backend.database.models import AppSettings

router = APIRouter(prefix="/settings", tags=["settings"])


class SettingsUpdateRequest(BaseModel):
    confidence_threshold: Optional[float] = None
    iou_threshold: Optional[float] = None
    camera_fps_target: Optional[int] = None
    helmet_frames_required: Optional[int] = None
    triple_frames_required: Optional[int] = None
    dedup_seconds: Optional[int] = None


async def _upsert_setting(db: AsyncSession, key: str, value: str) -> None:
    row = await db.get(AppSettings, key)
    if row:
        row.value = value
    else:
        db.add(AppSettings(key=key, value=value))


async def apply_saved_settings(db: AsyncSession) -> None:
    rows = (await db.execute(select(AppSettings))).scalars().all()
    overrides = {r.key: r.value for r in rows}
    if "confidence_threshold" in overrides:
        settings.confidence_threshold = float(overrides["confidence_threshold"])
    if "iou_threshold" in overrides:
        settings.iou_threshold = float(overrides["iou_threshold"])
    if "camera_fps_target" in overrides:
        settings.camera_fps_target = int(overrides["camera_fps_target"])
    if "helmet_frames_required" in overrides:
        settings.helmet_frames_required = int(overrides["helmet_frames_required"])
    if "triple_frames_required" in overrides:
        settings.triple_frames_required = int(overrides["triple_frames_required"])
    if "dedup_seconds" in overrides:
        settings.dedup_seconds = int(overrides["dedup_seconds"])


@router.get("/")
async def get_settings(db: AsyncSession = Depends(get_db)) -> dict:
    # Check for DB overrides
    rows = (await db.execute(select(AppSettings))).scalars().all()
    overrides = {r.key: r.value for r in rows}

    return {
        "model_name": settings.model_name,
        "confidence_threshold": float(overrides.get("confidence_threshold", settings.confidence_threshold)),
        "iou_threshold": float(overrides.get("iou_threshold", settings.iou_threshold)),
        "camera_source": settings.camera_source,
        "camera_fps_target": int(overrides.get("camera_fps_target", settings.camera_fps_target)),
        "inference_width": settings.inference_width,
        "inference_height": settings.inference_height,
        "evidence_dir": str(settings.evidence_dir),
        "helmet_frames_required": int(overrides.get("helmet_frames_required", settings.helmet_frames_required)),
        "triple_frames_required": int(overrides.get("triple_frames_required", settings.triple_frames_required)),
        "dedup_seconds": int(overrides.get("dedup_seconds", settings.dedup_seconds)),
    }


@router.patch("/")
async def update_settings(
    body: SettingsUpdateRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    if body.confidence_threshold is not None:
        settings.confidence_threshold = max(0.05, min(0.95, body.confidence_threshold))
        await _upsert_setting(db, "confidence_threshold", str(settings.confidence_threshold))

    if body.iou_threshold is not None:
        settings.iou_threshold = max(0.05, min(0.95, body.iou_threshold))
        await _upsert_setting(db, "iou_threshold", str(settings.iou_threshold))

    if body.camera_fps_target is not None:
        settings.camera_fps_target = max(1, min(60, body.camera_fps_target))
        await _upsert_setting(db, "camera_fps_target", str(settings.camera_fps_target))

    if body.helmet_frames_required is not None:
        settings.helmet_frames_required = max(1, min(30, body.helmet_frames_required))
        await _upsert_setting(db, "helmet_frames_required", str(settings.helmet_frames_required))

    if body.triple_frames_required is not None:
        settings.triple_frames_required = max(1, min(30, body.triple_frames_required))
        await _upsert_setting(db, "triple_frames_required", str(settings.triple_frames_required))

    if body.dedup_seconds is not None:
        settings.dedup_seconds = max(1, min(300, body.dedup_seconds))
        await _upsert_setting(db, "dedup_seconds", str(settings.dedup_seconds))

    await db.commit()

    return {
        "message": "Settings updated successfully.",
        "settings": {
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
        },
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
