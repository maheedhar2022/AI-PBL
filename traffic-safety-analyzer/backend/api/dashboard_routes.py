"""
Dashboard statistics API routes.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from backend.camera.capture import get_camera
from backend.database.db import get_db
from backend.database.models import Incident
from backend.pipeline import get_pipeline

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/stats")
async def dashboard_stats(db: AsyncSession = Depends(get_db)) -> Any:
    total = (await db.execute(select(func.count(Incident.id)))).scalar_one()
    helmet = (
        await db.execute(
            select(func.count(Incident.id)).where(
                Incident.violation_type == "helmet_violation"
            )
        )
    ).scalar_one()
    triple = (
        await db.execute(
            select(func.count(Incident.id)).where(
                Incident.violation_type == "triple_riding"
            )
        )
    ).scalar_one()
    pending = (
        await db.execute(
            select(func.count(Incident.id)).where(
                Incident.review_status == "pending"
            )
        )
    ).scalar_one()

    cam = get_camera()
    pipeline = get_pipeline()

    return {
        "total_incidents": total,
        "helmet_incidents": helmet,
        "triple_riding_incidents": triple,
        "pending_review": pending,
        "camera_running": cam.status.running if cam else False,
        "pipeline_running": pipeline._running if pipeline else False,
        "pipeline_metrics": pipeline.metrics.to_dict() if pipeline else {},
    }


@router.get("/recent")
async def recent_incidents(
    limit: int = 10,
    db: AsyncSession = Depends(get_db),
) -> Any:
    q = (
        select(Incident)
        .order_by(Incident.event_timestamp.desc())
        .limit(limit)
    )
    rows = (await db.execute(q)).scalars().all()
    return [
        {
            "id": r.id,
            "violation_type": r.violation_type,
            "event_timestamp": r.event_timestamp.isoformat(),
            "confidence": r.confidence,
            "review_status": r.review_status,
            "outcome": r.outcome,
        }
        for r in rows
    ]

