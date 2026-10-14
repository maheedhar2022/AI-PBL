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


@router.get("/analytics")
async def dashboard_analytics(db: AsyncSession = Depends(get_db)) -> dict:
    # Violation breakdown
    violations_q = await db.execute(
        select(Incident.violation_type, func.count(Incident.id))
        .group_by(Incident.violation_type)
    )
    v_counts = {row[0]: row[1] for row in violations_q.all()}

    # Review status breakdown
    reviews_q = await db.execute(
        select(Incident.review_status, func.count(Incident.id))
        .group_by(Incident.review_status)
    )
    r_counts = {row[0]: row[1] for row in reviews_q.all()}

    # Hourly/Daily incident trend
    rows = (
        await db.execute(
            select(Incident.event_timestamp, Incident.violation_type)
            .order_by(Incident.event_timestamp.desc())
            .limit(100)
        )
    ).all()

    timeline_map: dict[str, dict[str, int]] = {}
    for ts, vtype in reversed(rows):
        day_str = ts.strftime("%m-%d %H:00") if hasattr(ts, "strftime") else str(ts)[:13]
        if day_str not in timeline_map:
            timeline_map[day_str] = {"time": day_str, "helmet": 0, "triple": 0}
        if vtype == "helmet_violation":
            timeline_map[day_str]["helmet"] += 1
        elif vtype == "triple_riding":
            timeline_map[day_str]["triple"] += 1

    timeline = list(timeline_map.values())[-12:]

    pipeline = get_pipeline()
    metrics = pipeline.metrics.to_dict() if pipeline else {}

    return {
        "violation_distribution": [
            {"name": "Helmet Violation", "value": v_counts.get("helmet_violation", 0), "color": "#ef4444"},
            {"name": "Triple Riding", "value": v_counts.get("triple_riding", 0), "color": "#f59e0b"},
        ],
        "review_distribution": [
            {"name": "Pending", "value": r_counts.get("pending", 0), "color": "#eab308"},
            {"name": "Confirmed", "value": r_counts.get("confirmed", 0), "color": "#22c55e"},
            {"name": "Dismissed", "value": r_counts.get("dismissed", 0), "color": "#6b7280"},
        ],
        "timeline": timeline,
        "traffic_metrics": metrics,
    }

