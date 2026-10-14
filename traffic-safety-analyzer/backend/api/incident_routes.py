"""
Incident CRUD API routes.
"""
from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, func, update, delete
from sqlalchemy.ext.asyncio import AsyncSession

from backend.database.db import get_db
from backend.database.models import Incident

router = APIRouter(prefix="/incidents", tags=["incidents"])


# ---------- Schemas ----------

class IncidentOut(BaseModel):
    id: str
    violation_type: str
    event_timestamp: datetime
    camera_source: str
    track_id: Optional[int]
    confidence: float
    detection_metadata: dict
    evidence_image_path: Optional[str]
    review_status: str
    reviewer_notes: Optional[str]
    created_at: datetime
    outcome: str
    rider_count: Optional[int]
    model_config = ConfigDict(from_attributes=True)


class ReviewRequest(BaseModel):
    status: str   # "confirmed" or "dismissed"
    notes: Optional[str] = None


class IncidentListResponse(BaseModel):
    items: List[IncidentOut]
    total: int
    page: int
    page_size: int


class BatchReviewRequest(BaseModel):
    incident_ids: List[str]
    status: str   # "confirmed", "dismissed", or "pending"
    notes: Optional[str] = None


class BatchReviewResponse(BaseModel):
    updated_count: int
    status: str


class BulkDeleteRequest(BaseModel):
    incident_ids: List[str]


class BulkDeleteResponse(BaseModel):
    deleted_count: int


# ---------- Endpoints ----------

@router.get("/", response_model=IncidentListResponse)
async def list_incidents(
    violation_type: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    camera_source: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
) -> Any:
    q = select(Incident).order_by(Incident.event_timestamp.desc())

    if violation_type:
        q = q.where(Incident.violation_type == violation_type)
    if review_status:
        q = q.where(Incident.review_status == review_status)
    if camera_source:
        q = q.where(Incident.camera_source == camera_source)
    if start_time:
        q = q.where(Incident.event_timestamp >= start_time)
    if end_time:
        q = q.where(Incident.event_timestamp <= end_time)

    count_q = select(func.count()).select_from(q.subquery())
    total = (await db.execute(count_q)).scalar_one()

    q = q.offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(q)).scalars().all()

    return IncidentListResponse(
        items=[IncidentOut.model_validate(r) for r in rows],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/export")
async def export_incidents(
    format: str = Query("csv", pattern="^(csv|json)$"),
    violation_type: Optional[str] = Query(None),
    review_status: Optional[str] = Query(None),
    camera_source: Optional[str] = Query(None),
    start_time: Optional[datetime] = Query(None),
    end_time: Optional[datetime] = Query(None),
    db: AsyncSession = Depends(get_db),
) -> Response:
    q = select(Incident).order_by(Incident.event_timestamp.desc())

    if violation_type:
        q = q.where(Incident.violation_type == violation_type)
    if review_status:
        q = q.where(Incident.review_status == review_status)
    if camera_source:
        q = q.where(Incident.camera_source == camera_source)
    if start_time:
        q = q.where(Incident.event_timestamp >= start_time)
    if end_time:
        q = q.where(Incident.event_timestamp <= end_time)

    rows = (await db.execute(q)).scalars().all()

    if format == "json":
        data = [
            {
                "id": r.id,
                "violation_type": r.violation_type,
                "event_timestamp": r.event_timestamp.isoformat(),
                "camera_source": r.camera_source,
                "track_id": r.track_id,
                "confidence": r.confidence,
                "review_status": r.review_status,
                "outcome": r.outcome,
                "rider_count": r.rider_count,
                "reviewer_notes": r.reviewer_notes,
            }
            for r in rows
        ]
        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": "attachment; filename=incidents_export.json"},
        )
    else:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "violation_type", "event_timestamp", "camera_source",
            "track_id", "confidence", "review_status", "outcome", "rider_count", "reviewer_notes"
        ])
        for r in rows:
            writer.writerow([
                r.id,
                r.violation_type,
                r.event_timestamp.isoformat(),
                r.camera_source,
                r.track_id if r.track_id is not None else "",
                r.confidence,
                r.review_status,
                r.outcome,
                r.rider_count if r.rider_count is not None else "",
                r.reviewer_notes or "",
            ])
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=incidents_export.csv"},
        )


@router.post("/batch-review", response_model=BatchReviewResponse)
async def batch_review_incidents(
    body: BatchReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    valid_statuses = {"confirmed", "dismissed", "pending"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Choose from: {valid_statuses}",
        )

    if not body.incident_ids:
        return BatchReviewResponse(updated_count=0, status=body.status)

    values_to_update: dict = {"review_status": body.status}
    if body.notes is not None:
        values_to_update["reviewer_notes"] = body.notes

    stmt = (
        update(Incident)
        .where(Incident.id.in_(body.incident_ids))
        .values(**values_to_update)
    )
    result = await db.execute(stmt)
    await db.commit()

    return BatchReviewResponse(
        updated_count=result.rowcount or 0,
        status=body.status,
    )


@router.post("/bulk-delete", response_model=BulkDeleteResponse)
async def bulk_delete_incidents(
    body: BulkDeleteRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    if not body.incident_ids:
        return BulkDeleteResponse(deleted_count=0)

    stmt = delete(Incident).where(Incident.id.in_(body.incident_ids))
    result = await db.execute(stmt)
    await db.commit()

    return BulkDeleteResponse(deleted_count=result.rowcount or 0)


@router.get("/{incident_id}", response_model=IncidentOut)
async def get_incident(incident_id: str, db: AsyncSession = Depends(get_db)) -> Any:
    result = await db.get(Incident, incident_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    return IncidentOut.model_validate(result)


@router.patch("/{incident_id}/review", response_model=IncidentOut)
async def review_incident(
    incident_id: str,
    body: ReviewRequest,
    db: AsyncSession = Depends(get_db),
) -> Any:
    valid_statuses = {"confirmed", "dismissed", "pending"}
    if body.status not in valid_statuses:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid status. Choose from: {valid_statuses}",
        )

    incident = await db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")

    incident.review_status = body.status
    if body.notes is not None:
        incident.reviewer_notes = body.notes

    await db.commit()
    await db.refresh(incident)
    return IncidentOut.model_validate(incident)


@router.delete("/{incident_id}", status_code=204, response_class=Response)
async def delete_incident(incident_id: str, db: AsyncSession = Depends(get_db)):
    incident = await db.get(Incident, incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail="Incident not found.")
    await db.delete(incident)
    await db.commit()

