"""
Incident event management.

Handles deduplication, incident creation, and DB persistence.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.constants import (
    VIOLATION_TYPE_HELMET,
    VIOLATION_TYPE_TRIPLE,
    REVIEW_STATUS_PENDING,
    OUTCOME_POTENTIAL,
)
from backend.config.settings import settings
from backend.database.models import Incident
from backend.storage.evidence_store import save_evidence_image

import numpy as np


# (violation_type, track_id) -> last incident timestamp
_dedup_cache: Dict[Tuple[str, int], datetime] = {}


def _dedup_key(violation_type: str, track_id: int) -> Tuple[str, int]:
    return (violation_type, track_id)


def is_duplicate(
    violation_type: str,
    track_id: int,
    dedup_seconds: int | None = None,
) -> bool:
    """Return True if a recent incident exists for this type+track combination."""
    key = _dedup_key(violation_type, track_id)
    last = _dedup_cache.get(key)
    if last is None:
        return False
    window = dedup_seconds or settings.dedup_seconds
    now = datetime.now(timezone.utc)
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now - last).total_seconds() < window


def record_dedup_timestamp(violation_type: str, track_id: int) -> None:
    key = _dedup_key(violation_type, track_id)
    _dedup_cache[key] = datetime.now(timezone.utc)


def prune_dedup_cache() -> None:
    """Remove stale entries from the dedup cache."""
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=settings.dedup_seconds * 2)
    stale = []
    for k, ts in _dedup_cache.items():
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        if ts < cutoff:
            stale.append(k)
    for k in stale:
        del _dedup_cache[k]


async def create_incident(
    db: AsyncSession,
    violation_type: str,
    track_id: int,
    confidence: float,
    camera_source: str,
    frame: Optional[np.ndarray],
    metadata: dict,
    outcome: str = OUTCOME_POTENTIAL,
    rider_count: Optional[int] = None,
    vehicle_number: Optional[str] = None,
) -> Optional[Incident]:
    """
    Create and persist an incident, saving an evidence image.
    Skips creation if a duplicate exists within the dedup window.
    """
    if is_duplicate(violation_type, track_id):
        logger.debug(f"Duplicate suppressed: {violation_type} track={track_id}")
        return None

    record_dedup_timestamp(violation_type, track_id)

    incident_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    evidence_path: Optional[str] = None
    if frame is not None:
        evidence_path = save_evidence_image(
            frame, violation_type, timestamp=now, incident_id=incident_id[:8]
        )

    incident = Incident(
        id=incident_id,
        violation_type=violation_type,
        event_timestamp=now,
        camera_source=camera_source,
        track_id=track_id,
        confidence=round(confidence, 4),
        detection_metadata=metadata,
        evidence_image_path=evidence_path,
        review_status=REVIEW_STATUS_PENDING,
        outcome=outcome,
        rider_count=rider_count,
        vehicle_number=vehicle_number,
    )

    db.add(incident)
    await db.commit()
    await db.refresh(incident)

    logger.info(
        f"Incident created: {violation_type} | track={track_id} | conf={confidence:.2f}"
    )
    return incident

