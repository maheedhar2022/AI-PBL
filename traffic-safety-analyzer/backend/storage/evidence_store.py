"""
Evidence image storage.

Saves annotated frame crops as JPEG evidence images with structured filenames.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from backend.config.settings import settings


def save_evidence_image(
    frame: np.ndarray,
    violation_type: str,
    timestamp: datetime | None = None,
    incident_id: str | None = None,
) -> str | None:
    """
    Save frame as evidence image.

    Returns the relative path string or None on failure.
    """
    try:
        settings.ensure_dirs()
        ts = timestamp or datetime.now(timezone.utc)
        inc_id = incident_id or str(uuid.uuid4())[:8]
        filename = (
            f"{violation_type}_{ts.strftime('%Y%m%d_%H%M%S')}_{inc_id}.jpg"
        )
        path = settings.evidence_dir / filename
        success = cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if success:
            logger.debug(f"Evidence saved: {path}")
            return str(path)
        else:
            logger.warning(f"cv2.imwrite returned False for {path}")
            return None
    except Exception as exc:
        logger.error(f"Failed to save evidence: {exc}")
        return None

