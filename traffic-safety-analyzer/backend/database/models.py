"""
ORM models for incidents, experiments, and application settings.
"""
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.db import Base


def _new_uuid() -> str:
    return str(uuid.uuid4())


class Incident(Base):
    """A detected potential traffic violation."""

    __tablename__ = "incidents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    violation_type: Mapped[str] = mapped_column(String(64), index=True)
    event_timestamp: Mapped[datetime] = mapped_column(DateTime, index=True)
    camera_source: Mapped[str] = mapped_column(String(256))
    track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    confidence: Mapped[float] = mapped_column(Float)
    detection_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    evidence_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    reviewer_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    outcome: Mapped[str] = mapped_column(String(64), default="potential_violation")
    rider_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vehicle_number: Mapped[str | None] = mapped_column(String(32), nullable=True, index=True)


class ExperimentResult(Base):
    """Stores the output of an evaluation run."""

    __tablename__ = "experiment_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_uuid)
    experiment_name: Mapped[str] = mapped_column(String(256))
    model_name: Mapped[str] = mapped_column(String(256))
    dataset_name: Mapped[str] = mapped_column(String(256))
    run_timestamp: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    config: Mapped[dict] = mapped_column(JSON, default=dict)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    results_path: Mapped[str | None] = mapped_column(Text, nullable=True)


class AppSettings(Base):
    """Key-value store for runtime-adjustable settings."""

    __tablename__ = "app_settings"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

