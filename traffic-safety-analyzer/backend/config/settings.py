"""
Application-wide configuration loaded from environment variables / .env file.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=(),
    )

    # Application
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    # Camera
    camera_source: str = "0"  # "0" = webcam; path for file
    camera_fps_target: int = 10
    inference_width: int = 640
    inference_height: int = 640

    # AI Model
    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.40
    iou_threshold: float = 0.45
    max_detections: int = 50

    # Violation rules
    helmet_frames_required: int = 5
    triple_frames_required: int = 5
    min_rider_association_iou: float = 0.10
    dedup_seconds: int = 30

    # Storage
    evidence_dir: Path = Path("./evidence")
    database_url: str = "sqlite+aiosqlite:///./traffic_safety.db"

    # Evaluation
    dataset_dir: Path = Path("./research/datasets")
    results_dir: Path = Path("./research/reports")

    def ensure_dirs(self) -> None:
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    @property
    def camera_source_parsed(self) -> int | str:
        """Return int for webcam index, str for file/RTSP."""
        try:
            return int(self.camera_source)
        except ValueError:
            return self.camera_source


settings = Settings()

