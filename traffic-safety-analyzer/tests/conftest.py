"""
pytest configuration: use an in-memory SQLite DB for tests.
"""
import os
import pytest

# Override DB to in-memory for tests
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("CAMERA_SOURCE", "0")
os.environ.setdefault("MODEL_NAME", "yolov8n.pt")
os.environ.setdefault("EVIDENCE_DIR", "/tmp/test_evidence")

