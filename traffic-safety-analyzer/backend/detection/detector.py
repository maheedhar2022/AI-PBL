"""
AI Object Detection module.

Wraps an Ultralytics YOLO model.  The model is loaded once and reused.
Returns structured DetectionResult objects for downstream modules.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import cv2
import numpy as np
from loguru import logger

from backend.config.constants import (
    COCO_NAMES,
    VEHICLE_CLASSES,
    PERSON_CLASSES,
    HELMET_PRESENT_CLASSES,
    HELMET_ABSENT_CLASSES,
)
from backend.config.settings import settings


@dataclass
class Detection:
    """A single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 in original frame coords
    is_vehicle: bool = False
    is_person: bool = False
    is_helmet_present: bool = False
    is_helmet_absent: bool = False


@dataclass
class DetectionResult:
    detections: List[Detection] = field(default_factory=list)
    inference_time_ms: float = 0.0
    frame_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    model_name: str = ""
    annotated_frame: np.ndarray | None = None

    @property
    def vehicles(self) -> List[Detection]:
        return [d for d in self.detections if d.is_vehicle]

    @property
    def persons(self) -> List[Detection]:
        return [d for d in self.detections if d.is_person]


# Annotation colours
_COLOR_VEHICLE = (0, 200, 0)
_COLOR_PERSON  = (255, 120, 0)
_COLOR_DEFAULT = (200, 200, 200)
_COLOR_HELMET_OK  = (0, 255, 100)
_COLOR_HELMET_BAD = (0, 0, 255)


class YOLODetector:
    """Singleton-style YOLO detector.  Call load() once before inference."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or settings.model_name
        self._model = None
        self._loaded = False
        self._class_names: dict[int, str] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def load(self) -> bool:
        """Load model weights.  Returns True on success."""
        try:
            import torch
            # Compatibility with PyTorch 2.6+ default weights_only=True
            orig_load = torch.load

            def _patched_load(*args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                return orig_load(*args, **kwargs)

            torch.load = _patched_load

            from ultralytics import YOLO  # lazy import to avoid startup cost
            logger.info(f"Loading model: {self._model_name}")
            self._model = YOLO(self._model_name)
            # Warm up with a dummy frame
            dummy = np.zeros((640, 640, 3), dtype=np.uint8)
            self._model.predict(dummy, verbose=False)
            self._class_names = self._model.names or COCO_NAMES
            self._loaded = True
            logger.success(f"Model loaded: {self._model_name}")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model {self._model_name!r}: {exc}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supported_classes(self) -> dict[int, str]:
        return dict(self._class_names)

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def detect(
        self,
        frame: np.ndarray,
        conf: float | None = None,
        iou: float | None = None,
    ) -> DetectionResult:
        """Run inference on a single BGR frame."""
        result = DetectionResult(model_name=self._model_name)

        if not self._loaded or self._model is None:
            logger.warning("Model not loaded; skipping detection.")
            return result

        conf = conf or settings.confidence_threshold
        iou  = iou  or settings.iou_threshold

        t0 = time.perf_counter()
        try:
            raw = self._model.predict(
                frame,
                conf=conf,
                iou=iou,
                max_det=settings.max_detections,
                verbose=False,
                stream=False,
            )
        except Exception as exc:
            logger.error(f"Inference error: {exc}")
            return result

        result.inference_time_ms = (time.perf_counter() - t0) * 1000.0

        if not raw or len(raw) == 0:
            return result

        res = raw[0]
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            result.annotated_frame = frame.copy()
            return result

        detections: List[Detection] = []
        for box in boxes:
            cls_id  = int(box.cls[0])
            conf_sc = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            is_vehicle = cls_id in VEHICLE_CLASSES
            is_person  = cls_id in PERSON_CLASSES
            is_helmet_present = cls_id in HELMET_PRESENT_CLASSES
            is_helmet_absent  = cls_id in HELMET_ABSENT_CLASSES

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf_sc,
                    bbox=(x1, y1, x2, y2),
                    is_vehicle=is_vehicle,
                    is_person=is_person,
                    is_helmet_present=is_helmet_present,
                    is_helmet_absent=is_helmet_absent,
                )
            )

        result.detections = detections
        result.annotated_frame = self._annotate(frame.copy(), detections)
        return result

    # ------------------------------------------------------------------ #
    # Annotation                                                           #
    # ------------------------------------------------------------------ #

    def _annotate(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        class_colors = {
            "car": (240, 180, 0),
            "motorcycle": (0, 220, 100),
            "bus": (0, 165, 255),
            "truck": (210, 80, 210),
            "bicycle": (60, 220, 60),
            "person": (255, 140, 0),
        }
        for d in detections:
            x1, y1, x2, y2 = d.bbox
            if d.is_helmet_absent:
                color = _COLOR_HELMET_BAD
            elif d.is_helmet_present:
                color = _COLOR_HELMET_OK
            elif d.class_name in class_colors:
                color = class_colors[d.class_name]
            elif d.is_vehicle:
                color = _COLOR_VEHICLE
            elif d.is_person:
                color = _COLOR_PERSON
            else:
                color = _COLOR_DEFAULT

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{d.class_name} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, max(th + 6, y1)), color, -1)
            cv2.putText(
                frame, label, (x1 + 2, max(th + 2, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA
            )
        return frame


# Module-level singleton
_detector: YOLODetector | None = None


def get_detector() -> YOLODetector | None:
    return _detector


def load_detector(model_name: str | None = None) -> YOLODetector:
    global _detector
    _detector = YOLODetector(model_name=model_name)
    _detector.load()
    return _detector

