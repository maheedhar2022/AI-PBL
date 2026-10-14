"""
Helmet violation detection rules module.

Uses temporal evidence accumulation over multiple frames.
Standard COCO models don't have helmet classes, so this module uses a
heuristic based on head-region analysis.  If a custom helmet model is
loaded, HELMET_PRESENT_CLASSES / HELMET_ABSENT_CLASSES take precedence.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from backend.config.constants import (
    HELMET_PRESENT_CLASSES,
    HELMET_ABSENT_CLASSES,
    OUTCOME_POTENTIAL,
    OUTCOME_NO_VIOLATION,
    OUTCOME_UNCERTAIN,
)
from backend.config.settings import settings
from backend.tracking.tracker import Track


@dataclass
class HelmetEvidence:
    """Per-motorcycle-track accumulated helmet evidence."""
    moto_track_id: int
    rider_track_ids: List[int] = field(default_factory=list)
    # Frames where at least one rider showed no helmet
    no_helmet_frames: int = 0
    # Frames where all associated riders showed helmets
    helmet_frames: int = 0
    # Frames with uncertain status (no reliable classification)
    uncertain_frames: int = 0
    total_frames: int = 0

    def clear(self) -> None:
        self.no_helmet_frames = 0
        self.helmet_frames = 0
        self.uncertain_frames = 0
        self.total_frames = 0


@dataclass
class HelmetAnalysisResult:
    moto_track_id: int
    rider_track_ids: List[int]
    outcome: str
    confidence: float
    frames_evaluated: int
    details: dict = field(default_factory=dict)


class HelmetViolationAnalyzer:
    """
    Accumulates per-track evidence and classifies helmet status.

    NOTE: Without a dedicated helmet dataset/model, this module uses
    a head-region heuristic:
    - If person bbox has a tall aspect ratio and occupies sufficient
      vertical space within the motorcycle region, helmet status is
      marked 'uncertain' (conservative).
    - If HELMET_ABSENT_CLASSES are populated from a custom model,
      explicit no-helmet detections drive the classification.
    """

    def __init__(self, frames_required: int | None = None) -> None:
        self._frames_required = frames_required or settings.helmet_frames_required
        self._evidence: Dict[int, HelmetEvidence] = {}

    def _analyze_rider_head(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> str:
        """
        Analyze head region using HSV color & texture distribution.
        Returns: 'no_helmet', 'helmet', or OUTCOME_UNCERTAIN.
        """
        rx1, ry1, rx2, ry2 = bbox
        rw = rx2 - rx1
        rh = ry2 - ry1
        if rw < 15 or rh < 25:
            return OUTCOME_UNCERTAIN

        head_h = max(int(rh * 0.22), 8)
        head_y2 = min(frame.shape[0], ry1 + head_h)
        head_x1 = max(0, rx1)
        head_x2 = min(frame.shape[1], rx2)
        head_crop = frame[max(0, ry1):head_y2, head_x1:head_x2]

        if head_crop.shape[0] < 8 or head_crop.shape[1] < 8:
            return OUTCOME_UNCERTAIN

        try:
            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            # Skin tone masks (lower and upper hue regions)
            skin1 = cv2.inRange(hsv, np.array([0, 30, 60], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
            skin2 = cv2.inRange(hsv, np.array([170, 30, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
            skin_mask = cv2.bitwise_or(skin1, skin2)

            # Dark hair mask
            hair_mask = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 110, 50], dtype=np.uint8))

            total_pixels = head_crop.shape[0] * head_crop.shape[1]
            skin_ratio = cv2.countNonZero(skin_mask) / max(total_pixels, 1)
            hair_ratio = cv2.countNonZero(hair_mask) / max(total_pixels, 1)

            gray_head = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
            std_dev = float(np.std(gray_head))

            # Bare face/hair exposed: high skin+hair and texture variation
            if (skin_ratio + hair_ratio) > 0.42 and std_dev > 15.0:
                return "no_helmet"
            elif (skin_ratio + hair_ratio) < 0.18 and std_dev < 32.0:
                return "helmet"
            else:
                return OUTCOME_UNCERTAIN
        except Exception:
            return OUTCOME_UNCERTAIN

    def update(
        self,
        moto_track_id: int,
        rider_tracks: List[Track],
        frame_detections_class_ids: List[int],
        frame: Optional[np.ndarray] = None,
    ) -> Optional[HelmetAnalysisResult]:
        """
        Process one frame for a motorcycle and its associated riders.
        Returns a result when sufficient evidence is accumulated.
        """
        if moto_track_id not in self._evidence:
            self._evidence[moto_track_id] = HelmetEvidence(
                moto_track_id=moto_track_id
            )

        ev = self._evidence[moto_track_id]
        ev.rider_track_ids = [r.track_id for r in rider_tracks]
        ev.total_frames += 1

        has_custom_helmet_classes = bool(HELMET_PRESENT_CLASSES or HELMET_ABSENT_CLASSES)

        if has_custom_helmet_classes:
            # Explicit model support
            any_no_helmet = any(
                cls_id in HELMET_ABSENT_CLASSES for cls_id in frame_detections_class_ids
            )
            all_helmet_ok = all(
                cls_id in HELMET_PRESENT_CLASSES for cls_id in frame_detections_class_ids
            ) and bool(frame_detections_class_ids)

            if any_no_helmet:
                ev.no_helmet_frames += 1
            elif all_helmet_ok:
                ev.helmet_frames += 1
            else:
                ev.uncertain_frames += 1
        elif frame is not None and rider_tracks:
            # Computer vision head-region heuristic
            statuses = [self._analyze_rider_head(frame, r.bbox) for r in rider_tracks]
            if any(s == "no_helmet" for s in statuses):
                ev.no_helmet_frames += 1
            elif all(s == "helmet" for s in statuses):
                ev.helmet_frames += 1
            else:
                ev.uncertain_frames += 1
        else:
            # Conservative heuristic when no frame is passed
            if rider_tracks:
                ev.uncertain_frames += 1

        if ev.total_frames >= self._frames_required:
            result = self._classify(ev)
            ev.clear()  # reset for next window
            return result

        return None

    def _classify(self, ev: HelmetEvidence) -> HelmetAnalysisResult:
        total = ev.total_frames
        ratio_no_helmet = ev.no_helmet_frames / max(total, 1)
        ratio_helmet = ev.helmet_frames / max(total, 1)

        if ratio_no_helmet >= 0.5:
            outcome = OUTCOME_POTENTIAL
            confidence = round(min(0.65 + ratio_no_helmet * 0.25, 0.95), 3)
        elif ratio_helmet >= 0.65:
            outcome = OUTCOME_NO_VIOLATION
            confidence = round(ratio_helmet, 3)
        else:
            outcome = OUTCOME_UNCERTAIN
            confidence = 0.35

        return HelmetAnalysisResult(
            moto_track_id=ev.moto_track_id,
            rider_track_ids=list(ev.rider_track_ids),
            outcome=outcome,
            confidence=confidence,
            frames_evaluated=total,
            details={
                "no_helmet_frames": ev.no_helmet_frames,
                "helmet_frames": ev.helmet_frames,
                "uncertain_frames": ev.uncertain_frames,
                "custom_model_used": bool(HELMET_ABSENT_CLASSES),
            },
        )

    def remove_track(self, moto_track_id: int) -> None:
        self._evidence.pop(moto_track_id, None)

    def reset(self) -> None:
        self._evidence.clear()

