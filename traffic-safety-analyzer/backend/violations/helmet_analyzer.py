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
from backend.models.helmet_classifier import DeepHelmetClassifier


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
    deep_predictions: List[Dict[str, float]] = field(default_factory=list)

    def clear(self) -> None:
        self.no_helmet_frames = 0
        self.helmet_frames = 0
        self.uncertain_frames = 0
        self.total_frames = 0
        self.deep_predictions.clear()


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
    Accumulates per-track evidence and classifies helmet status using
    a hybrid Deep Learning CNN (DeepHelmetCNN-SE) and head-region computer vision.
    """

    def __init__(
        self,
        frames_required: int | None = None,
        deep_classifier: Optional[DeepHelmetClassifier] = None,
    ) -> None:
        self._frames_required = frames_required or settings.helmet_frames_required
        self._evidence: Dict[int, HelmetEvidence] = {}
        self._deep_classifier = deep_classifier

    def _get_deep_classifier(self) -> DeepHelmetClassifier:
        if self._deep_classifier is None:
            self._deep_classifier = DeepHelmetClassifier()
        return self._deep_classifier

    def _analyze_rider_head(
        self, frame: np.ndarray, bbox: tuple[int, int, int, int]
    ) -> Tuple[str, Dict[str, float]]:
        """
        Analyze head region using DeepHelmetCNN and texture/color distribution.
        Returns (status, probabilities_dict).
        """
        rx1, ry1, rx2, ry2 = bbox
        rw = rx2 - rx1
        rh = ry2 - ry1
        default_probs = {"helmet": 0.33, "no_helmet": 0.33, "uncertain": 0.34}
        if rw < 8 or rh < 8:
            return OUTCOME_UNCERTAIN, default_probs

        head_h = max(int(rh * 0.38), 6)
        head_y2 = min(frame.shape[0], ry1 + head_h)
        head_x1 = max(0, rx1)
        head_x2 = min(frame.shape[1], rx2)
        head_crop = frame[max(0, ry1):head_y2, head_x1:head_x2]

        if head_crop.shape[0] < 5 or head_crop.shape[1] < 5:
            return OUTCOME_UNCERTAIN, default_probs

        # 1. Deep Learning CNN Classification
        try:
            clf = self._get_deep_classifier()
            deep_label, deep_conf, probs = clf.classify_crop(head_crop)
        except Exception as e:
            logger.warning(f"Deep helmet classifier error: {e}")
            deep_label, deep_conf, probs = OUTCOME_UNCERTAIN, 0.33, default_probs

        # 2. Heuristic validation (skin/hair exposure)
        try:
            hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
            skin1 = cv2.inRange(hsv, np.array([0, 30, 60], dtype=np.uint8), np.array([25, 255, 255], dtype=np.uint8))
            skin2 = cv2.inRange(hsv, np.array([170, 30, 60], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8))
            skin_mask = cv2.bitwise_or(skin1, skin2)
            hair_mask = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 110, 50], dtype=np.uint8))

            total_pixels = head_crop.shape[0] * head_crop.shape[1]
            exposure_ratio = (cv2.countNonZero(skin_mask) + cv2.countNonZero(hair_mask)) / max(total_pixels, 1)

            # Ensemble deep CNN with physical exposure
            if exposure_ratio > 0.25 and (deep_label == "no_helmet" or deep_conf < 0.65):
                return "no_helmet", probs
            elif exposure_ratio < 0.15 and (deep_label == "helmet" or deep_conf < 0.65):
                return "helmet", probs
            elif deep_conf >= 0.60:
                return deep_label, probs
            else:
                # In standard Indian traffic, exposed head is no_helmet
                return "no_helmet" if exposure_ratio > 0.20 else OUTCOME_UNCERTAIN, probs
        except Exception:
            return deep_label, probs

    def update(
        self,
        moto_track_id: int,
        rider_tracks: List[Track],
        frame_detections_class_ids: List[int],
        frame: Optional[np.ndarray] = None,
        moto_bbox: Optional[Tuple[int, int, int, int]] = None,
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
        elif frame is not None:
            # Deep Learning CNN + Computer vision head-region ensemble
            head_boxes = [r.bbox for r in rider_tracks] if rider_tracks else []
            # Fallback to top 40% of motorcycle bounding box if person wasn't separately segmented
            if not head_boxes and moto_bbox is not None:
                mx1, my1, mx2, my2 = moto_bbox
                head_boxes = [(mx1, my1, mx2, my1 + int((my2 - my1) * 0.45))]

            if head_boxes:
                analyzed = [self._analyze_rider_head(frame, box) for box in head_boxes]
                statuses = [a[0] for a in analyzed]
                for a in analyzed:
                    ev.deep_predictions.append(a[1])

                if any(s == "no_helmet" for s in statuses):
                    ev.no_helmet_frames += 1
                elif all(s == "helmet" for s in statuses):
                    ev.helmet_frames += 1
                else:
                    ev.no_helmet_frames += 1
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

        if ratio_no_helmet >= 0.40:
            outcome = OUTCOME_POTENTIAL
            confidence = round(min(0.72 + ratio_no_helmet * 0.22, 0.96), 3)
        elif ratio_helmet >= 0.65:
            outcome = OUTCOME_NO_VIOLATION
            confidence = round(ratio_helmet, 3)
        else:
            outcome = OUTCOME_UNCERTAIN
            confidence = 0.35

        avg_deep = {}
        if ev.deep_predictions:
            avg_deep = {
                "helmet": round(float(np.mean([p.get("helmet", 0.0) for p in ev.deep_predictions])), 3),
                "no_helmet": round(float(np.mean([p.get("no_helmet", 0.0) for p in ev.deep_predictions])), 3),
                "uncertain": round(float(np.mean([p.get("uncertain", 0.0) for p in ev.deep_predictions])), 3),
            }

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
                "deep_model": "DeepHelmetCNN-SE",
                "deep_probabilities": avg_deep,
            },
        )

    def remove_track(self, moto_track_id: int) -> None:
        self._evidence.pop(moto_track_id, None)

    def reset(self) -> None:
        self._evidence.clear()

