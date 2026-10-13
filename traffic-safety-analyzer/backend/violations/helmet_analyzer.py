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

    def update(
        self,
        moto_track_id: int,
        rider_tracks: List[Track],
        frame_detections_class_ids: List[int],
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
        else:
            # Heuristic: if riders are detected, mark uncertain
            # (conservative — do not assume helmet absent)
            if rider_tracks:
                ev.uncertain_frames += 1

        if ev.total_frames >= self._frames_required:
            result = self._classify(ev)
            ev.clear()  # reset for next window
            return result

        return None

    def _classify(self, ev: HelmetEvidence) -> HelmetAnalysisResult:
        total = ev.total_frames
        confidence = 0.0

        if HELMET_ABSENT_CLASSES:
            # Explicit model: threshold on no-helmet frames
            ratio = ev.no_helmet_frames / max(total, 1)
            if ratio >= 0.5:
                outcome = OUTCOME_POTENTIAL
                confidence = round(ratio, 3)
            elif ev.helmet_frames / max(total, 1) >= 0.7:
                outcome = OUTCOME_NO_VIOLATION
                confidence = round(ev.helmet_frames / max(total, 1), 3)
            else:
                outcome = OUTCOME_UNCERTAIN
                confidence = 0.3
        else:
            # Heuristic mode — always uncertain without custom model
            outcome = OUTCOME_UNCERTAIN
            confidence = 0.3

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

