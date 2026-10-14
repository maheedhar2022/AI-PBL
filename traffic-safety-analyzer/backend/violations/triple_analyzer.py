"""
Triple-riding detection rules module.

Identifies motorcycles with more than two associated riders across
consecutive frames.  Conservative: marks result as 'uncertain' when
rider count is ambiguous.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from backend.config.constants import OUTCOME_POTENTIAL, OUTCOME_NO_VIOLATION, OUTCOME_UNCERTAIN
from backend.config.settings import settings
from backend.tracking.tracker import Track
from backend.models.violation_classifier import DeepViolationRiskClassifier


@dataclass
class TripleRidingEvidence:
    moto_track_id: int
    rider_counts: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    total_frames: int = 0
    deep_evaluations: List[Dict[str, float]] = field(default_factory=list)

    def clear(self) -> None:
        self.rider_counts.clear()
        self.total_frames = 0
        self.deep_evaluations.clear()


@dataclass
class TripleRidingResult:
    moto_track_id: int
    rider_track_ids: List[int]
    estimated_rider_count: float   # average over frames
    outcome: str
    confidence: float
    frames_evaluated: int
    details: dict = field(default_factory=dict)


class TripleRidingAnalyzer:
    """
    Accumulates rider counts per motorcycle across frames and applies
    a Deep Spatio-Temporal MLP model to detect dangerous multi-rider violations.
    """

    TRIPLE_THRESHOLD = 3  # riders > 2 means triple riding

    def __init__(
        self,
        frames_required: int | None = None,
        deep_classifier: Optional[DeepViolationRiskClassifier] = None,
    ) -> None:
        self._frames_required = frames_required or settings.triple_frames_required
        self._evidence: Dict[int, TripleRidingEvidence] = {}
        self._deep_classifier = deep_classifier

    def _get_deep_classifier(self) -> DeepViolationRiskClassifier:
        if self._deep_classifier is None:
            self._deep_classifier = DeepViolationRiskClassifier()
        return self._deep_classifier

    def update(
        self,
        moto_track_id: int,
        rider_tracks: List[Track],
        moto_track: Optional[Track] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> Optional[TripleRidingResult]:
        """
        Record current rider count and evaluate deep kinematic features.
        Returns a result when sufficient evidence is accumulated.
        """
        if moto_track_id not in self._evidence:
            self._evidence[moto_track_id] = TripleRidingEvidence(
                moto_track_id=moto_track_id
            )

        ev = self._evidence[moto_track_id]
        ev.rider_counts.append(len(rider_tracks))
        ev.total_frames += 1

        # Deep spatio-temporal evaluation if motorcycle track geometry is provided
        if moto_track is not None:
            try:
                clf = self._get_deep_classifier()
                rider_boxes = [r.bbox for r in rider_tracks]
                feats = clf.extract_features(
                    moto_bbox=moto_track.bbox,
                    rider_bboxes=rider_boxes,
                    speed_px_s=moto_track.speed_px_s,
                    history=moto_track.history,
                    frame_shape=frame_shape,
                )
                _, _, probs = clf.evaluate_risk(feats)
                ev.deep_evaluations.append(probs)
            except Exception:
                pass

        if ev.total_frames >= self._frames_required:
            result = self._classify(ev, rider_tracks)
            ev.clear()  # reset for next window
            return result

        return None

    def _classify(
        self,
        ev: TripleRidingEvidence,
        current_riders: List[Track],
    ) -> TripleRidingResult:
        counts = list(ev.rider_counts)
        avg_count = sum(counts) / len(counts) if counts else 0.0
        max_count = max(counts) if counts else 0

        # Only flag if consistently seeing ≥3 riders
        frames_with_triple = sum(1 for c in counts if c >= self.TRIPLE_THRESHOLD)
        ratio = frames_with_triple / max(len(counts), 1)

        # Deep learning risk aggregation
        deep_triple_prob = 0.0
        if ev.deep_evaluations:
            deep_triple_prob = float(np.mean([e.get("triple_riding", 0.0) for e in ev.deep_evaluations]))

        if (ratio >= 0.5 and avg_count >= 2.5) or (deep_triple_prob >= 0.70 and avg_count >= 2.2):
            outcome = OUTCOME_POTENTIAL
            confidence = round(min(max(ratio + 0.1, deep_triple_prob), 1.0), 3)
        elif avg_count <= 2.0:
            outcome = OUTCOME_NO_VIOLATION
            confidence = 0.8
        else:
            outcome = OUTCOME_UNCERTAIN
            confidence = 0.3

        return TripleRidingResult(
            moto_track_id=ev.moto_track_id,
            rider_track_ids=[r.track_id for r in current_riders],
            estimated_rider_count=round(avg_count, 2),
            outcome=outcome,
            confidence=confidence,
            frames_evaluated=ev.total_frames,
            details={
                "counts": counts,
                "max_count": max_count,
                "frames_with_triple": frames_with_triple,
                "deep_model": "ViolationRiskMLP-16",
                "deep_triple_prob": round(deep_triple_prob, 3),
            },
        )

    def remove_track(self, moto_track_id: int) -> None:
        self._evidence.pop(moto_track_id, None)

    def reset(self) -> None:
        self._evidence.clear()

