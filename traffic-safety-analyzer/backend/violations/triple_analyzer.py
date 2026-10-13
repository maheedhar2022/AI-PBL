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


@dataclass
class TripleRidingEvidence:
    moto_track_id: int
    rider_counts: Deque[int] = field(default_factory=lambda: deque(maxlen=10))
    total_frames: int = 0

    def clear(self) -> None:
        self.rider_counts.clear()
        self.total_frames = 0


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
    Accumulates rider counts per motorcycle across frames and emits
    potential triple-riding events.
    """

    TRIPLE_THRESHOLD = 3  # riders > 2 means triple riding

    def __init__(self, frames_required: int | None = None) -> None:
        self._frames_required = frames_required or settings.triple_frames_required
        self._evidence: Dict[int, TripleRidingEvidence] = {}

    def update(
        self,
        moto_track_id: int,
        rider_tracks: List[Track],
    ) -> Optional[TripleRidingResult]:
        """
        Record current rider count for a motorcycle.
        Returns a result when sufficient evidence is accumulated.
        """
        if moto_track_id not in self._evidence:
            self._evidence[moto_track_id] = TripleRidingEvidence(
                moto_track_id=moto_track_id
            )

        ev = self._evidence[moto_track_id]
        ev.rider_counts.append(len(rider_tracks))
        ev.total_frames += 1

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

        if ratio >= 0.5 and avg_count >= 2.5:
            outcome = OUTCOME_POTENTIAL
            confidence = round(min(ratio + 0.1, 1.0), 3)
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
            },
        )

    def remove_track(self, moto_track_id: int) -> None:
        self._evidence.pop(moto_track_id, None)

    def reset(self) -> None:
        self._evidence.clear()

