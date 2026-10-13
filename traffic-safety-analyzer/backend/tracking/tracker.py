"""
Multi-object tracker using the SORT-style IoU tracker bundled with ultralytics.

Maintains track IDs across frames for motorcycles and persons.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    age: int = 0          # frames since first seen
    missed: int = 0       # consecutive frames missed
    is_active: bool = True


class SimpleIOUTracker:
    """
    Lightweight IoU-based multi-object tracker.

    For production use, replace with ByteTrack via ultralytics' built-in
    tracker (model.track() with tracker='bytetrack.yaml').
    This implementation is dependency-free and sufficient for the pipeline.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missed: int = 10,
    ) -> None:
        self._iou_threshold = iou_threshold
        self._max_missed = max_missed
        self._tracks: Dict[int, Track] = {}
        self._next_id: int = 1

    # ------------------------------------------------------------------ #
    # Public                                                               #
    # ------------------------------------------------------------------ #

    def update(
        self,
        detections: List[dict],  # each: {class_id, class_name, confidence, bbox}
    ) -> List[Track]:
        """
        Match incoming detections to existing tracks and return all active tracks.
        """
        # Mark all existing tracks as unmatched
        unmatched_track_ids = set(self._tracks.keys())

        matched_pairs: List[Tuple[int, dict]] = []
        unmatched_dets: List[dict] = []

        for det in detections:
            best_iou = self._iou_threshold
            best_tid: Optional[int] = None

            for tid, track in self._tracks.items():
                if not track.is_active:
                    continue
                if track.class_id != det["class_id"]:
                    continue
                iou = _box_iou(track.bbox, det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_tid = tid

            if best_tid is not None:
                matched_pairs.append((best_tid, det))
                unmatched_track_ids.discard(best_tid)
            else:
                unmatched_dets.append(det)

        # Update matched tracks
        for tid, det in matched_pairs:
            t = self._tracks[tid]
            t.bbox = det["bbox"]
            t.confidence = det["confidence"]
            t.age += 1
            t.missed = 0
            t.is_active = True

        # Increment missed counter for unmatched tracks
        for tid in unmatched_track_ids:
            self._tracks[tid].missed += 1
            if self._tracks[tid].missed > self._max_missed:
                self._tracks[tid].is_active = False

        # Create new tracks for unmatched detections
        for det in unmatched_dets:
            tid = self._next_id
            self._next_id += 1
            self._tracks[tid] = Track(
                track_id=tid,
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                age=0,
                missed=0,
                is_active=True,
            )

        # Prune dead tracks
        dead = [tid for tid, t in self._tracks.items() if not t.is_active and t.missed > self._max_missed * 2]
        for tid in dead:
            del self._tracks[tid]

        return [t for t in self._tracks.values() if t.is_active]

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self._tracks.values() if t.is_active]

    def reset(self) -> None:
        self._tracks.clear()
        self._next_id = 1


# ------------------------------------------------------------------ #
# Utility                                                              #
# ------------------------------------------------------------------ #

def _box_iou(
    a: Tuple[int, int, int, int],
    b: Tuple[int, int, int, int],
) -> float:
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0

