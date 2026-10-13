"""
Rider-to-motorcycle spatial association.

Determines which persons are associated with which motorcycle based on
bounding-box overlap / proximity heuristics.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple
from backend.tracking.tracker import Track, _box_iou
from backend.config.settings import settings


def associate_riders_to_motorcycles(
    vehicle_tracks: List[Track],
    person_tracks: List[Track],
    min_iou: float | None = None,
) -> Dict[int, List[int]]:
    """
    Return mapping: motorcycle track_id -> [person track_id, ...]

    A person is associated if their bounding box overlaps the expanded
    motorcycle bounding box by at least min_iou, OR if the person's
    lower-body centroid is within the motorcycle box.
    """
    min_iou = min_iou or settings.min_rider_association_iou
    result: Dict[int, List[int]] = {v.track_id: [] for v in vehicle_tracks}

    for person in person_tracks:
        best_moto: int | None = None
        best_score: float = 0.0

        for vehicle in vehicle_tracks:
            # Expand vehicle box vertically to catch riders above it
            vx1, vy1, vx2, vy2 = vehicle.bbox
            expansion_y = int((vy2 - vy1) * 0.5)
            expanded_bbox = (vx1, max(0, vy1 - expansion_y), vx2, vy2 + expansion_y)

            iou = _box_iou(person.bbox, expanded_bbox)

            # Centroid containment bonus
            px1, py1, px2, py2 = person.bbox
            cx = (px1 + px2) // 2
            cy = py2  # bottom centre of person (foot position)
            contained = (vx1 <= cx <= vx2 and vy1 <= cy <= vy2 + 20)

            score = iou + (0.3 if contained else 0.0)
            if score > best_score:
                best_score = score
                best_moto = vehicle.track_id

        if best_moto is not None and best_score >= min_iou:
            result[best_moto].append(person.track_id)

    return result

