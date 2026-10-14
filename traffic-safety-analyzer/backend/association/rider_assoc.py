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
    motorcycle bounding box, or if the person's horizontal span and
    vertical position indicates they are riding the motorcycle.
    """
    min_iou = min_iou or settings.min_rider_association_iou
    result: Dict[int, List[int]] = {v.track_id: [] for v in vehicle_tracks}

    for person in person_tracks:
        px1, py1, px2, py2 = person.bbox
        pw = max(px2 - px1, 1)
        ph = max(py2 - py1, 1)
        pcx = (px1 + px2) // 2
        p_area = pw * ph

        best_moto: int | None = None
        best_score: float = 0.0

        for vehicle in vehicle_tracks:
            vx1, vy1, vx2, vy2 = vehicle.bbox
            vw = max(vx2 - vx1, 1)
            vh = max(vy2 - vy1, 1)

            # Expand vehicle box to cover rider above and slightly wider
            expansion_y = int(vh * 0.7)
            expansion_x = int(vw * 0.25)
            exp_x1 = max(0, vx1 - expansion_x)
            exp_y1 = max(0, vy1 - expansion_y)
            exp_x2 = vx2 + expansion_x
            exp_y2 = vy2 + int(vh * 0.2)
            expanded_bbox = (exp_x1, exp_y1, exp_x2, exp_y2)

            iou = _box_iou(person.bbox, expanded_bbox)

            # Overlap with respect to person footprint
            ix1, iy1 = max(px1, exp_x1), max(py1, exp_y1)
            ix2, iy2 = min(px2, exp_x2), min(py2, exp_y2)
            inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            person_overlap = inter_area / float(p_area)

            # Containment & alignment
            h_inter = max(0, min(px2, exp_x2) - max(px1, exp_x1))
            h_ratio = h_inter / float(pw)
            contained = (
                (exp_x1 <= pcx <= exp_x2)
                and (exp_y1 <= py2 <= (vy2 + 25))
            )

            score = max(iou, person_overlap * 0.8)
            if contained and h_ratio >= 0.25:
                score += 0.35

            if score > best_score:
                best_score = score
                best_moto = vehicle.track_id

        if best_moto is not None and best_score >= min_iou:
            result[best_moto].append(person.track_id)

    return result

