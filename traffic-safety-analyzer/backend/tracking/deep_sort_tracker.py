"""
Deep SORT Multi-Object Tracker.

Combines Kalman Filter motion projection with Deep Re-ID appearance
embeddings and the Hungarian algorithm (linear sum assignment) for
robust multi-vehicle and pedestrian tracking.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment
from loguru import logger

from backend.tracking.tracker import Track, _box_iou
from backend.models.reid_extractor import DeepReIDExtractor


class KalmanBoxTracker:
    """
    Standard 8-dimensional Kalman filter for 2D bounding boxes:
    State: [x, y, a, h, vx, vy, va, vh]
    Where (x, y) is center, a = w/h (aspect ratio), h is height.
    """

    count = 0

    def __init__(self, bbox: Tuple[int, int, int, int]) -> None:
        # Time step dt = 1
        # State transition matrix F (8x8)
        self.F = np.eye(8, dtype=np.float32)
        for i in range(4):
            self.F[i, i + 4] = 1.0

        # Measurement matrix H (4x8)
        self.H = np.zeros((4, 8), dtype=np.float32)
        for i in range(4):
            self.H[i, i] = 1.0

        # Covariance matrices
        self.P = np.eye(8, dtype=np.float32) * 10.0
        for i in range(4, 8):
            self.P[i, i] *= 10.0

        self.Q = np.eye(8, dtype=np.float32) * 0.01
        for i in range(4, 8):
            self.Q[i, i] *= 0.1

        self.R = np.eye(4, dtype=np.float32) * 1.0

        # Initialize state
        x, y, a, h = self._bbox_to_z(bbox)
        self.x = np.zeros((8, 1), dtype=np.float32)
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = a
        self.x[3, 0] = h

        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history: List[Tuple[int, int]] = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    @staticmethod
    def _bbox_to_z(bbox: Tuple[int, int, int, int]) -> Tuple[float, float, float, float]:
        w = max(1e-2, float(bbox[2] - bbox[0]))
        h = max(1e-2, float(bbox[3] - bbox[1]))
        x = float(bbox[0]) + w / 2.0
        y = float(bbox[1]) + h / 2.0
        a = w / h
        return x, y, a, h

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
        c_x = float(x[0, 0])
        c_y = float(x[1, 0])
        a = max(1e-2, float(x[2, 0]))
        h = max(1e-2, float(x[3, 0]))
        w = a * h
        return (
            int(round(c_x - w / 2.0)),
            int(round(c_y - h / 2.0)),
            int(round(c_x + w / 2.0)),
            int(round(c_y + h / 2.0)),
        )

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance the state vector and return the predicted bounding box."""
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self._x_to_bbox(self.x)

    def update(self, bbox: Tuple[int, int, int, int]) -> None:
        """Update the state vector with observed bounding box."""
        self.time_since_update = 0
        self.history.append((int(self.x[0, 0]), int(self.x[1, 0])))
        if len(self.history) > 30:
            self.history.pop(0)
        self.hits += 1
        self.hit_streak += 1

        z = np.array(self._bbox_to_z(bbox), dtype=np.float32).reshape((4, 1))
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)

    def get_state(self) -> Tuple[int, int, int, int]:
        return self._x_to_bbox(self.x)


class DeepSortTracker:
    """
    Deep SORT Multi-Object Tracker with Visual Re-ID and Kalman Filtering.
    """

    def __init__(
        self,
        reid_extractor: Optional[DeepReIDExtractor] = None,
        max_age: int = 15,
        min_hits: int = 1,
        iou_threshold: float = 0.30,
        reid_weight: float = 0.40,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.reid_weight = reid_weight

        self.reid = reid_extractor or DeepReIDExtractor()
        self.trackers: List[KalmanBoxTracker] = []
        self.tracks: Dict[int, Track] = {}
        self.track_features: Dict[int, np.ndarray] = {}  # Running gallery of Re-ID embeddings
        self.frame_count = 0

    def update(
        self,
        detections: List[dict],
        frame: Optional[np.ndarray] = None,
    ) -> List[Track]:
        """
        Update tracker with new detections.
        detections: list of dicts with keys {class_id, class_name, confidence, bbox}
        frame: full BGR image for extracting deep appearance embeddings
        """
        self.frame_count += 1

        # 1. Predict positions using Kalman Filter
        predicted_boxes = []
        to_del = []
        for i, trk in enumerate(self.trackers):
            pos = trk.predict()
            if np.any(np.isnan(pos)):
                to_del.append(i)
            else:
                predicted_boxes.append(pos)

        for i in reversed(to_del):
            tid = self.trackers[i].id
            self.trackers.pop(i)
            self.tracks.pop(tid, None)
            self.track_features.pop(tid, None)

        # 2. Extract deep visual Re-ID embeddings if frame is available
        det_features: List[Optional[np.ndarray]] = []
        if frame is not None and len(detections) > 0:
            crops = []
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                x1 = max(0, min(frame.shape[1] - 1, x1))
                y1 = max(0, min(frame.shape[0] - 1, y1))
                x2 = max(x1 + 1, min(frame.shape[1], x2))
                y2 = max(y1 + 1, min(frame.shape[0], y2))
                crops.append(frame[y1:y2, x1:x2])
            det_features = self.reid.extract_batch(crops)
        else:
            det_features = [None] * len(detections)

        # 3. Associate detections to existing Kalman trackers
        matched, unmatched_dets, unmatched_trks = self._associate(
            detections, predicted_boxes, det_features
        )

        # 4. Update matched trackers
        for trk_idx, det_idx in matched:
            trk = self.trackers[trk_idx]
            det = detections[det_idx]
            trk.update(det["bbox"])

            tid = trk.id
            # Smoothly update appearance embedding gallery
            feat = det_features[det_idx]
            if feat is not None:
                if tid in self.track_features:
                    # Exponential moving average for appearance model
                    self.track_features[tid] = 0.8 * self.track_features[tid] + 0.2 * feat
                    self.track_features[tid] /= np.linalg.norm(self.track_features[tid]) + 1e-6
                else:
                    self.track_features[tid] = feat

            cx = (det["bbox"][0] + det["bbox"][2]) // 2
            cy = (det["bbox"][1] + det["bbox"][3]) // 2
            t_obj = self.tracks.get(tid)
            if t_obj:
                t_obj.bbox = trk.get_state()
                t_obj.confidence = det["confidence"]
                t_obj.age = trk.age
                t_obj.missed = trk.time_since_update
                t_obj.is_active = True
                t_obj.history.append((cx, cy))
                if len(t_obj.history) > 25:
                    t_obj.history.pop(0)
                if len(t_obj.history) >= 2:
                    dx = t_obj.history[-1][0] - t_obj.history[-2][0]
                    dy = t_obj.history[-1][1] - t_obj.history[-2][1]
                    dist = float(np.hypot(dx, dy))
                    t_obj.speed_px_s = round(0.7 * t_obj.speed_px_s + 0.3 * (dist * 10.0), 1)
            else:
                self.tracks[tid] = Track(
                    track_id=tid,
                    class_id=det["class_id"],
                    class_name=det["class_name"],
                    bbox=trk.get_state(),
                    confidence=det["confidence"],
                    age=trk.age,
                    missed=0,
                    is_active=True,
                    history=[(cx, cy)],
                    speed_px_s=0.0,
                )

        # 5. Create new trackers for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            trk = KalmanBoxTracker(det["bbox"])
            self.trackers.append(trk)
            tid = trk.id
            if det_features[det_idx] is not None:
                self.track_features[tid] = det_features[det_idx]

            cx = (det["bbox"][0] + det["bbox"][2]) // 2
            cy = (det["bbox"][1] + det["bbox"][3]) // 2
            self.tracks[tid] = Track(
                track_id=tid,
                class_id=det["class_id"],
                class_name=det["class_name"],
                bbox=det["bbox"],
                confidence=det["confidence"],
                age=1,
                missed=0,
                is_active=True,
                history=[(cx, cy)],
                speed_px_s=0.0,
            )

        # 6. Remove old tracks
        active_output: List[Track] = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            i -= 1
            tid = trk.id
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
                self.tracks.pop(tid, None)
                self.track_features.pop(tid, None)
            elif trk.hits >= self.min_hits or self.frame_count <= self.min_hits:
                if tid in self.tracks and trk.time_since_update <= 1:
                    active_output.append(self.tracks[tid])

        return active_output

    def _associate(
        self,
        detections: List[dict],
        predicted_boxes: List[Tuple[int, int, int, int]],
        det_features: List[Optional[np.ndarray]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Bipartite matching via Hungarian algorithm using combined IoU and Re-ID cost."""
        if len(self.trackers) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(self.trackers)))

        cost_matrix = np.zeros((len(self.trackers), len(detections)), dtype=np.float32)

        for t_idx, trk in enumerate(self.trackers):
            p_box = predicted_boxes[t_idx]
            tid = trk.id
            gallery_feat = self.track_features.get(tid)

            for d_idx, det in enumerate(detections):
                # Class gating: mismatched classes get infinity cost
                trk_obj = self.tracks.get(tid)
                if trk_obj and trk_obj.class_id != det["class_id"]:
                    cost_matrix[t_idx, d_idx] = 1e5
                    continue

                # 1. Spatial IoU distance
                iou = _box_iou(p_box, det["bbox"])
                iou_dist = 1.0 - iou

                # 2. Deep Re-ID cosine distance
                d_feat = det_features[d_idx]
                if gallery_feat is not None and d_feat is not None:
                    reid_dist = float(1.0 - np.dot(gallery_feat, d_feat))
                    # Weighted combination of spatial and appearance cost
                    combined_cost = (1.0 - self.reid_weight) * iou_dist + self.reid_weight * reid_dist
                else:
                    combined_cost = iou_dist

                cost_matrix[t_idx, d_idx] = combined_cost

        # Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        matched: List[Tuple[int, int]] = []
        unmatched_trks = set(range(len(self.trackers)))
        unmatched_dets = set(range(len(detections)))

        for r, c in zip(row_ind, col_ind):
            cost = cost_matrix[r, c]
            # Max allowable threshold (IoU must be at least ~0.15 or Re-ID match)
            if cost < (1.0 - self.iou_threshold + 0.3):
                matched.append((r, c))
                unmatched_trks.discard(r)
                unmatched_dets.discard(c)

        return matched, sorted(unmatched_dets), sorted(unmatched_trks)

    def get_active_tracks(self) -> List[Track]:
        return [t for t in self.tracks.values() if t.is_active]

    def reset(self) -> None:
        self.trackers.clear()
        self.tracks.clear()
        self.track_features.clear()
        self.frame_count = 0

