"""
Unit tests for violation rule modules and event deduplication.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

# ──────────────────────────────────────────────────────────────────────────────
# Tracker tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.tracking.tracker import SimpleIOUTracker, _box_iou


def test_iou_perfect_overlap():
    assert _box_iou((0, 0, 100, 100), (0, 0, 100, 100)) == pytest.approx(1.0)


def test_iou_no_overlap():
    assert _box_iou((0, 0, 50, 50), (60, 60, 100, 100)) == pytest.approx(0.0)


def test_iou_partial_overlap():
    iou = _box_iou((0, 0, 100, 100), (50, 50, 150, 150))
    assert 0 < iou < 1


def test_tracker_assigns_id():
    tracker = SimpleIOUTracker()
    dets = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.9, "bbox": (10, 10, 100, 100)}]
    tracks = tracker.update(dets)
    assert len(tracks) == 1
    assert tracks[0].track_id == 1


def test_tracker_maintains_id_across_frames():
    tracker = SimpleIOUTracker()
    dets = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.9, "bbox": (10, 10, 100, 100)}]
    tracks1 = tracker.update(dets)
    # Slightly moved box
    dets2 = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.8, "bbox": (12, 12, 102, 102)}]
    tracks2 = tracker.update(dets2)
    assert tracks1[0].track_id == tracks2[0].track_id


def test_tracker_creates_new_id_for_new_object():
    tracker = SimpleIOUTracker()
    dets1 = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.9, "bbox": (10, 10, 100, 100)}]
    tracker.update(dets1)
    # Far away new object
    dets2 = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.9, "bbox": (500, 500, 600, 600)}]
    tracks2 = tracker.update(dets2)
    assert 2 in [t.track_id for t in tracks2]


def test_tracker_expires_missed_tracks():
    tracker = SimpleIOUTracker(max_missed=2)
    dets = [{"class_id": 3, "class_name": "motorcycle", "confidence": 0.9, "bbox": (10, 10, 100, 100)}]
    tracker.update(dets)
    # No detections for 5 frames
    for _ in range(5):
        tracks = tracker.update([])
    active_ids = [t.track_id for t in tracks]
    assert 1 not in active_ids


# ──────────────────────────────────────────────────────────────────────────────
# Rider association tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.association.rider_assoc import associate_riders_to_motorcycles
from backend.tracking.tracker import Track


def make_track(tid, cls_id, bbox):
    return Track(track_id=tid, class_id=cls_id, class_name="test", bbox=bbox, confidence=0.9)


def test_association_basic():
    moto = make_track(1, 3, (100, 100, 300, 250))
    rider = make_track(2, 0, (120, 60, 260, 240))
    assoc = associate_riders_to_motorcycles([moto], [rider], min_iou=0.05)
    assert 2 in assoc[1]


def test_association_no_overlap():
    moto = make_track(1, 3, (100, 100, 300, 250))
    rider = make_track(2, 0, (600, 600, 700, 700))
    assoc = associate_riders_to_motorcycles([moto], [rider])
    assert assoc[1] == []


# ──────────────────────────────────────────────────────────────────────────────
# Helmet analyzer tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.violations.helmet_analyzer import HelmetViolationAnalyzer
from backend.config.constants import OUTCOME_UNCERTAIN


def test_helmet_analyzer_returns_none_before_threshold():
    analyzer = HelmetViolationAnalyzer(frames_required=5)
    result = analyzer.update(
        moto_track_id=1,
        rider_tracks=[make_track(2, 0, (0, 0, 50, 100))],
        frame_detections_class_ids=[0],
    )
    assert result is None  # Not enough frames yet


def test_helmet_analyzer_returns_result_after_threshold():
    analyzer = HelmetViolationAnalyzer(frames_required=3)
    result = None
    for _ in range(3):
        result = analyzer.update(
            moto_track_id=1,
            rider_tracks=[make_track(2, 0, (0, 0, 50, 100))],
            frame_detections_class_ids=[0],
        )
    assert result is not None
    # Without custom model classes, result should be uncertain
    assert result.outcome == OUTCOME_UNCERTAIN


# ──────────────────────────────────────────────────────────────────────────────
# Triple riding analyzer tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.violations.triple_analyzer import TripleRidingAnalyzer
from backend.config.constants import OUTCOME_POTENTIAL, OUTCOME_NO_VIOLATION


def test_triple_analyzer_no_violation_with_two_riders():
    analyzer = TripleRidingAnalyzer(frames_required=3)
    riders = [make_track(2, 0, (0, 0, 50, 100)), make_track(3, 0, (60, 0, 110, 100))]
    result = None
    for _ in range(3):
        result = analyzer.update(moto_track_id=1, rider_tracks=riders)
    assert result is not None
    assert result.outcome == OUTCOME_NO_VIOLATION


def test_triple_analyzer_potential_with_three_riders():
    analyzer = TripleRidingAnalyzer(frames_required=5)
    riders = [
        make_track(2, 0, (0, 0, 50, 100)),
        make_track(3, 0, (60, 0, 110, 100)),
        make_track(4, 0, (120, 0, 170, 100)),
    ]
    result = None
    for _ in range(5):
        result = analyzer.update(moto_track_id=1, rider_tracks=riders)
    assert result is not None
    assert result.outcome == OUTCOME_POTENTIAL
    assert result.estimated_rider_count == pytest.approx(3.0)


# ──────────────────────────────────────────────────────────────────────────────
# Event deduplication tests
# ──────────────────────────────────────────────────────────────────────────────

from backend.incidents.event_manager import is_duplicate, record_dedup_timestamp, _dedup_cache
from backend.config.constants import VIOLATION_TYPE_HELMET


def test_dedup_not_duplicate_first_time():
    _dedup_cache.clear()
    assert not is_duplicate(VIOLATION_TYPE_HELMET, track_id=99, dedup_seconds=30)


def test_dedup_is_duplicate_within_window():
    _dedup_cache.clear()
    record_dedup_timestamp(VIOLATION_TYPE_HELMET, track_id=99)
    assert is_duplicate(VIOLATION_TYPE_HELMET, track_id=99, dedup_seconds=30)


def test_dedup_different_track_not_duplicate():
    _dedup_cache.clear()
    record_dedup_timestamp(VIOLATION_TYPE_HELMET, track_id=1)
    assert not is_duplicate(VIOLATION_TYPE_HELMET, track_id=2, dedup_seconds=30)


def test_dedup_expired_outside_window():
    from datetime import datetime, timedelta, timezone
    _dedup_cache.clear()
    _dedup_cache[("helmet_violation", 99)] = datetime.now(timezone.utc) - timedelta(seconds=60)
    assert not is_duplicate(VIOLATION_TYPE_HELMET, track_id=99, dedup_seconds=30)

