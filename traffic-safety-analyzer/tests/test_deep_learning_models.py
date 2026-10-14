"""
Tests for Deep Learning models:
- DeepHelmetCNN and DeepHelmetClassifier
- DeepReIDNet and DeepReIDExtractor
- ViolationMLP and DeepViolationRiskClassifier
- DeepSortTracker
- Deep Learning API endpoints (/api/v1/models/info and /api/v1/models/benchmark)
"""
from __future__ import annotations

import pytest
import numpy as np
import torch
from httpx import AsyncClient, ASGITransport

from backend.main import app
from backend.models.helmet_classifier import DeepHelmetCNN, DeepHelmetClassifier
from backend.models.reid_extractor import DeepReIDNet, DeepReIDExtractor
from backend.models.violation_classifier import ViolationMLP, DeepViolationRiskClassifier
from backend.models.registry import get_model_registry
from backend.tracking.deep_sort_tracker import DeepSortTracker, KalmanBoxTracker


# ──────────────────────────────────────────────────────────────────────────────
# 1. DeepHelmetCNN Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_deep_helmet_cnn_architecture():
    model = DeepHelmetCNN(num_classes=3)
    model.eval()
    dummy = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, 3)


def test_deep_helmet_classifier_inference():
    clf = DeepHelmetClassifier()
    assert clf.is_loaded is True
    assert clf.parameter_count > 100_000

    # Test synthetic crop
    synthetic_head = np.zeros((40, 40, 3), dtype=np.uint8)
    label, conf, probs = clf.classify_crop(synthetic_head)
    assert label in ["helmet", "no_helmet", "uncertain"]
    assert 0.0 <= conf <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-4

    # Test batch classification
    crops = [np.zeros((30, 30, 3), dtype=np.uint8), np.ones((50, 50, 3), dtype=np.uint8) * 128]
    batch_results = clf.classify_batch(crops)
    assert len(batch_results) == 2


# ──────────────────────────────────────────────────────────────────────────────
# 2. DeepReIDNet Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_deep_reid_net_architecture():
    model = DeepReIDNet(embedding_dim=128)
    model.eval()
    dummy = torch.randn(3, 3, 128, 64)
    with torch.no_grad():
        embeddings = model(dummy)
    assert embeddings.shape == (3, 128)
    # Check L2 normalization
    norms = torch.norm(embeddings, p=2, dim=1).numpy()
    for n in norms:
        assert abs(n - 1.0) < 1e-4


def test_deep_reid_extractor_similarity():
    extractor = DeepReIDExtractor()
    assert extractor.parameter_count > 100_000

    crop1 = np.zeros((80, 40, 3), dtype=np.uint8)
    crop2 = np.ones((80, 40, 3), dtype=np.uint8) * 200

    emb1 = extractor.extract_embedding(crop1)
    emb2 = extractor.extract_embedding(crop2)

    assert emb1.shape == (128,)
    assert emb2.shape == (128,)
    assert abs(np.linalg.norm(emb1) - 1.0) < 1e-3
    assert abs(np.linalg.norm(emb2) - 1.0) < 1e-3

    sim = extractor.cosine_similarity(emb1, emb2)
    assert -1.0 <= sim <= 1.0

    # Distance matrix
    dist_mat = extractor.cosine_distance_matrix(np.stack([emb1]), np.stack([emb2]))
    assert dist_mat.shape == (1, 1)
    assert 0.0 <= dist_mat[0, 0] <= 2.0


# ──────────────────────────────────────────────────────────────────────────────
# 3. ViolationMLP & DeepViolationRiskClassifier Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_violation_mlp_architecture():
    model = ViolationMLP(in_features=16, num_classes=3)
    model.eval()
    dummy = torch.randn(4, 16)
    with torch.no_grad():
        logits = model(dummy)
    assert logits.shape == (4, 3)


def test_violation_risk_classifier_features_and_eval():
    clf = DeepViolationRiskClassifier()
    assert clf.parameter_count > 1000

    moto_bbox = (100, 100, 250, 300)
    riders = [(110, 80, 170, 200), (160, 80, 220, 200), (130, 80, 190, 200)]  # 3 riders
    feats = clf.extract_features(
        moto_bbox=moto_bbox,
        rider_bboxes=riders,
        speed_px_s=45.0,
        history=[(150, 150), (155, 152), (160, 155)],
    )
    assert feats.shape == (16,)
    assert feats[0] == 3.0  # 3 riders

    outcome, conf, probs = clf.evaluate_risk(feats)
    assert outcome in ["safe", "triple_riding", "dangerous_maneuver"]
    assert 0.0 <= conf <= 1.0
    assert abs(sum(probs.values()) - 1.0) < 1e-4
    # With 3 riders, triple_riding should be strongly favored
    assert probs["triple_riding"] > 0.5


# ──────────────────────────────────────────────────────────────────────────────
# 4. DeepSortTracker & Kalman Filter Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_kalman_box_tracker():
    initial_bbox = (100, 100, 200, 200)
    tracker = KalmanBoxTracker(initial_bbox)
    pred_box = tracker.predict()
    assert len(pred_box) == 4

    tracker.update((105, 102, 205, 202))
    state_box = tracker.get_state()
    assert len(state_box) == 4


def test_deep_sort_tracker_lifecycle():
    tracker = DeepSortTracker()
    dets = [
        {"class_id": 3, "class_name": "motorcycle", "confidence": 0.85, "bbox": (100, 100, 200, 200)},
        {"class_id": 0, "class_name": "person", "confidence": 0.90, "bbox": (120, 80, 180, 180)},
    ]
    tracks = tracker.update(dets)
    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id

    # Second frame with slight shift
    dets2 = [
        {"class_id": 3, "class_name": "motorcycle", "confidence": 0.86, "bbox": (104, 102, 204, 202)},
        {"class_id": 0, "class_name": "person", "confidence": 0.91, "bbox": (124, 82, 184, 182)},
    ]
    tracks2 = tracker.update(dets2)
    assert len(tracks2) == 2
    # Track IDs should remain consistent across continuous frames
    assert tracks2[0].track_id == tracks[0].track_id
    assert tracks2[1].track_id == tracks[1].track_id


# ──────────────────────────────────────────────────────────────────────────────
# 5. Model Registry & API Endpoints Tests
# ──────────────────────────────────────────────────────────────────────────────

def test_model_registry_specs():
    registry = get_model_registry()
    specs = registry.get_all_model_specs()
    assert "device" in specs
    assert "models" in specs
    assert "object_detector" in specs["models"]
    assert "helmet_classifier" in specs["models"]
    assert "reid_extractor" in specs["models"]
    assert "violation_classifier" in specs["models"]
    assert specs["total_deep_parameters"] > 3_000_000


@pytest.mark.asyncio
async def test_models_info_api_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/models/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert data["models"]["helmet_classifier"]["parameters"] > 0
        assert data["models"]["reid_extractor"]["parameters"] > 0
        assert data["models"]["violation_classifier"]["parameters"] > 0


@pytest.mark.asyncio
async def test_models_benchmark_api_endpoint():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/models/benchmark", json={"iterations": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "helmet_cnn_inference" in data["results"]
        assert "reid_embedding_extraction" in data["results"]
        assert "violation_mlp_risk" in data["results"]
        assert data["results"]["helmet_cnn_inference"]["mean_ms"] > 0.0

