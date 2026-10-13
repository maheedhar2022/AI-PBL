# Implementation Plan

## Phase 1: Foundation ✅
- [x] Project directory structure
- [x] Configuration: Pydantic settings, constants
- [x] SQLAlchemy async models (Incident, ExperimentResult, AppSettings)
- [x] FastAPI application skeleton with CORS and lifespan

## Phase 2: AI Baseline ✅
- [x] Camera capture module (webcam, file, RTSP)
- [x] Frame preprocessing (letterbox, validation)
- [x] YOLO detector wrapper (ultralytics, lazy load)
- [x] Detection dataclasses with annotation drawing

## Phase 3: Violation Logic ✅
- [x] IoU-based multi-object tracker
- [x] Rider-to-motorcycle spatial association
- [x] Helmet violation analyzer (temporal, conservative)
- [x] Triple riding analyzer (temporal, configurable threshold)
- [x] Incident event manager with deduplication
- [x] Evidence image storage

## Phase 4: Evaluation Pipeline ✅
- [x] YOLO-format dataset loader
- [x] Precision / Recall / F1 / mAP@50 / mAP@50-95 computation
- [x] Results saved to JSON in research/reports/
- [x] ExperimentResult persisted in SQLite

## Phase 5: API & Dashboard ✅
- [x] FastAPI routes: camera, monitoring, incidents, dashboard, evaluation, settings
- [x] WebSocket route for live event broadcast
- [x] React + Vite + Tailwind frontend
- [x] Pages: Overview, Live Monitoring, Incident Explorer, Evaluation, Settings, Docs
- [x] API service module (axios)
- [x] WebSocket hook with auto-reconnect

## Phase 6: Integration & Validation ✅
- [x] Unit tests: tracker, association, analyzers, deduplication
- [x] Integration tests: API health, incidents, settings
- [x] .gitignore, README, architecture docs

## Remaining / Known Gaps

| Item | Status |
|---|---|
| Helmet detection (custom model) | Requires labeled dataset + fine-tuning |
| ByteTrack integration | Currently using IoU tracker; upgrade when stable |
| GPU auto-detection | Ultralytics handles this automatically |
| Production deployment | Out of scope for research build |
| MongoDB / S3 migration | Config abstraction in place; driver swap only |

## Model & Dataset Decisions

- **Model**: YOLOv8n (COCO pretrained) — lightweight, CPU-runnable, no license restrictions
- **Helmet model**: Not included; system is conservative (marks uncertain) without one
- **Dataset**: User must supply YOLO-format labeled data; annotation schema provided in docs

