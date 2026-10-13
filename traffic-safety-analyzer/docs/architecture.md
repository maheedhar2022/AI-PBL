# Architecture

## System Overview

The AI Traffic Safety Analyzer is a modular computer-vision pipeline that processes
camera footage and identifies potential traffic violations. It is designed as a
research-oriented system where model performance is measurable and reproducible.

## Data Flow

```
Camera / Video File
  ↓ OpenCV Capture Thread
  ↓ Bounded Frame Queue (max 8 frames; oldest dropped under load)
  ↓ ViolationPipeline (asyncio task)
      ↓ Frame Validation
      ↓ YOLO Inference (run_in_executor — non-blocking)
      ↓ Vehicle Tracker (IoU-based, per-class)
      ↓ Person Tracker (IoU-based)
      ↓ Rider → Motorcycle Association (spatial IoU + centroid)
      ↓ Helmet Analyzer (temporal accumulation, N frames)
      ↓ Triple Riding Analyzer (temporal accumulation, N frames)
      ↓ Incident Deduplication (per track_id + violation_type window)
      ↓ Evidence Image Save → filesystem
      ↓ Incident Record → SQLite (async)
      ↓ WebSocket Broadcast → Frontend
```

## Module Boundaries

| Module | Responsibility |
|---|---|
| `camera/` | OpenCV capture, reconnect, bounded queue |
| `preprocessing/` | Frame validation, letterbox resize |
| `detection/` | YOLO model load & inference |
| `tracking/` | IoU tracker, track expiry |
| `association/` | Rider-to-motorcycle spatial mapping |
| `violations/` | Rule logic: helmet & triple-riding temporal analyzers |
| `incidents/` | Deduplication, DB writes, evidence paths |
| `storage/` | Evidence JPEG save |
| `evaluation/` | Offline dataset loading, metric computation |
| `database/` | SQLAlchemy async models |
| `api/` | FastAPI routes and WebSocket |
| `config/` | Pydantic settings + shared constants |

## Key Design Decisions

1. **Bounded frame queue**: Prevents memory growth under slow inference. Fresh frames
   are preferred over older ones.

2. **Separate vehicle and person trackers**: Avoids class collision in the IoU matcher
   and allows independent track expiry tuning.

3. **Temporal evidence accumulation**: Violations require N consecutive (or near-
   consecutive) frames of evidence before an incident is raised. This reduces false
   positives from single-frame occlusions or misdetections.

4. **Deduplication window**: Once an incident is created for a (violation_type, track_id)
   pair, a configurable cooldown window suppresses duplicate records.

5. **Non-blocking inference**: `run_in_executor` offloads YOLO inference to a thread
   pool, keeping the asyncio event loop free for WebSocket messages and API requests.

6. **Conservative helmet analysis**: Without a custom model, helmet status is always
   marked `uncertain`. No false violation is created from heuristics alone.

