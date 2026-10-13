# AI-Based Traffic Violation & Road Safety Analyzer

A research-oriented computer vision system that processes live camera footage to detect potential traffic violations — specifically **helmet violations** and **triple riding** on motorcycles. Results are shown on a real-time monitoring dashboard.

> **Research disclaimer**: All detections are labelled as *potential violations* requiring human review. No detection is automatically treated as a confirmed legal violation.

---

## Features

| Module | Description |
|---|---|
| **Live Camera Monitoring** | Webcam / video file / RTSP feed capture with FPS and latency metrics |
| **AI Object Detection** | YOLOv8 pretrained on COCO (person, motorcycle); custom model support |
| **Multi-Object Tracking** | IoU-based tracker maintaining consistent track IDs across frames |
| **Rider Association** | Spatial IoU + centroid heuristic to associate riders with motorcycles |
| **Helmet Violation** | Temporal evidence accumulation; requires custom model for best accuracy |
| **Triple Riding** | Frame-level rider count averaging with configurable evidence threshold |
| **Incident Management** | SQLite persistence, deduplication, evidence images, review workflow |
| **AI Evaluation** | YOLO-format dataset loader, Precision/Recall/F1/mAP@50/mAP@50–95 |
| **React Dashboard** | Overview, Live Monitor, Incidents, Evaluation, Settings, Docs |

---

## Requirements

- **Python** 3.10+ (tested on 3.12)
- **Node.js** 18+ and **npm** 9+
- **CUDA GPU** (optional but strongly recommended for real-time inference)
- A webcam, video file, or RTSP stream

---

## Installation

### 1. Clone / copy the project

```bash
cd traffic-safety-analyzer
```

### 2. Backend setup

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and edit configuration
cp .env.example .env
# Edit .env to set CAMERA_SOURCE, MODEL_NAME, etc.
```

### 3. Frontend setup

```bash
cd frontend
npm install
cd ..
```

---

## Running the System

### Start the backend (from `traffic-safety-analyzer/`)

```bash
source .venv/bin/activate
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is available at `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

### Start the frontend (from `traffic-safety-analyzer/`)

```bash
cd frontend
npm run dev
```

The dashboard is available at `http://localhost:5173`

---

## Camera Sources

| Value | Description |
|---|---|
| `0` | Default system webcam |
| `1`, `2`, … | Additional webcams |
| `/path/to/video.mp4` | Local video file (for testing) |
| `rtsp://user:pass@ip:port/stream` | IP camera / RTSP stream |

Set via `CAMERA_SOURCE` in `.env` or the Settings page.

---

## AI Model

The system uses **YOLOv8** via [Ultralytics](https://github.com/ultralytics/ultralytics).

Default: `yolov8n.pt` (nano, fast, lower accuracy)  
Better: `yolov8s.pt`, `yolov8m.pt`, `yolov8l.pt` (larger, slower, more accurate)

Weights are downloaded automatically on first run from the Ultralytics CDN.

### Helmet Detection Note

Standard COCO-trained models **do not** have helmet classes. For reliable helmet detection:

1. Source a labeled helmet dataset (e.g., [Safety Helmet Detection on Roboflow](https://universe.roboflow.com/))
2. Fine-tune a YOLOv8 model: `yolo train model=yolov8n.pt data=helmet.yaml epochs=100`
3. Update `HELMET_PRESENT_CLASSES` and `HELMET_ABSENT_CLASSES` in `backend/config/constants.py`
4. Set `MODEL_NAME=your_helmet_model.pt` in `.env`

---

## Evaluation

Add a YOLO-format dataset to `research/datasets/`:

```
research/datasets/my_dataset/
├── images/val/    ← images (.jpg / .png)
└── labels/val/    ← annotations (.txt, YOLO format: class cx cy w h)
```

Run evaluation from the **AI Evaluation** page, or via the API:

```bash
curl -X POST http://localhost:8000/api/v1/evaluation/run \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "./research/datasets/my_dataset", "dataset_name": "my_dataset"}'
```

Results are saved to `research/reports/` and the Evaluation page.

---

## Running Tests

```bash
source .venv/bin/activate
cd /path/to/traffic-safety-analyzer
pip install pytest pytest-asyncio anyio httpx
pytest tests/ -v
```

---

## Project Structure

```
traffic-safety-analyzer/
├── backend/
│   ├── main.py                # FastAPI app entry point
│   ├── pipeline.py            # Pipeline orchestrator
│   ├── api/                   # Route handlers
│   ├── camera/                # Capture module
│   ├── preprocessing/         # Frame validation & resize
│   ├── detection/             # YOLO detector wrapper
│   ├── tracking/              # IoU tracker
│   ├── association/           # Rider-to-motorcycle spatial association
│   ├── violations/            # Helmet & triple-riding analyzers
│   ├── incidents/             # Event deduplication & creation
│   ├── database/              # SQLAlchemy models & async DB
│   ├── storage/               # Evidence image persistence
│   ├── evaluation/            # Offline evaluation pipeline
│   └── config/                # Settings & constants
├── frontend/src/
│   ├── pages/                 # Overview, Monitoring, Incidents, Eval, Settings, Docs
│   ├── components/            # Layout, shared UI
│   ├── services/api.ts        # API client
│   └── hooks/useWebSocket.ts  # Live WebSocket hook
├── research/
│   ├── datasets/              # ← add labeled datasets here
│   ├── experiments/
│   └── reports/               # Auto-generated evaluation JSON
├── tests/
│   ├── test_violations.py     # Unit tests
│   └── test_api.py            # Integration tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## Known Limitations

- Helmet detection accuracy is **uncertain** without a custom-trained helmet model.
- Rider-to-motorcycle association may fail in very dense traffic scenes.
- CPU inference is slow (~1–3 FPS on typical hardware); GPU is required for real-time use.
- RTSP camera support depends on network stability and codec compatibility.
- The deduplication window (default 30 s) may suppress some valid rapid re-occurrences.

---

## License

This project is for academic/research use. Model weights follow the [Ultralytics license](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). COCO dataset follows the [COCO terms of use](https://cocodataset.org/#termsofuse).

