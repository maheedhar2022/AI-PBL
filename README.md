# AI-Based Traffic Violation & Road Safety Analyzer

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688.svg)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18.3+-61DAFB.svg)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-5.4+-646CFF.svg)](https://vitejs.dev)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

An end-to-end AI/ML research and computer vision monitoring system that detects vehicles, analyzes road safety, and identifies traffic violations (helmet-less riding and triple riding) from live webcam feeds, IP/RTSP streams, and open-source CCTV road footage.

---

## 🌟 Key Features

- **Live Camera & CCTV Monitoring**: Real-time video ingestion from local webcams, RTSP streams, and looping open-source roadway CCTV feeds.
- **AI Object Detection**: Pretrained YOLOv8 architecture detecting cars, motorcycles, buses, trucks, bicycles, and pedestrians.
- **Multi-Object Tracking (MOT)**: High-performance temporal IoU tracking assigning persistent IDs across frames.
- **Road Safety Rules Engine**:
  - Spatial IoU & centroid mapping of riders to two-wheelers.
  - Multi-frame temporal evidence accumulation (minimum consecutive frames) preventing false alarms.
  - Helmet detection & conservative violation classification (`potential_violation` vs `uncertain`).
  - Triple-riding detection flagging motorcycles carrying $\ge 3$ riders.
- **Deduplication Engine**: Sliding-window cooldown (`DEDUP_SECONDS`) preventing repeated alerts for the same vehicle track.
- **Evidence Storage & Audit Trail**: SQLite database with full incident metadata, confidence scores, and crop snapshots saved to filesystem.
- **6-Page Monitoring Dashboard**:
  - **Overview**: Real-time traffic KPIs, violation distributions, and daily activity charts.
  - **Live Monitoring**: Real-time video canvas with top CCTV HUD, vehicle breakdown counters, and live alert ticker.
  - **Incident Explorer**: Paginated incident table with review status management (`confirmed`, `dismissed`, `pending`).
  - **Model Evaluation**: Offline mAP@50 and mAP@50–95 benchmarking against labeled validation datasets.
  - **Documentation**: Comprehensive methodology and ethical AI deployment guidelines.
  - **Settings**: System thresholds, confidence filters, and camera source configuration.

---

## 📂 Project Structure

```
AI-PBL/
└── traffic-safety-analyzer/
    ├── backend/                 # FastAPI REST API, AI Pipeline & Storage
    │   ├── api/                 # 7 modular APIRouters (camera, monitoring, incidents, ws, ...)
    │   ├── association/         # Rider-to-motorcycle spatial mapping
    │   ├── camera/              # Video capture with continuous 24/7 looping
    │   ├── config/              # Application settings and COCO constants
    │   ├── database/            # SQLAlchemy async SQLite models & migrations
    │   ├── detection/           # YOLOv8 inference wrapper
    │   ├── evaluation/          # mAP@50 / mAP@50-95 offline evaluator
    │   ├── incidents/           # Event manager & deduplication sliding window
    │   ├── tracking/            # SimpleIOUTracker persistent MOT
    │   └── violations/          # Multi-frame helmet & triple-riding analyzers
    ├── frontend/                # React 18 + Vite + Tailwind CSS + Recharts
    │   └── src/
    │       ├── components/      # Layout, Sidebar, Navbar, StatusBadge
    │       ├── pages/           # Overview, LiveMonitoring, IncidentExplorer, etc.
    │       └── services/        # Axios API client & WebSocket hook
    ├── research/                # Datasets, synthetic scenes, and evaluation reports
    ├── docs/                    # Architecture diagrams & ethical deployment guide
    ├── tests/                   # 24 automated unit & integration tests
    ├── start.sh                 # One-click startup script (Backend + Frontend)
    └── stop.sh                  # One-click shutdown script
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+ and npm

### 2. Launching Services
Run the automated startup script:
```bash
cd traffic-safety-analyzer
./start.sh
```

- **Frontend Dashboard**: [http://localhost:5173](http://localhost:5173)
- **Backend API & Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

### 3. Stopping Services
```bash
cd traffic-safety-analyzer
./stop.sh
```

### 4. Running Tests
```bash
cd traffic-safety-analyzer
.venv/bin/pytest tests/ -v
```
