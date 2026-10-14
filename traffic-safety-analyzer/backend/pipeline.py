"""
Main processing pipeline orchestrator.

Pulls frames from the camera queue, runs detection → tracking →
association → violation analysis → incident creation.

Designed to run as an asyncio background task.
"""
from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
from loguru import logger

from backend.camera.capture import CapturedFrame
from backend.config.constants import (
    COCO_MOTORCYCLE,
    VIOLATION_TYPE_HELMET,
    VIOLATION_TYPE_TRIPLE,
    OUTCOME_POTENTIAL,
    OUTCOME_UNCERTAIN,
)
from backend.config.settings import settings
from backend.database.db import AsyncSessionLocal
from backend.detection.detector import YOLODetector, DetectionResult
from backend.incidents.event_manager import create_incident, prune_dedup_cache
from backend.preprocessing.frame_processor import validate_frame
from backend.tracking.tracker import SimpleIOUTracker, Track
from backend.association.rider_assoc import associate_riders_to_motorcycles
from backend.violations.helmet_analyzer import HelmetViolationAnalyzer
from backend.violations.triple_analyzer import TripleRidingAnalyzer


class PipelineMetrics:
    def __init__(self) -> None:
        self.frames_processed: int = 0
        self.inference_fps: float = 0.0
        self.last_inference_ms: float = 0.0
        self.incidents_created: int = 0
        self.active_vehicles: int = 0
        self.vehicle_counts: dict[str, int] = {
            "cars": 0, "motorcycles": 0, "buses": 0, "trucks": 0, "bicycles": 0, "pedestrians": 0
        }
        self.traffic_density: str = "Light"
        self._ts_window: list[float] = []

    def record_frame(
        self,
        inference_ms: float,
        vehicle_counts: Optional[dict[str, int]] = None,
    ) -> None:
        self.frames_processed += 1
        self.last_inference_ms = inference_ms
        if vehicle_counts:
            self.vehicle_counts = vehicle_counts
            self.active_vehicles = (
                vehicle_counts.get("cars", 0)
                + vehicle_counts.get("motorcycles", 0)
                + vehicle_counts.get("buses", 0)
                + vehicle_counts.get("trucks", 0)
                + vehicle_counts.get("bicycles", 0)
            )
            if self.active_vehicles >= 8:
                self.traffic_density = "Heavy"
            elif self.active_vehicles >= 4:
                self.traffic_density = "Moderate"
            else:
                self.traffic_density = "Light"

        now = time.monotonic()
        self._ts_window.append(now)
        # Keep 30-frame window
        if len(self._ts_window) > 30:
            self._ts_window.pop(0)
        if len(self._ts_window) >= 2:
            elapsed = self._ts_window[-1] - self._ts_window[0]
            self.inference_fps = (len(self._ts_window) - 1) / max(elapsed, 1e-6)

    def to_dict(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "inference_fps": round(self.inference_fps, 2),
            "last_inference_ms": round(self.last_inference_ms, 2),
            "incidents_created": self.incidents_created,
            "active_vehicles": self.active_vehicles,
            "vehicle_counts": self.vehicle_counts,
            "traffic_density": self.traffic_density,
        }


class VideoRecorder:
    """Records live annotated CCTV stream frames to an MP4 video file."""

    def __init__(self, output_dir: Path, fps: int = 10) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self._writer: Optional[cv2.VideoWriter] = None
        self._filename: Optional[str] = None
        self._filepath: Optional[Path] = None
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._is_recording: bool = False

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def start(self, frame_shape: Tuple[int, int]) -> str:
        if self._is_recording:
            return self._filename or ""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h, w = frame_shape[:2]
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._filename = f"cctv_recording_{ts}.mp4"
        self._filepath = self.output_dir / self._filename
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._filepath), fourcc, float(self.fps), (w, h))
        self._frame_count = 0
        self._start_time = time.monotonic()
        self._is_recording = True
        logger.info(f"Started video recording: {self._filepath}")
        return self._filename

    def write_frame(self, frame: np.ndarray) -> None:
        if self._is_recording and self._writer is not None:
            self._writer.write(frame)
            self._frame_count += 1

    def stop(self) -> Optional[dict]:
        if not self._is_recording:
            return None
        self._is_recording = False
        duration = round(time.monotonic() - self._start_time, 1)
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        size_bytes = self._filepath.stat().st_size if (self._filepath and self._filepath.exists()) else 0
        info = {
            "filename": self._filename,
            "path": str(self._filepath),
            "frame_count": self._frame_count,
            "duration_sec": duration,
            "size_kb": round(size_bytes / 1024, 1),
        }
        logger.info(f"Stopped video recording: {info}")
        return info


class ViolationPipeline:
    """Full AI pipeline for a single camera stream."""

    def __init__(
        self,
        detector: YOLODetector,
        frame_queue: asyncio.Queue[CapturedFrame],
        broadcast_fn: Optional[Callable[[dict], Any]] = None,
    ) -> None:
        self._detector = detector
        self._frame_queue = frame_queue
        self._broadcast = broadcast_fn  # async callable for WS notifications

        # Sub-modules
        self._vehicle_tracker = SimpleIOUTracker(iou_threshold=0.35, max_missed=8)
        self._person_tracker  = SimpleIOUTracker(iou_threshold=0.35, max_missed=8)
        self._helmet_analyzer = HelmetViolationAnalyzer()
        self._triple_analyzer = TripleRidingAnalyzer()
        self._recorder = VideoRecorder(
            output_dir=settings.evidence_dir / "recordings",
            fps=settings.camera_fps_target,
        )

        self.metrics = PipelineMetrics()
        self._running = False
        self._task: Optional[asyncio.Task] = None

        # Latest annotated frame (JPEG bytes for WS streaming)
        self._latest_frame_b64: Optional[str] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Control                                                              #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.ensure_future(self._run())
        logger.info("ViolationPipeline started.")

    async def stop(self) -> None:
        self._running = False
        if self._recorder.is_recording:
            self._recorder.stop()
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._vehicle_tracker.reset()
        self._person_tracker.reset()
        self._helmet_analyzer.reset()
        self._triple_analyzer.reset()
        logger.info("ViolationPipeline stopped.")

    def start_recording(self, frame_shape: Tuple[int, int] = (720, 1280)) -> str:
        return self._recorder.start(frame_shape)

    def stop_recording(self) -> Optional[dict]:
        return self._recorder.stop()

    def save_snapshot(self) -> Optional[dict]:
        if self._latest_annotated_frame is None:
            return None
        snapshot_dir = settings.evidence_dir / "snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"snapshot_{ts}.jpg"
        filepath = snapshot_dir / filename
        cv2.imwrite(str(filepath), self._latest_annotated_frame)
        stat = filepath.stat()
        return {
            "filename": filename,
            "path": str(filepath),
            "size_kb": round(stat.st_size / 1024, 1),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

    @property
    def is_recording(self) -> bool:
        return self._recorder.is_recording

    @property
    def latest_frame_b64(self) -> Optional[str]:
        return self._latest_frame_b64

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #

    async def _run(self) -> None:
        while self._running:
            try:
                captured: CapturedFrame = await asyncio.wait_for(
                    self._frame_queue.get(), timeout=2.0
                )
            except asyncio.TimeoutError:
                continue

            if not validate_frame(captured.frame):
                continue

            await self._process_frame(captured)

    async def _process_frame(self, captured: CapturedFrame) -> None:
        frame = captured.frame

        # 1. Detection
        det_result: DetectionResult = await asyncio.get_event_loop().run_in_executor(
            None, self._detector.detect, frame
        )

        annotated = det_result.annotated_frame if det_result.annotated_frame is not None else frame

        # 2. Tracking — separate trackers for vehicles and persons
        vehicle_dets = [
            {"class_id": d.class_id, "class_name": d.class_name,
             "confidence": d.confidence, "bbox": d.bbox}
            for d in det_result.vehicles
        ]
        person_dets = [
            {"class_id": d.class_id, "class_name": d.class_name,
             "confidence": d.confidence, "bbox": d.bbox}
            for d in det_result.persons
        ]

        vehicle_tracks: List[Track] = self._vehicle_tracker.update(vehicle_dets)
        person_tracks:  List[Track] = self._person_tracker.update(person_dets)

        # Update metrics with real-time vehicle counts
        counts = {
            "cars": sum(1 for t in vehicle_tracks if t.class_name == "car"),
            "motorcycles": sum(1 for t in vehicle_tracks if t.class_name == "motorcycle"),
            "buses": sum(1 for t in vehicle_tracks if t.class_name == "bus"),
            "trucks": sum(1 for t in vehicle_tracks if t.class_name == "truck"),
            "bicycles": sum(1 for t in vehicle_tracks if t.class_name == "bicycle"),
            "pedestrians": len(person_tracks),
        }
        self.metrics.record_frame(det_result.inference_time_ms, vehicle_counts=counts)

        # Filter: only motorcycles for helmet/triple riding
        moto_tracks = [t for t in vehicle_tracks if t.class_id == COCO_MOTORCYCLE]

        # 3. Association
        association = associate_riders_to_motorcycles(moto_tracks, person_tracks)

        # 4. Draw track IDs and HUD on annotated frame
        annotated = self._draw_track_ids(annotated, vehicle_tracks + person_tracks)

        # 5. Violation analysis + incident creation
        await self._analyze_violations(
            moto_tracks, person_tracks, association,
            annotated, captured
        )

        # 6. Record frame if active
        if self._recorder.is_recording:
            self._recorder.write_frame(annotated)

        # 7. Encode frame for streaming
        self._encode_frame(annotated)

        # 7. Periodic dedup cache cleanup
        if self.metrics.frames_processed % 100 == 0:
            prune_dedup_cache()

        # 8. Broadcast live update
        if self._broadcast:
            await self._broadcast({
                "type": "frame_update",
                "metrics": self.metrics.to_dict(),
                "timestamp": captured.timestamp.isoformat(),
            })

    # ------------------------------------------------------------------ #
    # Violation logic                                                      #
    # ------------------------------------------------------------------ #

    async def _analyze_violations(
        self,
        moto_tracks: List[Track],
        person_tracks: List[Track],
        association: Dict[int, List[int]],
        annotated_frame: np.ndarray,
        captured: CapturedFrame,
    ) -> None:
        for moto in moto_tracks:
            rider_ids = association.get(moto.track_id, [])
            rider_tracks = [p for p in person_tracks if p.track_id in rider_ids]

            # Class IDs present in this frame for this motorcycle's region
            frame_class_ids = [r.class_id for r in rider_tracks]

            # ── Helmet analysis ──────────────────────────────────────────
            helmet_result = self._helmet_analyzer.update(
                moto_track_id=moto.track_id,
                rider_tracks=rider_tracks,
                frame_detections_class_ids=frame_class_ids,
                frame=captured.frame,
            )
            if helmet_result and helmet_result.outcome == OUTCOME_POTENTIAL:
                async with AsyncSessionLocal() as db:
                    inc = await create_incident(
                        db=db,
                        violation_type=VIOLATION_TYPE_HELMET,
                        track_id=moto.track_id,
                        confidence=helmet_result.confidence,
                        camera_source=captured.source,
                        frame=annotated_frame,
                        metadata={
                            "outcome": helmet_result.outcome,
                            "rider_count": len(rider_ids),
                            "frames_evaluated": helmet_result.frames_evaluated,
                            "details": helmet_result.details,
                        },
                        outcome=helmet_result.outcome,
                        rider_count=len(rider_ids),
                    )
                    if inc:
                        self.metrics.incidents_created += 1
                        if self._broadcast:
                            await self._broadcast({
                                "type": "new_incident",
                                "incident_id": inc.id,
                                "violation_type": inc.violation_type,
                                "confidence": inc.confidence,
                            })

            # ── Triple riding analysis ───────────────────────────────────
            triple_result = self._triple_analyzer.update(
                moto_track_id=moto.track_id,
                rider_tracks=rider_tracks,
            )
            if triple_result and triple_result.outcome == OUTCOME_POTENTIAL:
                async with AsyncSessionLocal() as db:
                    inc = await create_incident(
                        db=db,
                        violation_type=VIOLATION_TYPE_TRIPLE,
                        track_id=moto.track_id,
                        confidence=triple_result.confidence,
                        camera_source=captured.source,
                        frame=annotated_frame,
                        metadata={
                            "outcome": triple_result.outcome,
                            "estimated_rider_count": triple_result.estimated_rider_count,
                            "frames_evaluated": triple_result.frames_evaluated,
                            "details": triple_result.details,
                        },
                        outcome=triple_result.outcome,
                        rider_count=int(round(triple_result.estimated_rider_count)),
                    )
                    if inc:
                        self.metrics.incidents_created += 1
                        if self._broadcast:
                            await self._broadcast({
                                "type": "new_incident",
                                "incident_id": inc.id,
                                "violation_type": inc.violation_type,
                                "confidence": inc.confidence,
                            })

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _draw_track_ids(
        self, frame: np.ndarray, tracks: List[Track]
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        # Draw HUD bar at the top of the video
        hud_h = 32
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        counts = self.metrics.vehicle_counts
        hud_text = (
            f"LIVE CCTV | Vehicles: {self.metrics.active_vehicles} "
            f"(Cars:{counts.get('cars',0)} Bikes:{counts.get('motorcycles',0)} "
            f"Buses:{counts.get('buses',0)} Trucks:{counts.get('trucks',0)}) "
            f"| Flow: {self.metrics.traffic_density}"
        )
        density_color = (
            (0, 255, 120) if self.metrics.traffic_density == "Light"
            else (0, 200, 255) if self.metrics.traffic_density == "Moderate"
            else (0, 100, 255)
        )
        cv2.putText(frame, hud_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (240, 240, 240), 1, cv2.LINE_AA)
        cv2.circle(frame, (w - 20, 16), 5, density_color, -1)

        if self._recorder.is_recording:
            cv2.putText(frame, "REC", (w - 75, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, (w - 40, 16), 5, (0, 0, 255), -1)

        # Draw motion trajectory trails and track badges
        for t in tracks:
            if len(t.history) >= 2:
                pts = np.array(t.history, dtype=np.int32).reshape((-1, 1, 2))
                trail_color = (0, 240, 255) if t.class_name in ("car", "motorcycle") else (255, 180, 50)
                cv2.polylines(frame, [pts], isClosed=False, color=trail_color, thickness=2, lineType=cv2.LINE_AA)
                cv2.circle(frame, t.history[-1], 3, (0, 255, 255), -1)

            x1, y1, x2, y2 = t.bbox
            speed_str = f" {int(t.speed_px_s)}px/s" if t.speed_px_s > 8 else ""
            label = f"#{t.track_id}{speed_str}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            badge_y1 = min(h - 18, y2 + 2)
            badge_y2 = min(h - 2, badge_y1 + th + 6)
            cv2.rectangle(frame, (x1, badge_y1), (x1 + tw + 6, badge_y2), (20, 20, 20), -1)
            cv2.putText(
                frame, label, (x1 + 3, badge_y1 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 230, 0), 1, cv2.LINE_AA
            )
        return frame

    def _encode_frame(self, frame: np.ndarray) -> None:
        try:
            self._latest_annotated_frame = frame.copy()
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            self._latest_frame_b64 = base64.b64encode(buf).decode("utf-8")
        except Exception as exc:
            logger.warning(f"Frame encode error: {exc}")


# Module-level singleton
_pipeline: Optional[ViolationPipeline] = None


def get_pipeline() -> Optional[ViolationPipeline]:
    return _pipeline


def create_pipeline(
    detector: YOLODetector,
    frame_queue: asyncio.Queue,
    broadcast_fn: Optional[Callable] = None,
) -> ViolationPipeline:
    global _pipeline
    _pipeline = ViolationPipeline(
        detector=detector,
        frame_queue=frame_queue,
        broadcast_fn=broadcast_fn,
    )
    return _pipeline

