"""
Camera capture module.

Supports:
  - Local webcam (integer index)
  - Video file (path string)
  - RTSP / IP camera (URL string)

The CameraCapture class is designed to run in a background thread and
push frames into an asyncio.Queue for consumption by the pipeline.
"""
from __future__ import annotations

import asyncio
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Deque

import cv2
import numpy as np
from loguru import logger

from backend.config.settings import settings


@dataclass
class CameraStatus:
    running: bool = False
    connected: bool = False
    source: str = ""
    capture_fps: float = 0.0
    frame_count: int = 0
    last_error: str = ""
    started_at: datetime | None = None
    stopped_at: datetime | None = None


@dataclass
class CapturedFrame:
    frame: np.ndarray
    timestamp: datetime
    frame_index: int
    source: str


class CameraCapture:
    """Thread-safe camera capture with bounded frame queue."""

    MAX_QUEUE = 8  # keep only the freshest frames

    def __init__(
        self,
        source: int | str | None = None,
        fps_target: int | None = None,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        self._source = source if source is not None else settings.camera_source_parsed
        self._fps_target = fps_target or settings.camera_fps_target
        self._loop = loop or asyncio.get_event_loop()

        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._status = CameraStatus(source=str(self._source))

        # Bounded queue — older frames are dropped when full
        self._queue: asyncio.Queue[CapturedFrame] = asyncio.Queue(
            maxsize=self.MAX_QUEUE
        )

        # Rolling FPS window
        self._ts_window: Deque[float] = deque(maxlen=30)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def status(self) -> CameraStatus:
        return self._status

    @property
    def queue(self) -> asyncio.Queue[CapturedFrame]:
        return self._queue

    def start(self) -> bool:
        """Open camera and begin capture thread. Returns True on success."""
        if self._status.running:
            logger.warning("Camera already running.")
            return True

        self._stop_event.clear()
        self._cap = self._open_capture()
        if self._cap is None:
            return False

        self._status.running = True
        self._status.connected = True
        self._status.started_at = datetime.now(timezone.utc)
        self._status.frame_count = 0
        self._status.last_error = ""

        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="camera-capture"
        )
        self._thread.start()
        logger.info(f"Camera started: source={self._source!r}")
        return True

    def stop(self) -> None:
        """Signal the capture thread to stop and release resources."""
        self._stop_event.set()
        self._status.running = False
        self._status.connected = False
        self._status.stopped_at = datetime.now(timezone.utc)
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera stopped.")

    def _drain_queue(self) -> None:
        """Discard any unread frames from previous video source."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except (asyncio.QueueEmpty, ValueError):
                break

    def change_source(self, source: int | str) -> None:
        """Hot-swap the camera source."""
        was_running = self._status.running
        if was_running:
            self.stop()

        try:
            parsed = int(str(source).strip())
        except (ValueError, TypeError):
            parsed = source

        self._source = parsed
        self._status.source = str(parsed)
        self._drain_queue()

        if was_running:
            self.start()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _open_capture(self) -> cv2.VideoCapture | None:
        logger.info(f"Opening camera: {self._source!r}")
        cap = cv2.VideoCapture(self._source)  # type: ignore[arg-type]
        if not cap.isOpened():
            self._status.last_error = f"Cannot open source: {self._source!r}"
            logger.error(self._status.last_error)
            cap.release()
            return None
        # Try to set a reasonable buffer size
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        return cap

    def _capture_loop(self) -> None:
        interval = 1.0 / self._fps_target
        frame_index = 0

        while not self._stop_event.is_set():
            t0 = time.monotonic()

            if self._cap is None or not self._cap.isOpened():
                logger.warning("Camera disconnected; attempting reconnect …")
                self._status.connected = False
                time.sleep(2.0)
                self._cap = self._open_capture()
                if self._cap is None:
                    continue
                self._status.connected = True

            ret, frame = self._cap.read()
            if not ret or frame is None:
                # If reading from a video file, loop continuously like a live CCTV feed
                is_stream_url = isinstance(self._source, str) and (
                    self._source.startswith(("rtsp://", "http://", "https://"))
                )
                if not is_stream_url and not isinstance(self._source, int):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame — skipping.")
                    self._status.connected = False
                    time.sleep(0.5)
                    continue

            self._status.connected = True
            frame_index += 1
            self._status.frame_count = frame_index

            # FPS calculation
            now = time.monotonic()
            self._ts_window.append(now)
            if len(self._ts_window) >= 2:
                elapsed = self._ts_window[-1] - self._ts_window[0]
                self._status.capture_fps = (len(self._ts_window) - 1) / elapsed

            captured = CapturedFrame(
                frame=frame,
                timestamp=datetime.now(timezone.utc),
                frame_index=frame_index,
                source=str(self._source),
            )

            # Non-blocking put; drop oldest if full
            try:
                self._loop.call_soon_threadsafe(self._enqueue_frame, captured)
            except RuntimeError:
                break  # event loop closed

            elapsed = time.monotonic() - t0
            sleep = max(0.0, interval - elapsed)
            time.sleep(sleep)

    def _enqueue_frame(self, frame: CapturedFrame) -> None:
        if self._queue.full():
            try:
                self._queue.get_nowait()  # drop oldest
            except asyncio.QueueEmpty:
                pass
        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            pass


# Module-level singleton managed by the API layer
_camera_instance: CameraCapture | None = None


def get_camera() -> CameraCapture | None:
    return _camera_instance


def create_camera(source: int | str, loop: asyncio.AbstractEventLoop) -> CameraCapture:
    global _camera_instance
    if _camera_instance and _camera_instance.status.running:
        _camera_instance.stop()
    _camera_instance = CameraCapture(source=source, loop=loop)
    return _camera_instance

