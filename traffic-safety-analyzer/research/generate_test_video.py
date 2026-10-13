"""
Generate a synthetic test video that the pipeline can process
without a real camera.

Draws moving rectangles that resemble motorcycles and riders —
enough to exercise the full detection/tracking/violation pipeline
without needing real footage.

Usage:
    python research/generate_test_video.py
"""
from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path

OUTPUT_PATH = Path("research/datasets/test_video.mp4")
WIDTH, HEIGHT = 640, 480
FPS = 15
DURATION_S = 10


def make_frame(t: float) -> np.ndarray:
    """Draw a synthetic scene with a 'motorcycle + riders' rectangle."""
    frame = np.full((HEIGHT, WIDTH, 3), 30, dtype=np.uint8)

    # Moving motorcycle bounding box
    x = int((t / DURATION_S) * (WIDTH - 150))
    y = HEIGHT // 2 - 40

    # Motorcycle body (green rectangle)
    cv2.rectangle(frame, (x, y + 30), (x + 120, y + 80), (0, 200, 0), -1)

    # Three riders on top (simulating triple riding)
    for i in range(3):
        rx = x + 10 + i * 35
        ry = y - 30
        cv2.rectangle(frame, (rx, ry), (rx + 30, ry + 60), (180, 100, 60), -1)
        # No helmet (bare head — orange oval)
        cv2.ellipse(frame, (rx + 15, ry - 10), (12, 12), 0, 0, 360, (0, 100, 255), -1)

    # Labels
    cv2.putText(frame, f"t={t:.1f}s  [Synthetic test scene]",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    cv2.putText(frame, "3 riders / no helmets (simulated)",
                (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 200, 255), 1)

    return frame


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT_PATH), fourcc, FPS, (WIDTH, HEIGHT))

    total_frames = FPS * DURATION_S
    for i in range(total_frames):
        t = i / FPS
        frame = make_frame(t)
        writer.write(frame)

    writer.release()
    print(f"Test video written: {OUTPUT_PATH}  ({total_frames} frames, {DURATION_S}s)")


if __name__ == "__main__":
    main()

