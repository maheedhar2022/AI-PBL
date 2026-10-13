"""
Frame preprocessing: resize, normalize, and validate frames.
"""
from __future__ import annotations

import cv2
import numpy as np

from backend.config.settings import settings


def validate_frame(frame: np.ndarray | None) -> bool:
    """Return True if frame is a valid, non-empty image array."""
    if frame is None:
        return False
    if not isinstance(frame, np.ndarray):
        return False
    if frame.ndim not in (2, 3):
        return False
    if frame.size == 0:
        return False
    return True


def resize_for_inference(
    frame: np.ndarray,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    """Resize frame to inference dimensions while preserving aspect ratio via letterbox."""
    w = width or settings.inference_width
    h = height or settings.inference_height
    return letterbox(frame, (h, w))


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """Resize and pad image to new_shape (h, w) maintaining aspect ratio."""
    h0, w0 = img.shape[:2]
    target_h, target_w = new_shape
    scale = min(target_w / w0, target_h / h0)
    new_w, new_h = int(round(w0 * scale)), int(round(h0 * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Padding
    pad_w = (target_w - new_w) / 2
    pad_h = (target_h - new_h) / 2
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=color
    )
    return img_padded


def bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

