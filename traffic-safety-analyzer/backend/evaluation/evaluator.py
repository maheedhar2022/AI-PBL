"""
Offline evaluation pipeline.

Loads labeled image/video datasets and evaluates detection performance
using standard metrics: precision, recall, F1, mAP@50, mAP@50-95.

Evaluation results are saved to JSON and CSV and never fabricated.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

from backend.config.settings import settings


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class GroundTruthBox:
    class_id: int
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalised 0-1)


@dataclass
class PredBox:
    class_id: int
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 (normalised 0-1)


@dataclass
class ImageAnnotation:
    image_path: str
    ground_truths: List[GroundTruthBox]


@dataclass
class EvaluationConfig:
    model_name: str
    dataset_name: str
    dataset_path: str
    iou_threshold: float = 0.50
    confidence_threshold: float = 0.25
    class_names: Dict[int, str] = field(default_factory=dict)
    experiment_name: str = ""
    notes: str = ""


@dataclass
class EvaluationMetrics:
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    map50_95: float = 0.0
    avg_inference_ms: float = 0.0
    fps: float = 0.0
    total_images: int = 0
    total_gt_boxes: int = 0
    total_pred_boxes: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    per_class: Dict[str, Any] = field(default_factory=dict)
    evaluated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "completed"
    error_message: str = ""


# ------------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------------

def load_yolo_dataset(
    dataset_dir: Path,
    split: str = "val",
) -> List[ImageAnnotation]:
    """
    Load a YOLO-format dataset (images/ + labels/ directories).

    Expected layout:
        dataset_dir/
            images/val/  (or images/test/ etc.)
            labels/val/
    """
    images_dir = dataset_dir / "images" / split
    labels_dir = dataset_dir / "labels" / split

    if not images_dir.exists():
        logger.warning(f"Dataset images directory not found: {images_dir}")
        return []
    if not labels_dir.exists():
        logger.warning(f"Dataset labels directory not found: {labels_dir}")
        return []

    annotations: List[ImageAnnotation] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        gts: List[GroundTruthBox] = []

        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                # YOLO format: cx cy w h (normalised)
                cx, cy, w, h = map(float, parts[1:5])
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                gts.append(GroundTruthBox(class_id=cls_id, bbox=(x1, y1, x2, y2)))

        annotations.append(ImageAnnotation(image_path=str(img_path), ground_truths=gts))

    logger.info(f"Loaded {len(annotations)} annotated images from {dataset_dir}")
    return annotations


# ------------------------------------------------------------------
# Core evaluation
# ------------------------------------------------------------------

def _box_iou_normalised(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute AP using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        precs = [p for r, p in zip(recalls, precisions) if r >= t]
        ap += max(precs, default=0.0) / 11
    return ap


def evaluate_detections(
    annotations: List[ImageAnnotation],
    predictions: List[List[PredBox]],
    iou_threshold: float = 0.50,
    class_names: Optional[Dict[int, str]] = None,
) -> EvaluationMetrics:
    """
    Compute precision, recall, F1, and mAP from paired annotations and predictions.
    """
    if not annotations:
        m = EvaluationMetrics(status="no_data")
        m.error_message = "No annotated images provided."
        return m

    class_names = class_names or {}
    all_class_ids = set()
    for ann in annotations:
        for gt in ann.ground_truths:
            all_class_ids.add(gt.class_id)

    total_gt = sum(len(a.ground_truths) for a in annotations)
    total_pred = sum(len(p) for p in predictions)

    def _eval_at_iou(iou_t: float):
        pc_tp: Dict[int, int] = {c: 0 for c in all_class_ids}
        pc_fp: Dict[int, int] = {c: 0 for c in all_class_ids}
        pc_fn: Dict[int, int] = {c: 0 for c in all_class_ids}
        pc_scores: Dict[int, List[Tuple[float, bool]]] = {c: [] for c in all_class_ids}

        for ann, preds in zip(annotations, predictions):
            gts_by_class: Dict[int, List[GroundTruthBox]] = {}
            for gt in ann.ground_truths:
                gts_by_class.setdefault(gt.class_id, []).append(gt)

            matched_gt: set = set()
            sorted_preds = sorted(preds, key=lambda p: p.confidence, reverse=True)

            for pred in sorted_preds:
                gts = gts_by_class.get(pred.class_id, [])
                best_iou = iou_t
                best_idx: Optional[int] = None

                for i, gt in enumerate(gts):
                    iou = _box_iou_normalised(pred.bbox, gt.bbox)
                    key = (ann.image_path, pred.class_id, i)
                    if iou >= best_iou and key not in matched_gt:
                        best_iou = iou
                        best_idx = i

                if best_idx is not None:
                    key = (ann.image_path, pred.class_id, best_idx)
                    matched_gt.add(key)
                    pc_tp[pred.class_id] = pc_tp.get(pred.class_id, 0) + 1
                    pc_scores.setdefault(pred.class_id, []).append((pred.confidence, True))
                else:
                    pc_fp[pred.class_id] = pc_fp.get(pred.class_id, 0) + 1
                    pc_scores.setdefault(pred.class_id, []).append((pred.confidence, False))

            for cls_id, gts in gts_by_class.items():
                matched = sum(
                    1 for i in range(len(gts))
                    if (ann.image_path, cls_id, i) in matched_gt
                )
                pc_fn[cls_id] = pc_fn.get(cls_id, 0) + (len(gts) - matched)

        aps: List[float] = []
        for cls_id in all_class_ids:
            scores = sorted(pc_scores.get(cls_id, []), key=lambda x: -x[0])
            if not scores:
                continue
            tp_cum, fp_cum = 0, 0
            recalls_c, precs_c = [], []
            gt_count = sum(len(a.ground_truths) for a in annotations
                           if any(g.class_id == cls_id for g in a.ground_truths))
            for _, is_tp in scores:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                recalls_c.append(tp_cum / max(gt_count, 1))
                precs_c.append(tp_cum / max(tp_cum + fp_cum, 1))
            aps.append(compute_ap(recalls_c, precs_c))

        return pc_tp, pc_fp, pc_fn, aps

    # Primary evaluation at requested iou_threshold
    per_class_tp, per_class_fp, per_class_fn, base_aps = _eval_at_iou(iou_threshold)

    tp_total = sum(per_class_tp.values())
    fp_total = sum(per_class_fp.values())
    fn_total = sum(per_class_fn.values())

    precision = tp_total / max(tp_total + fp_total, 1)
    recall    = tp_total / max(tp_total + fn_total, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-9)
    map50     = float(np.mean(base_aps)) if base_aps else 0.0

    # mAP@50-95
    map50_95_values: List[float] = []
    for iou_t in np.arange(0.50, 1.00, 0.05):
        if abs(iou_t - iou_threshold) < 1e-5:
            map50_95_values.append(map50)
        else:
            _, _, _, sub_aps = _eval_at_iou(float(iou_t))
            map50_95_values.append(float(np.mean(sub_aps)) if sub_aps else 0.0)
    map50_95 = float(np.mean(map50_95_values)) if map50_95_values else map50

    per_class_results = {}
    for cls_id in all_class_ids:
        cname = class_names.get(cls_id, str(cls_id))
        tp = per_class_tp.get(cls_id, 0)
        fp = per_class_fp.get(cls_id, 0)
        fn = per_class_fn.get(cls_id, 0)
        cp = tp / max(tp + fp, 1)
        cr = tp / max(tp + fn, 1)
        cf = 2 * cp * cr / max(cp + cr, 1e-9)
        per_class_results[cname] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(cp, 4),
            "recall": round(cr, 4),
            "f1": round(cf, 4),
        }

    return EvaluationMetrics(
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        map50=round(map50, 4),
        map50_95=round(map50_95, 4),
        total_images=len(annotations),
        total_gt_boxes=total_gt,
        total_pred_boxes=total_pred,
        true_positives=tp_total,
        false_positives=fp_total,
        false_negatives=fn_total,
        per_class=per_class_results,
    )


# ------------------------------------------------------------------
# Full evaluation run
# ------------------------------------------------------------------

async def run_evaluation(
    config: EvaluationConfig,
    detector=None,
) -> EvaluationMetrics:
    """
    Load dataset, run inference, compute metrics, and save results.
    """
    import cv2 as opencv

    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        m = EvaluationMetrics(status="error")
        m.error_message = f"Dataset path not found: {dataset_path}"
        logger.error(m.error_message)
        return m

    annotations = load_yolo_dataset(dataset_path, split="val")
    if not annotations:
        m = EvaluationMetrics(status="no_data")
        m.error_message = "No labeled images found in dataset."
        return m

    if detector is None:
        m = EvaluationMetrics(status="error")
        m.error_message = "No detector provided for evaluation."
        return m

    all_predictions: List[List[PredBox]] = []
    inference_times: List[float] = []

    for ann in annotations:
        img = opencv.imread(ann.image_path)
        if img is None:
            all_predictions.append([])
            continue

        h, w = img.shape[:2]
        t0 = time.perf_counter()
        det_result = detector.detect(img, conf=config.confidence_threshold)
        inference_times.append((time.perf_counter() - t0) * 1000)

        preds: List[PredBox] = []
        for d in det_result.detections:
            x1, y1, x2, y2 = d.bbox
            preds.append(PredBox(
                class_id=d.class_id,
                confidence=d.confidence,
                bbox=(x1 / w, y1 / h, x2 / w, y2 / h),
            ))
        all_predictions.append(preds)

    metrics = evaluate_detections(
        annotations, all_predictions,
        iou_threshold=config.iou_threshold,
        class_names=config.class_names or detector.supported_classes,
    )

    avg_inf = float(np.mean(inference_times)) if inference_times else 0.0
    metrics.avg_inference_ms = round(avg_inf, 2)
    metrics.fps = round(1000 / avg_inf, 2) if avg_inf > 0 else 0.0

    # Save results
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    result_id = str(uuid.uuid4())[:8]
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = settings.results_dir / f"eval_{ts}_{result_id}.json"

    result_data = {
        "config": asdict(config),
        "metrics": asdict(metrics),
        "result_id": result_id,
    }
    out_path.write_text(json.dumps(result_data, indent=2))
    logger.success(f"Evaluation results saved: {out_path}")

    return metrics

