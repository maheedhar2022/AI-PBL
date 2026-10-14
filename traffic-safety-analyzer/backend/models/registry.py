"""
Unified Deep Learning Model Registry.

Manages all deep neural networks used across the traffic safety analysis pipeline:
1. YOLOv8 Deep Object Detection
2. DeepHelmetCNN with Squeeze-and-Excitation Attention
3. DeepReIDNet 128-d Visual Feature Embedding
4. SpatioTemporalViolationMLP Kinematic & Multi-Rider Risk Classifier
"""
from __future__ import annotations

import time
from typing import Dict, Any, Optional

import numpy as np
import torch
from loguru import logger

from backend.detection.detector import get_detector, load_detector, YOLODetector
from backend.models.helmet_classifier import DeepHelmetClassifier
from backend.models.reid_extractor import DeepReIDExtractor
from backend.models.violation_classifier import DeepViolationRiskClassifier


class ModelRegistry:
    """Singleton model coordinator and benchmark harness."""

    _instance: Optional[ModelRegistry] = None

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing ModelRegistry on device: {self.device}")

        # 1. Detector
        self._detector: Optional[YOLODetector] = get_detector()
        if self._detector is None:
            self._detector = load_detector()

        # 2. Helmet CNN
        self.helmet_classifier = DeepHelmetClassifier()

        # 3. Visual Re-ID Net
        self.reid_extractor = DeepReIDExtractor()

        # 4. Spatio-Temporal Risk Classifier
        self.violation_classifier = DeepViolationRiskClassifier()

    @classmethod
    def get_instance(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = ModelRegistry()
        return cls._instance

    @property
    def detector(self) -> Optional[YOLODetector]:
        if self._detector is None:
            self._detector = get_detector() or load_detector()
        return self._detector

    def get_all_model_specs(self) -> Dict[str, Any]:
        """Return comprehensive specifications for all loaded deep learning models."""
        yolo_params = 3_157_200  # yolov8n baseline param count
        yolo_loaded = self.detector is not None and self.detector.is_loaded

        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "models": {
                "object_detector": {
                    "name": self.detector.model_name if self.detector else "YOLOv8n",
                    "type": "Deep Object Detector (CSPDarknet53 + PANet)",
                    "parameters": yolo_params,
                    "input_resolution": "3x640x640",
                    "framework": "Ultralytics PyTorch",
                    "classes_count": len(self.detector.supported_classes) if self.detector else 80,
                    "loaded": yolo_loaded,
                },
                "helmet_classifier": self.helmet_classifier.get_info(),
                "reid_extractor": self.reid_extractor.get_info(),
                "violation_classifier": self.violation_classifier.get_info(),
            },
            "total_deep_parameters": (
                yolo_params
                + self.helmet_classifier.parameter_count
                + self.reid_extractor.parameter_count
                + self.violation_classifier.parameter_count
            ),
        }

    def run_benchmark(self, iterations: int = 15) -> Dict[str, Any]:
        """
        Execute an actual deep learning forward pass benchmark on current hardware.
        Measures mean, min, p95 latencies and throughput (FPS).
        """
        logger.info(f"Running Deep Learning Benchmark ({iterations} iterations)...")

        # 1. Benchmark Helmet CNN
        dummy_crop = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        # Warmup
        self.helmet_classifier.classify_crop(dummy_crop)
        helmet_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.helmet_classifier.classify_crop(dummy_crop)
            helmet_times.append((time.perf_counter() - t0) * 1000.0)

        # 2. Benchmark ReID Extractor
        dummy_person = np.random.randint(0, 255, (128, 64, 3), dtype=np.uint8)
        self.reid_extractor.extract_embedding(dummy_person)
        reid_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.reid_extractor.extract_embedding(dummy_person)
            reid_times.append((time.perf_counter() - t0) * 1000.0)

        # 3. Benchmark Violation Risk MLP
        dummy_features = np.random.rand(16).astype(np.float32)
        self.violation_classifier.evaluate_risk(dummy_features)
        risk_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            self.violation_classifier.evaluate_risk(dummy_features)
            risk_times.append((time.perf_counter() - t0) * 1000.0)

        # 4. Benchmark YOLOv8 (5 iterations due to full frame size)
        yolo_times = []
        if self.detector and self.detector.is_loaded:
            dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
            yolo_iters = min(iterations, 5)
            for _ in range(yolo_iters):
                t0 = time.perf_counter()
                self.detector.detect(dummy_frame)
                yolo_times.append((time.perf_counter() - t0) * 1000.0)

        def _stats(arr: list[float]) -> dict:
            if not arr:
                return {"mean_ms": 0.0, "p95_ms": 0.0, "fps": 0.0}
            mean = float(np.mean(arr))
            p95 = float(np.percentile(arr, 95))
            return {
                "mean_ms": round(mean, 2),
                "p95_ms": round(p95, 2),
                "min_ms": round(float(np.min(arr)), 2),
                "max_ms": round(float(np.max(arr)), 2),
                "throughput_fps": round(1000.0 / mean, 1) if mean > 0 else 0.0,
            }

        return {
            "device": self.device,
            "benchmark_iterations": iterations,
            "results": {
                "yolov8_detection": _stats(yolo_times),
                "helmet_cnn_inference": _stats(helmet_times),
                "reid_embedding_extraction": _stats(reid_times),
                "violation_mlp_risk": _stats(risk_times),
            },
            "timestamp": time.time(),
        }


# Global accessor
def get_model_registry() -> ModelRegistry:
    return ModelRegistry.get_instance()

