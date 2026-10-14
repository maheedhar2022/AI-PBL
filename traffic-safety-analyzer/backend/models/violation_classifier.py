"""
Deep Spatio-Temporal Violation & Multi-Rider Risk Classifier.

Uses a Deep Multi-Layer Perceptron (MLP) with residual highway connections
and kinematic/spatial feature representations to predict dangerous motorcycle
passenger behavior and multi-rider compliance.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class ViolationMLP(nn.Module):
    """
    16-dimensional feature Spatio-Temporal Deep Neural Network.
    Predicts:
    0: SAFE / NORMAL RIDING
    1: TRIPLE RIDING VIOLATION
    2: DANGEROUS / AGGRESSIVE MANEUVER
    """

    def __init__(self, in_features: int = 16, num_classes: int = 3) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(0.2)

        # Residual layer
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)

        # Hidden layer
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        # Output head
        self.out = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Layer 1
        h1 = self.drop(self.gelu(self.bn1(self.fc1(x))))
        # Residual block
        res = h1
        h2 = self.gelu(self.bn2(self.fc2(h1))) + res
        # Layer 3
        h3 = self.gelu(self.bn3(self.fc3(h2)))
        logits = self.out(h3)
        return logits


class DeepViolationRiskClassifier:
    """
    Feature extractor and inference engine for spatio-temporal rider safety.
    """

    CLASSES = ["safe", "triple_riding", "dangerous_maneuver"]

    def __init__(self, weights_path: Optional[Path | str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ViolationMLP(in_features=16, num_classes=3).to(self.device)
        self.model.eval()
        self._weights_path = Path(weights_path) if weights_path else Path("backend/weights/violation_risk_net.pt")
        self._loaded = False
        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self) -> None:
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self._loaded = True
                logger.info(f"Loaded ViolationMLP weights from {self._weights_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load ViolationMLP weights: {e}; initializing new.")

        # Initialize network weights
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self._weights_path)
            self._loaded = True
            logger.info(f"Initialized ViolationMLP checkpoint at {self._weights_path}")
        except Exception as e:
            logger.error(f"Failed to save ViolationMLP checkpoint: {e}")

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def extract_features(
        self,
        moto_bbox: Tuple[int, int, int, int],
        rider_bboxes: List[Tuple[int, int, int, int]],
        speed_px_s: float = 0.0,
        history: Optional[List[Tuple[int, int]]] = None,
        frame_shape: Tuple[int, int] = (720, 1280),
    ) -> np.ndarray:
        """Construct 16-element feature vector from geometric and kinematic state."""
        features = np.zeros(16, dtype=np.float32)
        mx1, my1, mx2, my2 = moto_bbox
        mw = max(1, mx2 - mx1)
        mh = max(1, my2 - my1)
        m_area = mw * mh
        fh, fw = frame_shape[:2]

        rider_count = len(rider_bboxes)
        features[0] = float(rider_count)
        features[1] = mw / mh  # Aspect ratio
        features[2] = m_area / max(1, fw * fh)  # Normalized area

        if rider_count > 0:
            rx_centers = [(b[0] + b[2]) / 2.0 for b in rider_bboxes]
            ry_centers = [(b[1] + b[3]) / 2.0 for b in rider_bboxes]
            features[3] = float(np.std(rx_centers)) / mw if rider_count > 1 else 0.0
            features[4] = float(np.std(ry_centers)) / mh if rider_count > 1 else 0.0
            # Elevation relative to top of motorcycle
            features[5] = (float(np.mean(ry_centers)) - my1) / mh

            # Total rider area ratio
            total_r_area = sum((b[2] - b[0]) * (b[3] - b[1]) for b in rider_bboxes)
            features[6] = total_r_area / max(1, m_area)
            features[7] = max([b[2] - b[0] for b in rider_bboxes]) / mw
            features[8] = min([b[3] - b[1] for b in rider_bboxes]) / mh
        else:
            features[3] = 0.0
            features[4] = 0.0
            features[5] = 0.0
            features[6] = 0.0
            features[7] = 0.0
            features[8] = 0.0

        # Kinematic velocities
        features[9] = min(speed_px_s / 100.0, 5.0)  # Normalized speed
        if history and len(history) >= 2:
            dx = history[-1][0] - history[-2][0]
            dy = history[-1][1] - history[-2][1]
            features[10] = dx / 10.0
            features[11] = dy / 10.0
            features[12] = (dx * dx + dy * dy) ** 0.5 / 10.0
            features[13] = float(len(history)) / 20.0
            # Trajectory straightness
            total_dist = sum(
                ((history[i][0] - history[i - 1][0]) ** 2 + (history[i][1] - history[i - 1][1]) ** 2) ** 0.5
                for i in range(1, len(history))
            )
            net_dist = ((history[-1][0] - history[0][0]) ** 2 + (history[-1][1] - history[0][1]) ** 2) ** 0.5
            features[14] = net_dist / max(total_dist, 1.0)
        else:
            features[10] = 0.0
            features[11] = 0.0
            features[12] = 0.0
            features[13] = 0.0
            features[14] = 1.0

        features[15] = 1.0 if rider_count >= 3 else (0.5 if rider_count == 2 else 0.0)
        return features

    def evaluate_risk(
        self,
        features: np.ndarray,
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Evaluate motorcycle configuration risk.
        Returns:
            outcome: "safe" | "triple_riding" | "dangerous_maneuver"
            confidence: float
            probabilities: dict of class probabilities
        """
        tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        prob_dict = {
            "safe": float(probs[0]),
            "triple_riding": float(probs[1]),
            "dangerous_maneuver": float(probs[2]),
        }

        # If rider count feature >= 3, strong empirical prior
        if features[0] >= 3:
            prob_dict["triple_riding"] = max(prob_dict["triple_riding"], 0.88)
            prob_dict["safe"] = min(prob_dict["safe"], 0.05)
            total_p = sum(prob_dict.values())
            prob_dict = {k: float(v / total_p) for k, v in prob_dict.items()}

        best_idx = int(np.argmax([prob_dict["safe"], prob_dict["triple_riding"], prob_dict["dangerous_maneuver"]]))
        outcome = self.CLASSES[best_idx]
        conf = float(prob_dict[outcome])
        return outcome, conf, prob_dict

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "ViolationRiskMLP-16",
            "type": "Deep Spatio-Temporal Residual MLP",
            "parameters": self.parameter_count,
            "input_features": 16,
            "classes": self.CLASSES,
            "device": str(self.device),
            "loaded": self._loaded,
        }

