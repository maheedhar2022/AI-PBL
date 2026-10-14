"""
Deep Learning Helmet Classifier Module.

Implements a Convolutional Neural Network (CNN) with Squeeze-and-Excitation
channel attention for classifying rider head crops into:
- helmet (class 0)
- no_helmet (class 1)
- uncertain (class 2)
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block."""

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, reduced),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.fc(x).view(b, c, 1, 1)
        return x * w


class ResidualBlock(nn.Module):
    """Residual convolution block with BatchNorm and SiLU."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.silu = nn.SiLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.silu(out + residual)
        return out


class DeepHelmetCNN(nn.Module):
    """
    Deep Convolutional Neural Network for Helmet Compliance Classification.
    Accepts 3x64x64 head crop tensors and predicts 3 classes:
    0: helmet
    1: no_helmet
    2: uncertain
    """

    def __init__(self, num_classes: int = 3) -> None:
        super().__init__()
        # Stage 1: Initial feature extraction
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
            nn.MaxPool2d(2, 2),  # 32x32
        )

        # Stage 2: Deep features with residual blocks
        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(inplace=True),
            ResidualBlock(64),
            nn.MaxPool2d(2, 2),  # 16x16
        )

        # Stage 3: High-level semantic features with SE Attention
        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.SiLU(inplace=True),
            SEBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),  # 1x1
        )

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        logits = self.classifier(x)
        return logits


class DeepHelmetClassifier:
    """
    Inference manager and pre/post-processor for DeepHelmetCNN.
    """

    CLASSES = ["helmet", "no_helmet", "uncertain"]

    def __init__(self, weights_path: Optional[Path | str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepHelmetCNN(num_classes=3).to(self.device)
        self.model.eval()
        self._weights_path = Path(weights_path) if weights_path else Path("backend/weights/helmet_classifier.pt")
        self._loaded = False
        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self) -> None:
        """Load trained weights or initialize and bootstrap an initial checkpoint."""
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self._loaded = True
                logger.info(f"Loaded DeepHelmetCNN weights from {self._weights_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load existing weights: {e}; initializing new.")

        # Initialize network weights with Kaiming He initialization
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        # Save baseline model checkpoint
        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self._weights_path)
            self._loaded = True
            logger.info(f"Initialized DeepHelmetCNN checkpoint at {self._weights_path}")
        except Exception as e:
            logger.error(f"Failed to save baseline checkpoint: {e}")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def preprocess_crop(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess an RGB/BGR image crop into normalized PyTorch tensor."""
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        # Convert BGR to RGB
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 and crop.shape[2] == 3 else crop
        resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
        # Normalization with ImageNet standard mean & std
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def classify_crop(self, head_crop: np.ndarray) -> Tuple[str, float, Dict[str, float]]:
        """
        Classify a single head crop.
        Returns:
            label: "helmet" | "no_helmet" | "uncertain"
            confidence: float (0.0 - 1.0)
            class_probabilities: dict mapping label -> probability
        """
        tensor = self.preprocess_crop(head_crop)
        if tensor is None:
            return "uncertain", 0.33, {"helmet": 0.33, "no_helmet": 0.33, "uncertain": 0.34}

        tensor = tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()

        prob_dict = {
            "helmet": float(probs[0]),
            "no_helmet": float(probs[1]),
            "uncertain": float(probs[2]),
        }
        best_idx = int(np.argmax(probs))
        label = self.CLASSES[best_idx]
        conf = float(probs[best_idx])
        return label, conf, prob_dict

    def classify_batch(self, head_crops: List[np.ndarray]) -> List[Tuple[str, float, Dict[str, float]]]:
        """Batch inference for multiple head crops simultaneously."""
        tensors = []
        valid_indices = []
        for i, crop in enumerate(head_crops):
            t = self.preprocess_crop(crop)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        results: List[Tuple[str, float, Dict[str, float]]] = [
            ("uncertain", 0.33, {"helmet": 0.33, "no_helmet": 0.33, "uncertain": 0.34})
            for _ in range(len(head_crops))
        ]

        if not tensors:
            return results

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits = self.model(batch_tensor)
            all_probs = F.softmax(logits, dim=1).cpu().numpy()

        for original_idx, probs in zip(valid_indices, all_probs):
            prob_dict = {
                "helmet": float(probs[0]),
                "no_helmet": float(probs[1]),
                "uncertain": float(probs[2]),
            }
            best_idx = int(np.argmax(probs))
            results[original_idx] = (self.CLASSES[best_idx], float(probs[best_idx]), prob_dict)

        return results

    def get_info(self) -> Dict[str, Any]:
        """Get model architectural specification."""
        return {
            "name": "DeepHelmetCNN-SE",
            "type": "Convolutional Neural Network + SE Attention",
            "parameters": self.parameter_count,
            "input_resolution": "3x64x64",
            "classes": self.CLASSES,
            "device": str(self.device),
            "loaded": self._loaded,
        }

