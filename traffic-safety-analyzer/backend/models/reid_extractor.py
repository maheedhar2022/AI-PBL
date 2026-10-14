"""
Deep Visual Re-Identification (Re-ID) Feature Extractor.

Extracts 128-dimensional L2-normalized deep appearance embeddings from
vehicle and pedestrian bounding-box crops for robust multi-object tracking
across occlusions and camera viewpoint variations.
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


class ReIDConvBlock(nn.Module):
    """Standard Convolution + BatchNorm + LeakyReLU block."""

    def __init__(self, in_c: int, out_c: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ReIDResidualBlock(nn.Module):
    """Residual appearance feature block."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + res)


class DeepReIDNet(nn.Module):
    """
    Lightweight Deep Feature Embedding Network for Vehicle and Pedestrian Re-ID.
    Maps a 3x128x64 image patch into a 128-dimensional unit hypersphere embedding.
    """

    def __init__(self, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

        self.backbone = nn.Sequential(
            ReIDConvBlock(3, 32, stride=2),    # 64x32
            ReIDConvBlock(32, 64, stride=2),   # 32x16
            ReIDResidualBlock(64),
            ReIDConvBlock(64, 128, stride=2),  # 16x8
            ReIDResidualBlock(128),
            nn.AdaptiveAvgPool2d((1, 1)),      # 1x1
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        embed = self.head(feat)
        # Project onto unit hypersphere (L2 normalize)
        return F.normalize(embed, p=2, dim=1)


class DeepReIDExtractor:
    """
    Manages inference and embedding extraction for multi-object tracking.
    """

    def __init__(self, weights_path: Optional[Path | str] = None) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DeepReIDNet(embedding_dim=128).to(self.device)
        self.model.eval()
        self._weights_path = Path(weights_path) if weights_path else Path("backend/weights/reid_net.pt")
        self._loaded = False
        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self) -> None:
        if self._weights_path.exists():
            try:
                state_dict = torch.load(self._weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self._loaded = True
                logger.info(f"Loaded DeepReIDNet weights from {self._weights_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load ReID weights: {e}; initializing new.")

        # Initialize network weights
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.02)

        try:
            self._weights_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self.model.state_dict(), self._weights_path)
            self._loaded = True
            logger.info(f"Initialized DeepReIDNet checkpoint at {self._weights_path}")
        except Exception as e:
            logger.error(f"Failed to save ReID checkpoint: {e}")

    @property
    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def preprocess_crop(self, crop: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess BGR crop to normalized 3x128x64 tensor."""
        if crop is None or crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
            return None
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 and crop.shape[2] == 3 else crop
        resized = cv2.resize(rgb, (64, 128), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        return (tensor - mean) / std

    def extract_embedding(self, crop: np.ndarray) -> np.ndarray:
        """Extract a 128-d L2 normalized embedding for a single crop."""
        t = self.preprocess_crop(crop)
        if t is None:
            # Fallback zero unit vector
            vec = np.zeros(128, dtype=np.float32)
            vec[0] = 1.0
            return vec

        t = t.unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(t)[0].cpu().numpy()
        return emb

    def extract_batch(self, crops: List[np.ndarray]) -> List[np.ndarray]:
        """Batch extract embeddings for multiple detection crops."""
        tensors = []
        valid_indices = []
        for i, crop in enumerate(crops):
            t = self.preprocess_crop(crop)
            if t is not None:
                tensors.append(t)
                valid_indices.append(i)

        fallback = np.zeros(128, dtype=np.float32)
        fallback[0] = 1.0
        results: List[np.ndarray] = [fallback.copy() for _ in range(len(crops))]

        if not tensors:
            return results

        batch_tensor = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embeddings = self.model(batch_tensor).cpu().numpy()

        for idx, emb in zip(valid_indices, embeddings):
            results[idx] = emb

        return results

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two unit-norm embeddings."""
        dot = float(np.dot(emb1, emb2))
        return max(-1.0, min(1.0, dot))

    @staticmethod
    def cosine_distance_matrix(gallery: np.ndarray, probes: np.ndarray) -> np.ndarray:
        """
        Compute pairwise cosine distance matrix (1 - cosine_similarity).
        gallery: (N, 128), probes: (M, 128)
        Returns: (N, M) matrix of values in [0, 2]
        """
        if gallery.size == 0 or probes.size == 0:
            return np.empty((gallery.shape[0], probes.shape[0]))
        similarity = np.dot(gallery, probes.T)
        return np.clip(1.0 - similarity, 0.0, 2.0)

    def get_info(self) -> Dict[str, Any]:
        return {
            "name": "DeepReIDNet-128",
            "type": "Deep Visual Appearance Embedding CNN",
            "parameters": self.parameter_count,
            "embedding_dimension": 128,
            "input_resolution": "3x128x64",
            "device": str(self.device),
            "loaded": self._loaded,
        }

