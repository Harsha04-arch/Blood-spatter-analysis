"""
Module 2 — ResNet50-based spatter classifier.

Architecture:
  - Pretrained ResNet50 backbone (ImageNet weights)
  - Replaced final FC layer for N spatter classes
  - Additional velocity estimation head

No hard-coded predictions. If model weights are unavailable,
an explicit error is raised.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from backend.utils.logger import get_logger

log = get_logger(__name__)

_SPATTER_CLASSES = [
    "Passive Drip",
    "Transfer Smear",
    "Medium Velocity Impact Spatter",
    "High Velocity Impact Spatter",
]

_VELOCITY_CLASSES = [
    "Passive / Low Velocity",
    "Medium Velocity",
    "High Velocity",
]


class BPAClassifier(nn.Module):
    """
    ResNet50 with dual classification heads:
      1. Spatter type (4 classes)
      2. Impact velocity category (3 classes)
    """

    def __init__(self, num_spatter: int = 4, num_velocity: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        # Remove original FC
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        self.dropout = nn.Dropout(0.4)
        self.spatter_head = nn.Linear(feature_dim, num_spatter)
        self.velocity_head = nn.Linear(feature_dim, num_velocity)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.spatter_head(x), self.velocity_head(x)


# --------------------------------------------------------------------------- #
#  Singleton model loader                                                      #
# --------------------------------------------------------------------------- #

_model_cache: dict = {}


def load_model(model_path: Path, device: torch.device | None = None) -> BPAClassifier:
    """Load model from checkpoint. Raises FileNotFoundError if checkpoint is absent."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    key = str(model_path)
    if key not in _model_cache:
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {model_path}. "
                "Please train the model first via the Training page."
            )
        log.info(f"Loading model from {model_path} on {device}")
        model = BPAClassifier(
            num_spatter=len(_SPATTER_CLASSES),
            num_velocity=len(_VELOCITY_CLASSES),
            pretrained=False,
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()
        _model_cache[key] = model
        log.info("Model loaded and cached.")
    return _model_cache[key]


def clear_model_cache():
    _model_cache.clear()


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
