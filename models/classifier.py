"""Common interface shared by the Food-101 classifier backbones."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch import nn


@dataclass
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float


def freeze_all_except(model: nn.Module, trainable_prefixes: tuple[str, ...]) -> None:
    """Freeze every parameter except those whose name starts with one of `trainable_prefixes`.

    Used for phase 1 of the two-phase fine-tune strategy: train only the
    replaced classification head while the pretrained backbone stays frozen.
    """
    for name, param in model.named_parameters():
        param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)


def unfreeze_all(model: nn.Module) -> None:
    """Make every parameter trainable again (phase 2: full-network fine-tune)."""
    for param in model.parameters():
        param.requires_grad = True


class FoodClassifier(ABC):
    """Interface implemented by each classifier backbone (ResNet-50, EfficientNet-B4)."""

    def __init__(self, weights_path: Path, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights into `self.model`."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, image_crop: np.ndarray) -> ClassificationResult:
        """Classify a single cropped food image."""
        raise NotImplementedError
