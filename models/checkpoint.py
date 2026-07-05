"""Checkpoint save/load helpers shared across the detector and classifiers."""

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(state: dict[str, Any], path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: Path, map_location: str = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
