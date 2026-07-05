"""Training hyperparameter configs for the detector and classifiers."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass
class DetectorTrainConfig:
    data_yaml: Path
    epochs: int = 100
    batch_size: int = 16
    image_size: int = 640
    lr0: float = 0.01
    device: str = "cpu"
    output_dir: Path = Path("runs/detect")
    # Starting checkpoint: a stock "yolov8n.pt" to fine-tune from COCO
    # pretraining, or a path to a previous UECFOOD-256 checkpoint to resume from.
    pretrained_weights: str = "yolov8n.pt"
    run_name: str = "train"
    # DataLoader worker processes. On Windows these are spawned (not forked),
    # which requires the launching script to guard its top-level code with
    # `if __name__ == "__main__":` -- set to 0 to sidestep that entirely for
    # small/quick runs.
    workers: int = 8


@dataclass
class ClassifierTrainConfig:
    backbone: Literal["resnet50", "efficientnet_b4"]
    data_dir: Path
    # None = all 101 Food-101 classes; set to e.g. 10-20 for a fast smoke test.
    num_classes: int | None = None
    # 224 is standard for ResNet-50; EfficientNet-B4 is normally trained at a
    # larger resolution (e.g. 380) -- override per backbone to compare fairly.
    image_size: int = 224
    # Two-phase strategy: freeze backbone / train head only, then unfreeze
    # everything and fine-tune end-to-end at a smaller learning rate.
    head_epochs: int = 3
    finetune_epochs: int = 5
    batch_size: int = 32
    head_lr: float = 1e-3
    finetune_lr: float = 1e-4
    device: str = "cpu"
    output_dir: Path = Path("runs/classify")
    run_name: str = "train"
    # See DetectorTrainConfig.workers -- same Windows spawn caveat applies.
    workers: int = 0
    download: bool = False
