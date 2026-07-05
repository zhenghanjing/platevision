"""Fine-tune YOLOv8 on the UECFOOD-256 dataset (converted to YOLO format)."""

import argparse
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics

from training.config import DetectorTrainConfig


def train_detector(config: DetectorTrainConfig) -> DetMetrics | dict | None:
    """Fine-tune a YOLOv8 model per `config`, saving checkpoints under
    `config.output_dir / config.run_name`. Returns Ultralytics' training results."""
    model = YOLO(config.pretrained_weights)
    return model.train(
        data=str(config.data_yaml),
        epochs=config.epochs,
        batch=config.batch_size,
        imgsz=config.image_size,
        lr0=config.lr0,
        device=config.device,
        # Resolve to an absolute path: Ultralytics auto-prepends its own
        # global `runs_dir/<task>` to any *relative* project path, which
        # would otherwise double up with our own "runs/detect" convention.
        project=str(Path(config.output_dir).resolve()),
        name=config.run_name,
        workers=config.workers,
        exist_ok=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/detect"))
    parser.add_argument("--pretrained-weights", default="yolov8n.pt")
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    config = DetectorTrainConfig(
        data_yaml=args.data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        image_size=args.image_size,
        lr0=args.lr0,
        device=args.device,
        output_dir=args.output_dir,
        pretrained_weights=args.pretrained_weights,
        run_name=args.run_name,
        workers=args.workers,
    )
    train_detector(config)


if __name__ == "__main__":
    main()
