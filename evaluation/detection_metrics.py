"""Detection evaluation for the fine-tuned YOLOv8 food detector.

Calls Ultralytics' own `model.val()` directly rather than re-implementing
mAP/precision/recall matching by hand -- it already does IoU matching,
per-class AP, and mAP@50 / mAP@50-95 aggregation correctly.
"""

import argparse
from pathlib import Path
from typing import Any

from ultralytics import YOLO


def evaluate_detector(
    weights_path: Path | str,
    data_yaml: Path | str,
    device: str = "cpu",
    split: str = "val",
) -> dict[str, float]:
    """Run Ultralytics' built-in validation and return the headline metrics."""
    model = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), device=device, split=split)

    results = metrics.results_dict
    return {
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "mAP50": float(results["metrics/mAP50(B)"]),
        "mAP50-95": float(results["metrics/mAP50-95(B)"]),
    }


def print_report(metrics: dict[str, Any], weights_path: Path | str, data_yaml: Path | str) -> None:
    print(f"Detection evaluation: {weights_path}")
    print(f"  data:      {data_yaml}")
    print(f"  precision: {metrics['precision']:.4f}")
    print(f"  recall:    {metrics['recall']:.4f}")
    print(f"  mAP50:     {metrics['mAP50']:.4f}")
    print(f"  mAP50-95:  {metrics['mAP50-95']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data-yaml", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    metrics = evaluate_detector(args.weights, args.data_yaml, device=args.device, split=args.split)
    print_report(metrics, args.weights, args.data_yaml)


if __name__ == "__main__":
    main()
