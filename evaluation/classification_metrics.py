"""Classification evaluation: full-val-set Top-1/Top-5 accuracy, and the
ResNet-50 vs EfficientNet-B4 comparison.

Runs the raw underlying `nn.Module` directly (batched, via DataLoader)
rather than going through `FoodClassifier.predict()`'s one-image-at-a-time
API -- that API is meant for single-crop inference in the live pipeline,
not for scoring thousands of images efficiently.
"""

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from data.food101 import Food101Dataset
from models.checkpoint import load_checkpoint
from models.efficientnet_b4 import build_efficientnet_b4
from models.resnet50 import build_resnet50

_BUILDERS = {"resnet50": build_resnet50, "efficientnet_b4": build_efficientnet_b4}


def _topk_correct(logits: torch.Tensor, labels: torch.Tensor, k: int) -> int:
    top_k = logits.topk(k, dim=1).indices
    return int((top_k == labels.unsqueeze(1)).any(dim=1).sum().item())


def evaluate_classifier_checkpoint(
    checkpoint_path: Path | str,
    data_dir: Path | str,
    device: str = "cpu",
    batch_size: int = 64,
    workers: int = 8,
) -> dict[str, Any]:
    """Run Top-1/Top-5 accuracy for a saved classifier checkpoint over the
    *full* Food-101 val split (whatever `num_classes` it was trained with --
    101 for the full-scale checkpoints)."""
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    backbone = checkpoint["backbone"]
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = _BUILDERS[backbone](num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    val_set = Food101Dataset(
        root_dir=data_dir,
        split="val",
        download=False,
        num_classes=len(class_names) if len(class_names) < 101 else None,
        image_size=image_size,
    )
    loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=workers)

    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            top1_correct += _topk_correct(logits, labels, k=1)
            top5_correct += _topk_correct(logits, labels, k=min(5, len(class_names)))
            total += images.size(0)

    return {
        "backbone": backbone,
        "image_size": image_size,
        "num_classes": len(class_names),
        "num_samples": total,
        "top1_accuracy": top1_correct / total,
        "top5_accuracy": top5_correct / total,
    }


def format_comparison_table(results: list[dict[str, Any]]) -> str:
    """Render a plain-text side-by-side comparison table for a report."""
    header = f"{'Backbone':<18} {'Image size':>10} {'Samples':>8} {'Top-1':>8} {'Top-5':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r['backbone']:<18} {r['image_size']:>10} {r['num_samples']:>8} "
            f"{r['top1_accuracy']:>7.2%} {r['top5_accuracy']:>7.2%}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, help="One or more checkpoint paths")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    results = [
        evaluate_classifier_checkpoint(
            ckpt, args.data_dir, device=args.device, batch_size=args.batch_size, workers=args.workers
        )
        for ckpt in args.checkpoints
    ]
    print(format_comparison_table(results))


if __name__ == "__main__":
    main()
