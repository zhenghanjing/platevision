"""Fine-tune a Food-101 classifier backbone (ResNet-50 or EfficientNet-B4).

Two-phase strategy (per the project proposal):
  1. Freeze the ImageNet-pretrained backbone, train only the replaced
     classification head for a few epochs.
  2. Unfreeze the whole network and fine-tune end-to-end at a smaller
     learning rate.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from data.food101 import Food101Dataset
from models.checkpoint import save_checkpoint
from models.classifier import freeze_all_except, unfreeze_all
from models.efficientnet_b4 import build_efficientnet_b4
from models.efficientnet_b4 import HEAD_PREFIX as EFFICIENTNET_HEAD_PREFIX
from models.resnet50 import build_resnet50
from models.resnet50 import HEAD_PREFIX as RESNET_HEAD_PREFIX
from training.config import ClassifierTrainConfig

_BACKBONES = {
    "resnet50": (build_resnet50, RESNET_HEAD_PREFIX),
    "efficientnet_b4": (build_efficientnet_b4, EFFICIENTNET_HEAD_PREFIX),
}


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over `loader`; trains if `optimizer` is given, else evaluates. Returns (loss, accuracy)."""
    is_train = optimizer is not None
    model.train(is_train)
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_classifier(config: ClassifierTrainConfig) -> dict[str, Any]:
    """Fine-tune `config.backbone` on Food-101. Returns per-epoch loss/accuracy history."""
    build_fn, head_prefix = _BACKBONES[config.backbone]

    train_set = Food101Dataset(
        root_dir=config.data_dir,
        split="train",
        download=config.download,
        num_classes=config.num_classes,
        image_size=config.image_size,
    )
    val_set = Food101Dataset(
        root_dir=config.data_dir,
        split="val",
        download=False,
        num_classes=config.num_classes,
        image_size=config.image_size,
    )
    class_names = train_set.classes

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size, shuffle=True, num_workers=config.workers
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size, shuffle=False, num_workers=config.workers
    )

    model = build_fn(num_classes=len(class_names), pretrained=True).to(config.device)

    history: dict[str, Any] = {"backbone": config.backbone, "head": [], "finetune": []}

    def _run_phase(phase: str, epochs: int, optimizer: torch.optim.Optimizer) -> None:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = _run_epoch(model, train_loader, config.device, optimizer)
            val_loss, val_acc = _run_epoch(model, val_loader, config.device)
            print(
                f"[{config.backbone}] {phase} epoch {epoch}/{epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} ({time.time() - t0:.1f}s)",
                flush=True,
            )
            history[phase].append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }
            )

    # Phase 1: frozen backbone, head-only training.
    freeze_all_except(model, (head_prefix,))
    head_optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=config.head_lr
    )
    _run_phase("head", config.head_epochs, head_optimizer)

    # Phase 2: unfrozen, full-network fine-tune at a smaller learning rate.
    unfreeze_all(model)
    finetune_optimizer = torch.optim.Adam(model.parameters(), lr=config.finetune_lr)
    _run_phase("finetune", config.finetune_epochs, finetune_optimizer)

    checkpoint_path = Path(config.output_dir) / f"{config.run_name}_{config.backbone}.pt"
    save_checkpoint(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "backbone": config.backbone,
            "image_size": config.image_size,
        },
        checkpoint_path,
    )
    history["checkpoint_path"] = str(checkpoint_path)
    last_phase = history["finetune"] or history["head"]
    history["final_val_acc"] = last_phase[-1]["val_acc"]

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", choices=list(_BACKBONES), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--head-epochs", type=int, default=3)
    parser.add_argument("--finetune-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-lr", type=float, default=1e-3)
    parser.add_argument("--finetune-lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/classify"))
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    config = ClassifierTrainConfig(
        backbone=args.backbone,
        data_dir=args.data_dir,
        num_classes=args.num_classes,
        image_size=args.image_size,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        head_lr=args.head_lr,
        finetune_lr=args.finetune_lr,
        device=args.device,
        output_dir=args.output_dir,
        run_name=args.run_name,
        workers=args.workers,
        download=args.download,
    )
    train_classifier(config)


if __name__ == "__main__":
    main()
