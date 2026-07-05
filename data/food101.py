"""Food-101 dataset for classifier fine-tuning.

Wraps `torchvision.datasets.Food101` (torchvision already implements the
download, checksum, and `meta/{train,test}.json` parsing for this dataset,
so we reuse it instead of re-parsing the raw archive ourselves).

torchvision's Food101 only defines the official "train" (75,750 images) and
"test" (25,250 images) splits. To get a "val" split for model selection
without touching the held-out test set, we carve a stratified subset out of
"train" (same `val_ratio` fraction taken from every class, fixed `seed` so
the split is reproducible across runs).
"""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import Food101 as _Food101

from data.config import FOOD101_ROOT

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224

Split = Literal["train", "val", "test"]


def build_transforms(split: Split, image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """`image_size`x`image_size` + ImageNet normalization; adds flip/crop/color-jitter for train.

    `image_size` is configurable because EfficientNet-B4 is normally trained
    at a larger resolution (e.g. 380) than the 224 that's standard for
    ResNet-50 -- both backbones share this function, so a fair comparison
    needs it exposed rather than hardcoded.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
                transforms.ToTensor(),
                normalize,
            ]
        )

    # Keep the same resize:crop ratio as the original 256:224 convention.
    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def download_food101(dest_dir: Path = FOOD101_ROOT) -> None:
    """Download Food-101 into `dest_dir` (no-op if already present)."""
    _Food101(root=dest_dir, split="train", download=True)
    _Food101(root=dest_dir, split="test", download=True)


def load_category_map(dest_dir: Path = FOOD101_ROOT) -> dict[int, str]:
    """Return {class_id: class_name} using torchvision's own class ordering."""
    dataset = _Food101(root=dest_dir, split="train", download=False)
    return {idx: name for name, idx in dataset.class_to_idx.items()}


def _stratified_train_val_indices(
    labels: list[int], val_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices so `val_ratio` of each class's examples land in val, deterministically."""
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(labels):
        by_class.setdefault(label, []).append(idx)

    generator = torch.Generator().manual_seed(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []
    for class_indices in by_class.values():
        perm = torch.randperm(len(class_indices), generator=generator).tolist()
        shuffled = [class_indices[i] for i in perm]
        num_val = max(1, round(len(shuffled) * val_ratio))
        val_indices.extend(shuffled[:num_val])
        train_indices.extend(shuffled[num_val:])
    return train_indices, val_indices


class Food101Dataset(Dataset):
    """train/val/test Food-101 split with ImageNet preprocessing (+ train-time augmentation).

    `num_classes` restricts to the first N classes in torchvision's own sorted
    class order (same subset across train/val/test, since that ordering is
    fixed dataset-wide) -- useful for a fast smoke test before scaling up to
    all 101 classes.
    """

    def __init__(
        self,
        root_dir: Path = FOOD101_ROOT,
        split: Split = "train",
        download: bool = False,
        val_ratio: float = 0.1,
        seed: int = 42,
        transform: transforms.Compose | None = None,
        num_classes: int | None = None,
        image_size: int = IMAGE_SIZE,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform or build_transforms(split, image_size)

        underlying_split = "test" if split == "test" else "train"
        base = _Food101(
            root=self.root_dir,
            split=underlying_split,
            download=download,
            transform=self.transform,
        )
        self.classes = base.classes[:num_classes] if num_classes is not None else base.classes
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes)}

        indices = range(len(base._labels))
        if num_classes is not None:
            indices = [i for i in indices if base._labels[i] < num_classes]

        if split == "test":
            self._dataset: Dataset = Subset(base, list(indices))
        else:
            filtered_labels = [base._labels[i] for i in indices]
            train_idx, val_idx = _stratified_train_val_indices(filtered_labels, val_ratio, seed)
            local_idx = train_idx if split == "train" else val_idx
            self._dataset = Subset(base, [indices[i] for i in local_idx])

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int):
        return self._dataset[index]
