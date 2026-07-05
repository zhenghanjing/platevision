"""Paths and constants shared by the dataset modules."""

from pathlib import Path

DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

# Root of the extracted UECFOOD-256 archive: contains category.txt plus one
# numbered folder (1..256) per food category. See data/uecfood256.py for the
# full layout.
UECFOOD256_DIR = DATA_ROOT / "UECFOOD256"

# Best-effort mirror of the official distribution (foodcam.mobi/dataset256.html).
# That page can require manually accepting terms before download, so this URL
# may not always be reachable by script — download_uecfood256() falls back to
# a manually-downloaded archive_path in that case.
UECFOOD256_URL = "http://foodcam.mobi/dataset256.zip"

# Passed as `root=` to torchvision.datasets.Food101, which manages its own
# download URL/checksum and creates `FOOD101_ROOT / "food-101"` itself.
FOOD101_ROOT = DATA_ROOT
FOOD101_DIR = FOOD101_ROOT / "food-101"

YOLO_DATASET_DIR = DATA_ROOT.parent / "yolo" / "uecfood256"
