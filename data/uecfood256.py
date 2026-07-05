"""UECFOOD-256 dataset download, label map, and YOLO-format conversion.

Raw layout (as distributed from http://foodcam.mobi/dataset256.html, and
confirmed directly against the live archive -- see its bundled README.txt)::

    UECFOOD256/
      README.txt
      category.txt        # "<id>\\t<name>" per line, 256 food categories.
                           # NOTE: there is no "plate"/tableware category here
                           # -- every one of the 256 classes is a food dish
                           # (confirmed against the real category.txt: the
                           # closest matches are dish names like "sashimi
                           # bowl"/"beef bowl", not a tableware bbox class).
                           # Plate-area estimation (needed by
                           # estimation/calorie.py) has to be solved
                           # separately; see estimation/plate_detector.py.
      1/                   # one folder per category id, 1..256
        <id>.jpg, ...      # photos containing that food, named by an
                           # arbitrary numeric image id (not sequential per
                           # category, e.g. "67916.jpg"). Per the dataset's
                           # own README: "some photos are duplicated in two
                           # or more directories, since they include two or
                           # more food items."
        bb_info.txt        # header line "img x1 y1 x2 y2", then one row per
                           # food instance:
                           #   <image_id_without_ext> <x1> <y1> <x2> <y2>
                           # x1,y1,x2,y2 are absolute pixel coordinates of the
                           # box's top-left and bottom-right corners. One
                           # image_id can repeat if that photo contains
                           # multiple instances of the same food.
      2/
        ...
      256/
        ...

Conversion to YOLO format:
    The detector's job is only to localize food regions, not identify them
    -- that's the separate Food-101 classifier's job, on a 101-class
    taxonomy that doesn't overlap with UECFOOD-256's 256 categories anyway.
    So every one of UECFOOD-256's original category ids collapses to a
    single YOLO class, `0` ("food"), rather than `category_id - 1` per
    category. For every box:
        x_center = ((x1 + x2) / 2) / image_width
        y_center = ((y1 + y2) / 2) / image_height
        width    = (x2 - x1) / image_width
        height   = (y2 - y1) / image_height
    all normalized to [0, 1], written space-separated as
    `0 x_center y_center width height` (one line per box).

Known caveat: because a multi-food photo is duplicated across category
folders, this converter treats each (category, image) pair as an
independent training image rather than merging duplicates by photo
identity. A multi-food photo therefore appears more than once in the
converted dataset, each copy labeled with only that one category's boxes.
Merging by true photo identity would need content-hash dedup across folders,
which the dataset doesn't support via filename alone, so it's out of scope
here.
"""

import random
import shutil
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from data.remote_zip import open_remote_zip

CATEGORY_FILE = "category.txt"
BBOX_FILE = "bb_info.txt"


@dataclass
class RawBBox:
    image_id: str
    category_id: int
    x1: float
    y1: float
    x2: float
    y2: float


def download_uecfood256(
    dest_dir: Path,
    url: str = "",
    archive_path: Path | None = None,
    category_ids: list[int] | None = None,
    images_per_category: int | None = None,
) -> None:
    """Download and extract UECFOOD-256 into `dest_dir` if not already present.

    The official distribution page can require accepting usage terms before
    download, so `url` may not always be scriptable. If the automated
    download fails (or `url` is left blank), download the archive manually
    from http://foodcam.mobi/dataset256.html and either place it at
    `archive_path` or at the default `dest_dir.parent / "UECFOOD256.zip"`,
    then rerun.

    The full archive is ~4GB. If `category_ids` and/or `images_per_category`
    are given, this instead streams just those entries out of the remote ZIP
    via HTTP range requests (see data/remote_zip.py) -- useful for building
    a small subset (e.g. for `convert_to_yolo_format`'s own `num_classes` /
    `images_per_class`) without downloading the whole thing.
    """
    if (dest_dir / CATEGORY_FILE).exists():
        return  # already extracted

    dest_dir.parent.mkdir(parents=True, exist_ok=True)

    if category_ids is not None or images_per_category is not None:
        if not url:
            raise RuntimeError("Partial download requires an explicit `url=`.")
        _download_subset_via_http_range(dest_dir, url, category_ids, images_per_category)
        return

    if archive_path is None:
        archive_path = dest_dir.parent / "UECFOOD256.zip"

    if not archive_path.exists():
        if not url:
            raise RuntimeError(
                f"No UECFOOD-256 archive found at {archive_path} and no download URL "
                "was given. Download it manually from "
                "http://foodcam.mobi/dataset256.html and place the archive at "
                f"{archive_path}, or pass an explicit url=/archive_path=."
            )
        try:
            urllib.request.urlretrieve(url, archive_path)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to download UECFOOD-256 from {url}: {exc}. Download it "
                "manually from http://foodcam.mobi/dataset256.html and place the "
                f"archive at {archive_path}, then rerun."
            ) from exc

    _extract_archive(archive_path, dest_dir.parent)

    if not (dest_dir / CATEGORY_FILE).exists():
        raise RuntimeError(
            f"Extracted {archive_path} but {dest_dir / CATEGORY_FILE} still doesn't "
            "exist -- the archive's top-level folder name may not be 'UECFOOD256'. "
            "Check the extracted contents and adjust dest_dir accordingly."
        )


def _extract_archive(archive_path: Path, dest_parent: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest_parent)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path) as tf:
            tf.extractall(dest_parent)
    else:
        raise RuntimeError(f"Unrecognized archive format: {archive_path}")


def _zip_root_prefix(names: list[str]) -> str:
    """Return the archive's top-level folder name (e.g. "UECFOOD256/"), if any."""
    for name in names:
        if name.endswith(CATEGORY_FILE):
            return name[: -len(CATEGORY_FILE)]
    return ""


def _download_subset_via_http_range(
    dest_dir: Path,
    url: str,
    category_ids: list[int] | None,
    images_per_category: int | None,
) -> None:
    """Extract only `category_ids` (first `images_per_category` images each) from the
    remote archive, without downloading it in full. Prints progress as it goes --
    this makes ~500 sequential small HTTP range requests, so it can take a few
    minutes with no other visible output otherwise."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"[uecfood256] opening remote zip: {url}", flush=True)
    t0 = time.time()
    with open_remote_zip(url) as zf:
        names = zf.namelist()
        root_prefix = _zip_root_prefix(names)
        name_set = set(names)
        print(
            f"[uecfood256] central directory read in {time.time() - t0:.1f}s "
            f"({len(names)} entries, root={root_prefix!r})",
            flush=True,
        )

        zf.extract(f"{root_prefix}{CATEGORY_FILE}", dest_dir.parent)

        if category_ids is None:
            available = {
                int(name[len(root_prefix) :].split("/")[0])
                for name in names
                if name[len(root_prefix) :].split("/")[0].isdigit()
            }
            category_ids = sorted(available)

        print(f"[uecfood256] fetching {len(category_ids)} categories: {category_ids}", flush=True)

        for cat_num, category_id in enumerate(category_ids, start=1):
            bbox_name = f"{root_prefix}{category_id}/{BBOX_FILE}"
            if bbox_name not in name_set:
                print(f"[uecfood256] ({cat_num}/{len(category_ids)}) category {category_id}: no bb_info.txt, skipping", flush=True)
                continue
            zf.extract(bbox_name, dest_dir.parent)

            image_ids: list[str] = []
            seen: set[str] = set()
            for box in _read_bboxes(dest_dir, category_id):
                if box.image_id not in seen:
                    seen.add(box.image_id)
                    image_ids.append(box.image_id)
            if images_per_category is not None:
                image_ids = image_ids[:images_per_category]

            cat_t0 = time.time()
            print(
                f"[uecfood256] ({cat_num}/{len(category_ids)}) category {category_id}: "
                f"downloading {len(image_ids)} images",
                flush=True,
            )
            for img_num, image_id in enumerate(image_ids, start=1):
                image_name = f"{root_prefix}{category_id}/{image_id}.jpg"
                if image_name in name_set:
                    zf.extract(image_name, dest_dir.parent)
                if img_num % 10 == 0 or img_num == len(image_ids):
                    print(
                        f"[uecfood256]   category {category_id}: {img_num}/{len(image_ids)} images "
                        f"({time.time() - cat_t0:.1f}s elapsed)",
                        flush=True,
                    )

    print(f"[uecfood256] done in {time.time() - t0:.1f}s total", flush=True)


def load_category_map(root_dir: Path) -> dict[int, str]:
    """Parse UECFOOD-256's `category.txt` into a {category_id: name} map."""
    category_map: dict[int, str] = {}
    with open(root_dir / CATEGORY_FILE, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t") if "\t" in line else line.split(None, 1)
            if len(parts) != 2:
                continue
            try:
                category_id = int(parts[0])
            except ValueError:
                continue  # header row, e.g. "id  name"
            category_map[category_id] = parts[1].strip()
    return category_map


def _read_bboxes(root_dir: Path, category_id: int) -> list[RawBBox]:
    """Parse one category folder's bb_info.txt."""
    bb_path = root_dir / str(category_id) / BBOX_FILE
    boxes: list[RawBBox] = []
    with open(bb_path, encoding="utf-8") as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header "img x1 y1 x2 y2"
        parts = line.split()
        if len(parts) != 5:
            continue
        image_id, x1, y1, x2, y2 = parts
        boxes.append(RawBBox(image_id, category_id, float(x1), float(y1), float(x2), float(y2)))
    return boxes


def convert_to_yolo_format(
    src_dir: Path,
    dest_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_classes: int | None = None,
    images_per_class: int | None = None,
) -> None:
    """Convert UECFOOD-256 bbox annotations into YOLOv8's expected layout.

    Writes `images/{train,val}/`, `labels/{train,val}/*.txt`, and a
    single-class `data.yaml` (`nc: 1`, `names: {0: food}`) under `dest_dir`
    -- see the module docstring for why every original category collapses
    to one YOLO class.

    `num_classes` / `images_per_class` still cap which *source* categories
    / images get sampled (the first N category ids, first M images per
    category) so a full training run can be smoke-tested quickly instead of
    processing all ~31k images -- they no longer affect the number of
    output classes, since there's only ever one.
    """
    category_map = load_category_map(src_dir)
    category_ids = sorted(category_map)
    if num_classes is not None:
        category_ids = category_ids[:num_classes]

    for split in ("train", "val"):
        (dest_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    for category_id in category_ids:
        boxes_by_image: dict[str, list[RawBBox]] = {}
        for box in _read_bboxes(src_dir, category_id):
            boxes_by_image.setdefault(box.image_id, []).append(box)

        image_ids = sorted(boxes_by_image, key=lambda i: (int(i) if i.isdigit() else 0, i))
        if images_per_class is not None:
            image_ids = image_ids[:images_per_class]

        image_ids = list(image_ids)
        rng.shuffle(image_ids)
        num_val = max(1, round(len(image_ids) * val_ratio)) if len(image_ids) > 1 else 0
        val_ids = set(image_ids[:num_val])

        for image_id in image_ids:
            split = "val" if image_id in val_ids else "train"
            src_image = src_dir / str(category_id) / f"{image_id}.jpg"
            if not src_image.exists():
                continue

            dest_stem = f"{category_id}_{image_id}"
            dest_image = dest_dir / "images" / split / f"{dest_stem}.jpg"
            shutil.copyfile(src_image, dest_image)

            with Image.open(src_image) as img:
                width, height = img.size

            label_lines = []
            for box in boxes_by_image[image_id]:
                x_center = ((box.x1 + box.x2) / 2) / width
                y_center = ((box.y1 + box.y2) / 2) / height
                w = (box.x2 - box.x1) / width
                h = (box.y2 - box.y1) / height
                label_lines.append(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

            label_path = dest_dir / "labels" / split / f"{dest_stem}.txt"
            label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    _write_data_yaml(dest_dir)


def _write_data_yaml(dest_dir: Path) -> None:
    """Write the single-class ("food") data.yaml Ultralytics expects."""
    content = (
        f"path: {dest_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names:\n"
        "  0: food\n"
    )
    (dest_dir / "data.yaml").write_text(content, encoding="utf-8")


class UECFood256Dataset(Dataset):
    """Reads an already-YOLO-converted UECFOOD-256 split (see `convert_to_yolo_format`).

    Returns raw PIL images and their normalized YOLO-format boxes. This is
    meant for inspection/evaluation -- Ultralytics trains directly off
    `images/` + `labels/` + `data.yaml`, it doesn't need a torch Dataset.
    """

    def __init__(self, root_dir: Path, split: str = "train") -> None:
        self.image_dir = Path(root_dir) / "images" / split
        self.label_dir = Path(root_dir) / "labels" / split
        self.image_paths = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]]:
        image_path = self.image_paths[index]
        label_path = self.label_dir / f"{image_path.stem}.txt"

        image = Image.open(image_path).convert("RGB")
        boxes: list[tuple[int, float, float, float, float]] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id, x, y, w, h = line.split()
                boxes.append((int(class_id), float(x), float(y), float(w), float(h)))

        return image, boxes
