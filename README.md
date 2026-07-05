# PlateVision

Multi-Food Detection and Portion-Aware Calorie Estimation System — CS5330 Group 3

Given a photo of a plate of food, PlateVision detects each individual food
item on it, classifies what kind of dish it is, and estimates its calories
based on how much of the plate it covers. The goal is a lightweight,
end-to-end computer-vision pipeline that goes from a single photo straight
to a per-item, itemized calorie estimate — no manual food logging.

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Environment setup](#environment-setup)
- [How to run](#how-to-run)
- [Module overview](#module-overview)
- [Known issues and limitations](#known-issues-and-limitations)
- [Evaluation results](#evaluation-results)
- [Datasets and citations](#datasets-and-citations)

## What it does

1. **Detect** where the food items are on the plate.
2. **Classify** what each detected item actually is.
3. **Estimate** its calories from a USDA reference value, scaled by how
   much of the plate that item's bounding box covers.
4. **Visualize** all of that in a web demo: annotated photo, one card per
   food item, and a total calorie count for the plate.

## Architecture

```
                 ┌─────────────────────┐
   Plate photo   │   YOLOv8 Detector    │  fine-tuned on UECFOOD-256,
  ──────────────►│  (find food regions) │  single class: "food" --
                 └──────────┬───────────┘  it localizes, it doesn't identify
                            │ food bounding boxes
                            ▼
                 ┌─────────────────────┐
                 │   Plate Localizer    │  classical CV, no training:
                 │  (find plate region) │  Hough circle -> largest round
                 └──────────┬───────────┘  contour -> full image fallback
                            │ plate bounding box
                            ▼
        for each detected food region:
        ┌───────────────────────────────────────────┐
        │  crop  ──►  ResNet-50 / EfficientNet-B4     │  fine-tuned on
        │             classifier (101 Food-101 classes)│  Food-101
        │                       │                      │
        │                       ▼                      │
        │        USDA FoodData Central lookup          │  Base_Cal,
        │            (cached locally as JSON)           │  kcal per 100g
        │                       │                      │
        │                       ▼                      │
        │   Calories = Base_Cal x (food_bbox_area       │
        │                          / plate_bbox_area)   │
        └───────────────────────┬─────────────────────┘
                                 ▼
                     ┌─────────────────────┐
                     │    Streamlit app     │  annotated image,
                     │    (visualization)   │  per-item cards, total kcal
                     └─────────────────────┘
```

The detector and classifier are deliberately decoupled: the detector's own
class predictions are discarded and only its bounding boxes are used as
region proposals (it was fine-tuned on UECFOOD-256's food categories, a
completely different taxonomy from the classifier's Food-101 classes).
Food identity comes entirely from the classifier. This is a standard
detect-then-classify split, not a workaround — see
[Known issues](#known-issues-and-limitations) for what it costs.

## Environment setup

**Requirements:** Python 3.11+ (developed and tested on 3.11.9). A CUDA GPU
is strongly recommended for training (this project was developed on an
RTX 5090) but not required for running the Streamlit demo against
pre-trained checkpoints on CPU.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` installs a CPU/generic `torch`/`torchvision` build. If
you have an NVIDIA GPU and want CUDA acceleration, install the matching
CUDA build from the [PyTorch install selector](https://pytorch.org/get-started/locally/)
*instead of* the plain `torch`/`torchvision` lines (uninstall those first
if you already ran the command above).

### USDA FoodData Central API key

Calorie lookups call the real USDA FoodData Central API. You need your own
free key:

1. Sign up at <https://fdc.nal.usda.gov/api-key-signup.html> (instant, no
   approval wait).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Put your key in `.env`:
   ```
   FDC_API_KEY=your_own_key_here
   ```

`.env` is gitignored — **never commit your real key**. If you skip this
step entirely, the app still works by falling back to api.data.gov's
shared `DEMO_KEY`, but that key is rate-limited (30 requests/hour, 50/day)
and shared across every project using it, so it's easy to get temporarily
locked out. Get your own key before relying on this for anything beyond a
quick look.

## How to run

All commands assume your shell's working directory is the repo root.

### 1. Get the datasets

UECFOOD-256 (detector) and Food-101 (classifier) aren't committed to the
repo (see `.gitignore`) — download them once:

```python
from pathlib import Path
from data.config import UECFOOD256_DIR, UECFOOD256_URL, FOOD101_ROOT, YOLO_DATASET_DIR
from data.uecfood256 import download_uecfood256, convert_to_yolo_format
from data.food101 import download_food101

# UECFOOD-256: ~4GB. Pass category_ids/images_per_category to only stream a
# subset via HTTP range requests instead of the full archive (see
# data/uecfood256.py's docstring for why this dataset supports that).
download_uecfood256(UECFOOD256_DIR, url=UECFOOD256_URL,
                     category_ids=list(range(1, 11)), images_per_category=50)
convert_to_yolo_format(UECFOOD256_DIR, YOLO_DATASET_DIR,
                        num_classes=10, images_per_class=50)

# Food-101: ~5GB, single tar.gz, no partial-download option.
download_food101(FOOD101_ROOT)
```

### 2. Train the detector

```bash
python -m training.train_detector \
  --data-yaml data/yolo/uecfood256/data.yaml \
  --epochs 10 --batch-size 16 --image-size 320 \
  --device 0 --run-name my_run
```

`--device` follows Ultralytics' convention here (`"0"` for GPU 0, `"cpu"`
for CPU). Checkpoints land in `runs/detect/my_run/weights/{best,last}.pt`.

### 3. Train a classifier

```bash
# ResNet-50 @ 224x224
python -m training.train_classifier --backbone resnet50 \
  --data-dir data/raw --image-size 224 \
  --head-epochs 3 --finetune-epochs 5 \
  --device cuda:0 --workers 8 --run-name my_run

# EfficientNet-B4 @ 380x380 (its native training resolution)
python -m training.train_classifier --backbone efficientnet_b4 \
  --data-dir data/raw --image-size 380 \
  --head-epochs 3 --finetune-epochs 5 \
  --device cuda:0 --workers 8 --run-name my_run
```

Note `--device` here takes a *PyTorch* device string (`"cuda:0"`, not
Ultralytics' `"0"`) — the two training scripts intentionally follow each
underlying framework's own convention rather than inventing a third one.
Checkpoints land in `runs/classify/my_run_{resnet50,efficientnet_b4}.pt`.

On Windows, if you call `train_detector()`/`train_classifier()` directly
from your own script (rather than through the `python -m ...` CLIs above)
with `workers > 0`, guard your script's top-level code with
`if __name__ == "__main__":` — DataLoader workers are spawned as new
processes on Windows, and without the guard they re-import and re-execute
your launching script, crashing on startup.

### 4. Evaluate

```bash
# Detection: precision/recall/mAP50/mAP50-95 via Ultralytics' own model.val()
python -m evaluation.detection_metrics \
  --weights runs/detect/my_run/weights/best.pt \
  --data-yaml data/yolo/uecfood256/data.yaml --device 0

# Classification: Top-1/Top-5 over the full Food-101 val split
python -m evaluation.classification_metrics \
  --checkpoints runs/classify/my_run_resnet50.pt runs/classify/my_run_efficientnet_b4.pt \
  --data-dir data/raw --device cuda:0

# Calorie lookup sanity check (MAE/MAPE against a small reference set)
python evaluation/calorie_metrics.py
```

### 5. Run the Streamlit demo

```bash
streamlit run app/streamlit_app.py
```

Opens at `http://localhost:8501`. Upload a plate photo, pick a classifier
in the sidebar, and it runs the full pipeline live. By default it looks
for checkpoints at `runs/detect/smoke_test_food_only/weights/best.pt` and
`runs/classify/full101_{resnet50,efficientnet_b4}.pt` — update the paths
at the top of `app/streamlit_app.py` if you trained under different
`--run-name`s.

### Run the tests

```bash
pytest tests/ -v
```

## Module overview

- **`data/`** — dataset download and preprocessing. `uecfood256.py`
  downloads UECFOOD-256 (supports partial/subset downloads via HTTP range
  requests) and converts its bounding-box annotations into YOLO format as a
  single "food" class. `food101.py` wraps `torchvision.datasets.Food101`
  with a stratified train/val split and configurable image size.
  `remote_zip.py` is the HTTP-range reader that makes partial dataset
  downloads possible. `config.py` centralizes dataset paths.
- **`models/`** — model architecture wrappers and checkpoint I/O.
  `detector.py` wraps Ultralytics YOLOv8. `resnet50.py`/`efficientnet_b4.py`
  build each backbone from ImageNet weights with a replaced classification
  head. `classifier.py` holds the shared `FoodClassifier` interface and the
  freeze/unfreeze helpers used by the two-phase training strategy.
  `checkpoint.py` is a thin save/load wrapper.
- **`training/`** — training entry points. `train_detector.py` fine-tunes
  YOLOv8 on the converted UECFOOD-256 data. `train_classifier.py` runs the
  two-phase strategy (frozen-backbone head training, then full-network
  fine-tuning at a lower LR) for either classifier backbone.
- **`inference/`** — `pipeline.py`'s `PlateVisionPipeline` chains detection,
  plate localization, classification, and calorie estimation into one
  `run(image_path)` call; `schema.py` defines the `PlateResult`/
  `FoodItemResult` result types.
- **`estimation/`** — the calorie math and the USDA integration.
  `calorie.py` implements `Calories = Base_Cal * (food_area / plate_area)`.
  `fdc_client.py` queries USDA FoodData Central for Base_Cal values.
  `cache.py` persists those lookups to disk. `plate_detector.py` is the
  classical-CV plate localizer (Hough circle -> contour -> full-image
  fallback).
- **`app/`** — the Streamlit demo (`streamlit_app.py`) and its rendering
  helpers (`components.py`: bounding-box overlay, per-item cards, totals).
- **`evaluation/`** — standalone evaluation scripts, decoupled from
  training: `detection_metrics.py` (Ultralytics `model.val()`),
  `classification_metrics.py` (Top-1/Top-5 over the full val set),
  `calorie_metrics.py` (MAE/MAPE against a small reference set).
- **`tests/`** — unit tests, mirroring the modules above. Network calls
  (USDA API) are mocked in the automated suite; the real API was verified
  by hand separately (see `CLAUDE.md`).

## Known issues and limitations

Being upfront about what this is and isn't:

- **The detector was fine-tuned on a small subset, not the full dataset.**
  Training and evaluation here use 10 of UECFOOD-256's 256 categories
  (50 images each, 500 images total) — enough to validate that the
  pipeline and training strategy work end-to-end, not a full-scale result.
  Reproducing a result representative of the whole dataset means retraining
  on all 256 categories' images (though note the detector is single-class
  regardless of how many source categories feed it — see the architecture
  note above on why category count doesn't map to detector output classes).

- **Plate localization is classical computer vision, not a trained model.**
  There's no plate/tableware annotation in UECFOOD-256 to train a detector
  head on, so `estimation/plate_detector.py` finds the plate with a Hough
  circle transform, falling back to the largest sufficiently round contour,
  and finally to treating the whole photo as the plate if neither finds
  anything. That last fallback is a real degradation: calorie estimates
  computed against "the whole image" as the portion-size denominator are
  less meaningful than ones computed against an actual plate boundary.
  `PlateResult.plate_detection_method` reports which tier fired, so this is
  at least visible rather than silent, but it isn't fixed by having more
  training data — it would need a real trained plate/tableware segmentation
  model.

- **The USDA calorie lookup takes the first search match, which isn't
  always the right one.** `estimation/fdc_client.py` searches USDA
  FoodData Central and uses the top result's Energy value. That can land on
  a record with a different preparation or portion than what was actually
  detected. Two examples from this project's own evaluation
  (`evaluation/calorie_metrics.py`): searching "apple pie" matched *"Pie
  fillings, apple, canned"* (100 kcal/100g) rather than a baked/prepared
  pie (~237 kcal/100g in USDA's own records) — both are real entries for
  the same query, and picking between them isn't something a single
  top-result lookup can do. Searching "paella" similarly matched a record
  reporting 422 kcal/100g against a commonly-cited reference of roughly
  158 kcal/100g. Fixing this would need either a curated food-name-to-fdcId
  mapping or a smarter re-ranking/disambiguation step.

- **Classifier accuracy has room to grow.** Top-1 accuracy across the full
  101 Food-101 classes is 78.3% (ResNet-50 @ 224x224) and 83.9%
  (EfficientNet-B4 @ 380x380) — solid for 8 epochs of fine-tuning, but well
  short of published Food-101 state-of-the-art results, which typically
  involve longer training schedules, stronger augmentation, and/or larger
  backbones.

## Evaluation results

**Detection** (food-only YOLOv8, UECFOOD-256 val split, 10-category/50-image
subset — see limitations above):

| Metric | Value |
|---|---|
| Precision | 0.932 |
| Recall | 0.980 |
| mAP50 | 0.962 |
| mAP50-95 | 0.745 |

**Classification** (Food-101, full 101 classes, full 7575-image val split):

| Backbone | Image size | Top-1 | Top-5 |
|---|---|---|---|
| ResNet-50 | 224x224 | 78.34% | 93.60% |
| EfficientNet-B4 | 380x380 | **83.88%** | **95.80%** |

**Calorie lookup** (7-sample sanity check against commonly-cited reference
values — see limitations above for why this is illustrative, not rigorous):

| Metric | Value |
|---|---|
| MAE | 82.1 kcal/100g |
| MAPE | 45.1% |

## Datasets and citations

- **UECFOOD-256** — Y. Kawano and K. Yanai, "Automatic Expansion of a Food
  Image Dataset Leveraging Existing Categories with Domain Adaptation,"
  *Proc. of ECCV Workshop on Transferring and Adapting Source Knowledge in
  Computer Vision (TASK-CV)*, 2014.
  <http://foodcam.mobi/dataset256.html>
- **Food-101** — L. Bossard, M. Guillaumin, and L. Van Gool, "Food-101 —
  Mining Discriminative Components with Random Forests," *European
  Conference on Computer Vision (ECCV)*, 2014.
  <https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/>
- **USDA FoodData Central** — U.S. Department of Agriculture, Agricultural
  Research Service. FoodData Central. <https://fdc.nal.usda.gov/>
