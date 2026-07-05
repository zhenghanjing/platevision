# PlateVision

Computer vision course project (CS5330 Group 3): given a photo of a plate of
food, detect the individual food items on it, classify each item's food type,
and estimate the calories of each item.

## Pipeline

1. **Detection** — YOLOv8 (Ultralytics), fine-tuned on **UECFOOD-256** to
   produce bounding boxes for food items on the plate (plus a plate bbox).
2. **Classification** — each detected food crop is classified by one of two
   candidate backbones fine-tuned on **Food-101**, compared against each
   other:
   - ResNet-50
   - EfficientNet-B4
3. **Calorie estimation** — portion-aware estimate using bbox area as a proxy
   for portion size relative to the plate:

   ```
   Calories = Base_Cal × (food_bbox_area / plate_bbox_area)
   ```

   `Base_Cal` (calories for a reference/full portion of that food class) is
   looked up from the **USDA FoodData Central API**.
4. **Frontend** — a Streamlit web demo that lets a user upload a plate photo
   and see detected items, classified labels, and estimated calories.

## Repository layout

- `data/` — dataset download/prep scripts and dataset config (UECFOOD-256,
  Food-101), plus local data caching. Raw/processed data itself is not
  committed.
- `models/` — model architecture wrappers and checkpoint I/O for the YOLOv8
  detector and the ResNet-50 / EfficientNet-B4 classifiers.
- `training/` — training entry points and training-loop logic for the
  detector and each classifier.
- `inference/` — runtime pipeline that chains detection → classification →
  calorie estimation for a single input image.
- `estimation/` — calorie math (bbox-ratio formula) and the USDA FoodData
  Central API client/cache for `Base_Cal` lookups.
- `app/` — Streamlit demo app.
- `evaluation/` — metrics and evaluation scripts (detection mAP,
  classification accuracy for the ResNet-50 vs EfficientNet-B4 comparison,
  calorie estimation error).
- `tests/` — unit tests mirroring the module layout above.

## Conventions

- Python, type-hinted function signatures throughout.
- Most modules are still scaffolded with signatures/docstrings and
  `raise NotImplementedError` bodies — fill in real logic incrementally.
  Fully implemented so far: `data/food101.py`, `data/uecfood256.py`,
  `data/remote_zip.py`, `estimation/plate_detector.py`, `estimation/calorie.py`,
  `estimation/fdc_client.py`, `estimation/cache.py`, `models/detector.py`
  (YOLOv8Detector), `models/checkpoint.py`, `models/classifier.py`
  (freeze/unfreeze helpers), `models/resnet50.py`, `models/efficientnet_b4.py`,
  `training/train_detector.py`, `training/train_classifier.py`,
  `inference/pipeline.py` (`PlateVisionPipeline`), `inference/schema.py`,
  `evaluation/detection_metrics.py`, `evaluation/classification_metrics.py`,
  `evaluation/calorie_metrics.py`, `app/streamlit_app.py`, `app/components.py`.
  Nothing left stubbed at this point.
- Keep dataset-specific label maps (UECFOOD-256 category list, Food-101
  category list) in `data/`, not hardcoded elsewhere.
- USDA FoodData Central API key should be read from an environment variable
  (e.g. `FDC_API_KEY`), never hardcoded.
- On Windows, any script that calls `train_detector`/`train_classifier`/
  `model.train()` with `workers > 0` MUST guard its top-level code with
  `if __name__ == "__main__":` -- DataLoader workers are spawned (not
  forked), and without the guard they re-import and re-execute the launching
  script, crashing with `RuntimeError: An attempt has been made to start a
  new process before the current process has finished its bootstrapping
  phase`. `training/train_detector.py`/`train_classifier.py`'s own `main()`
  are already guarded; this only bites ad-hoc scripts that call
  `train_detector()`/`train_classifier()` directly. Both configs default
  `workers` to a safe value already (`DetectorTrainConfig.workers=8` since
  its own CLI is guarded; `ClassifierTrainConfig.workers=0` since it's more
  often called from ad-hoc scripts) -- override deliberately, not by habit.
- Device string convention differs between the two training paths:
  `DetectorTrainConfig.device` goes through Ultralytics, which accepts short
  forms like `"0"`/`"cpu"`. `ClassifierTrainConfig.device` goes straight into
  `torch.Tensor.to()`/`nn.Module.to()`, which needs `"cuda:0"`/`"cuda"`/`"cpu"`
  -- `"0"` raises `RuntimeError: Invalid device string`.

## Dataset notes

- **Food-101** (`data/food101.py`) wraps `torchvision.datasets.Food101`
  directly rather than re-parsing the archive. torchvision only defines
  "train"/"test"; a "val" split is carved out of "train" via a deterministic
  per-class stratified split (`val_ratio`, fixed `seed`).
- **UECFOOD-256** (`data/uecfood256.py`) has its own download + YOLO-format
  conversion pipeline (see that module's docstring for the exact raw layout
  and conversion formulas). `convert_to_yolo_format()` supports
  `num_classes`/`images_per_class` to build a small subset for fast
  pipeline smoke tests.
- **The YOLO conversion is single-class on purpose.** Every one of
  UECFOOD-256's 256 original categories collapses to one YOLO class, `0`
  ("food") -- `num_classes`/`images_per_class` still control which *source*
  categories/images get sampled, they just no longer become separate output
  classes. Reason: the detector and the Food-101 classifier are trained on
  two disjoint taxonomies (256 UECFOOD-256 dishes vs. 101 different,
  mostly-Western Food-101 dishes) that were never going to agree on names,
  so the detector's role is only to localize food regions -- identity is
  entirely the classifier's job (see "Pipeline notes" below). An earlier
  version kept the original 256-way (well, N-way per smoke test) classes and
  ran a full pipeline test where the detector's own class guesses were
  compared against the classifier's -- that comparison was never meaningful
  and the mismatch looked like a bug. It wasn't; it just needed the class
  head removed from the detector's job. Re-running the detector smoke test
  single-class (10 source categories, same 400/100 split, same 10 epochs)
  produced a *cleaner* result than the original N-way run: mAP50 0.961 /
  mAP50-95 0.744 (was 0.776 / 0.629), because localizing "is there food
  here" is a strictly easier task than also discriminating between
  visually-similar dishes. It also means each food region only gets
  detected once now, instead of competing overlapping boxes from different
  class hypotheses on the same region.
- **No "plate" category exists in UECFOOD-256.** All 256 categories are food
  dish names (rice, curry, sushi, ramen, ...) — there is no tableware/plate
  annotation to use as the area denominator in
  `Calories = Base_Cal × (food_bbox_area / plate_bbox_area)`. Resolved via
  `estimation/plate_detector.py`: a classical-CV fallback (Hough circle
  transform, then largest round contour, then full-image bbox as a last
  resort) run directly on the image rather than trained into the YOLO
  detector.

## Classifier training notes

- Both `models/resnet50.py` and `models/efficientnet_b4.py` load ImageNet
  weights via `torchvision.models` and replace the final head
  (`fc` for ResNet-50; `classifier[1]` for EfficientNet-B4, exposed as each
  module's `HEAD_PREFIX` constant) for Food-101's class count.
- `training/train_classifier.py` runs the two-phase strategy from the
  proposal: freeze everything except the head for `head_epochs`, then
  `unfreeze_all()` and fine-tune the whole network for `finetune_epochs` at
  a smaller LR (`ClassifierTrainConfig.head_lr` / `finetune_lr`).
  `ClassifierTrainConfig.num_classes` subsets Food-101 the same way
  UECFOOD-256's converter does, for fast smoke tests.
- Checkpoints are plain dicts (`model_state_dict`, `class_names`,
  `backbone`) via `models/checkpoint.py` -- `class_names` travels with the
  weights so `ResNet50Classifier.load()`/`EfficientNetB4Classifier.load()`
  can map predicted indices back to names without a separate label file.
- `ClassifierTrainConfig.image_size` (default 224) flows through
  `Food101Dataset`/`build_transforms()` into training, and is saved into the
  checkpoint so `ResNet50Classifier.load()`/`EfficientNetB4Classifier.load()`
  reconstruct the *same* resize/crop pipeline the model was trained with --
  checkpoints predating this field fall back to 224 via `checkpoint.get(...)`.
- Smoke test (15 classes, 3 head + 5 finetune epochs, RTX 5090):
  - ResNet-50 @ 224x224 (~7min): **86.0%** val accuracy; train acc climbed to
    98% by the last fine-tune epoch (mild overfitting expected on this small
    a class subset, val_loss ticked back up slightly in later epochs).
  - EfficientNet-B4 @ 224x224 (~7min): 82.0% val accuracy, smaller train/val
    gap than ResNet-50, but behind it -- expected, since EfficientNet-B4 is
    normally trained at a larger native resolution.
  - EfficientNet-B4 @ 380x380 (~14min, same everything else): **87.7%** val
    accuracy -- passes ResNet-50's 224x224 result once given its intended
    resolution, with less overfitting (train/val gap ~4pts vs ResNet-50's
    ~12pts). Confirms the 224x224 comparison was an apples-to-oranges
    handicap for EfficientNet-B4, not a real architecture disadvantage.
  - None of this is a final verdict at only 15/101 classes and 8 epochs --
    it validates the pipeline and the two-phase strategy, and gives a
    resolution-matched basis to extend to all 101 classes for the real
    report numbers.
- **DataLoader `workers=0` was a real bottleneck, not just a Windows-spawn
  safety default.** Measured on the full 101-class dataset (single head
  epoch, batch_size=32): `workers=0` took 286.5s/epoch (ResNet-50 @ 224) and
  605.5s/epoch (EfficientNet-B4 @ 380); `workers=8` cut both to ~84s and
  ~182s/epoch respectively (~3.3-3.4x speedup) -- JPEG decode + resize was
  CPU-bound and serialized on the main process the whole time, GPU was
  underused. Every ad-hoc training script in this project now has the
  `if __name__ == "__main__":` guard (see the Windows-spawn note above), so
  `workers=8`+ is safe to use by default going forward, not just `workers=0`.
- **Full 101-class training** (3 head + 5 finetune epochs, `workers=8`,
  RTX 5090, checkpoints `runs/classify/full101_{resnet50,efficientnet_b4}.pt`):
  - ResNet-50 @ 224x224: **78.3%** val accuracy in ~14.3 min (epoch times
    ~84s head / ~121s finetune -- finetune's full backward pass through an
    unfrozen backbone costs meaningfully more once `workers=8` removed the
    data-loading bottleneck that used to mask it).
  - EfficientNet-B4 @ 380x380: **83.9%** val accuracy in ~41.8 min (epoch
    times ~184s head / ~391s finetune -- a much bigger head/finetune gap
    than ResNet-50's, again because finetune's cost is now GPU-compute-bound
    rather than data-loading-bound, and EfficientNet-B4 at 380x380 has a lot
    more compute per batch). This came in over even the "conservative"
    pre-run estimate of ~41 min combined for *both* backbones -- the
    head-epoch timing alone undersold how much slower unfrozen fine-tuning
    at this resolution would be once the bottleneck moved to the GPU.
  - Both accuracies dropped somewhat vs. the 15-class smoke test (as
    expected: 101-way is a harder discrimination problem), but the
    *ordering* held: EfficientNet-B4 @ 380x380 still beats ResNet-50 @
    224x224 (83.9% vs 78.3%), consistent with the smoke test's finding that
    224x224 undersells EfficientNet-B4.

## Calorie estimation notes

- `estimation/calorie.py::estimate_calories()` raises `ValueError` on a
  non-positive `plate_bbox` area (undefined ratio, not just "small") rather
  than returning inf/nan; a zero-area `food_bbox` naturally yields 0
  calories; a `food_bbox` bigger than `plate_bbox` (ratio > 1) is left
  unclamped on purpose -- it's a real signal of a bad detection, and it's on
  the caller to decide how to react, not this function to silently mask it.
- `estimation/fdc_client.py::FoodDataCentralClient` hits the real USDA
  FoodData Central `/foods/search` endpoint (verified live, not just
  mocked). Base_Cal is the "Energy" nutrient (USDA nutrient id 1008, kcal)
  from the best match, preferring `Foundation`/`SR Legacy` data types (no
  branded-product noise) before falling back to an unrestricted search.
  That's kcal **per 100g**, not literally "one serving" -- consistent with
  Base_Cal being a reference amount scaled by the food/plate area ratio.
- Falls back to api.data.gov's shared `DEMO_KEY` when `FDC_API_KEY` is
  unset (rate-limited but real; get a personal key for actual use).
  `BaseCalCache` (`estimation/cache.py`) persists lookups to a JSON file
  (default `estimation/cache/base_calories.json`) so repeat lookups for the
  same food name skip the network entirely -- verified live: e.g.
  apple_pie=100, sushi=94, pizza=280, fried_rice=174 kcal/100g, all cache
  hits (0ms) on the second call.
- A personal `FDC_API_KEY` is stored in a `.env` file at the repo root
  (gitignored, confirmed via `git check-ignore`; never hardcoded in source).
  `estimation/fdc_client.py` calls `load_dotenv()` on import (pointed at
  the repo-root `.env` explicitly, not cwd-relative) before reading
  `os.environ["FDC_API_KEY"]`, so a personal key just needs to exist in
  that file -- no manual `export`/`$env:` needed each session. With a real
  key, the earlier `DEMO_KEY` 429 rate-limit failures on `carrot_cake` /
  `beet_salad` / `chocolate_mousse` / `lasagna` / `fried_rice` / `paella`
  all resolved cleanly on retry (all now cached).
- `tests/test_fdc_client.py` mocks `requests.get` (network calls don't
  belong in the automated suite) but the module itself was verified against
  the live API by hand, not just against mocks.

## Pipeline notes (`inference/pipeline.py`)

- `PlateVisionPipeline` uses the detector *only* for region proposals --
  each detected bbox is cropped and reclassified by the Food-101 classifier;
  `Detection.class_name` is now always literally `"food"` (see "Dataset
  notes" above -- the detector is single-class on purpose) and isn't treated
  as an identity guess at all, just confirmation that *something* was
  localized. Food identity comes entirely from the classifier.
- Plate localization edge cases were resolved as fallback tiers inside
  `estimation/plate_detector.py` itself (see that module), not as pipeline
  branching: no plate boundary found -> full image is used as the plate
  bbox (better than skipping calorie estimation outright), and multiple
  candidate circles -> the largest one wins (the plate is expected to be the
  biggest round object in frame). `PlateResult.plate_detection_method`
  carries which tier fired, so callers can flag/discount a `"full_image"`
  result as a low-confidence estimate rather than being unable to tell.
- A failed Base_Cal lookup for one food item (no USDA match, network error)
  returns `estimated_calories=None` for just that item rather than raising
  out of `run()` -- `PlateResult.total_calories()` skips `None` entries when
  summing, so one bad food name doesn't zero out or crash the whole plate.
- `PlateVisionPipeline.from_checkpoints(...)` is the convenient entry point:
  pass `classifier_backbone="resnet50"` or `"efficientnet_b4"` plus the
  matching checkpoint path to pick which trained classifier to use (e.g.
  the 224x224 ResNet-50 checkpoint, currently the best all-around one, or
  the 380x380 EfficientNet-B4 one). It also normalizes a single `device`
  string for both Ultralytics and raw-PyTorch conventions internally
  (`_to_ultralytics_device` / `_to_torch_device`) so callers don't have to
  remember the `"0"` vs `"cuda:0"` split documented above.
- End-to-end smoke test on 4 real UECFOOD-256 validation photos (food-only
  detector + ResNet-50 224x224 classifier) ran clean: detection, plate
  localization (`hough_circle` on 3/4 photos, `full_image` on the one where
  Hough circle found nothing usable), cropping, classification, and USDA
  calorie lookups all worked with no crashes. Each image now yields exactly
  one food detection instead of the 2 overlapping ones the old per-category
  detector produced (different UECFOOD-256 classes no longer compete for
  the same region). Two of the four calorie lookups (`carrot_cake`,
  `beet_salad`) hit the shared `DEMO_KEY`'s rate limit (`429 Too Many
  Requests`) after a session's worth of testing -- a real, unmocked
  demonstration that `_estimate_calories()`'s per-item try/except works as
  designed: those two items got `estimated_calories=None` and
  `total_calories()` correctly excluded them rather than the run failing or
  a bad value getting cached (confirmed: neither name ended up in
  `base_calories.json`). Get a personal `FDC_API_KEY` before relying on
  this for anything beyond occasional testing.
- Reran the same 4-image smoke test after swapping in the full-101-class
  ResNet-50 checkpoint (`full101_resnet50.pt`) and the single-class
  ("food") detector: classification content is now dramatically more
  plausible than with the 15-class checkpoint -- one exact match
  (`fried_rice` at 93.9% confidence on the actual UECFOOD-256 "fried rice"
  photo, since Food-101 happens to include that class too), and the misses
  are now semantically reasonable (`lasagna` for beef curry, `paella` for a
  tempura-on-rice bowl) rather than nonsensical (previously `apple_pie` for
  everything). This directly confirms the fix: classification quality was
  bottlenecked by the classifier's tiny 15-class vocabulary, not by
  anything in the pipeline's plumbing. All 4 calorie lookups failed this
  run (`DEMO_KEY` fully rate-limited after a session's worth of testing);
  none got cached, so a real `FDC_API_KEY` will cleanly re-attempt all four
  on the next run.

## Evaluation results (`evaluation/`)

- `detection_metrics.py` calls Ultralytics' own `model.val()` rather than
  reimplementing IoU matching/mAP aggregation. Formal report on the
  food-only detector (`smoke_test_food_only/weights/best.pt`, UECFOOD-256
  val split): **precision 0.932, recall 0.980, mAP50 0.962, mAP50-95
  0.745** -- matches the training run's own final-epoch numbers almost
  exactly, confirming the standalone eval reproduces training-time
  validation rather than measuring something different.
- `classification_metrics.py` runs the raw `nn.Module` batched via
  DataLoader (not `FoodClassifier.predict()`'s one-image-at-a-time API,
  which would be far too slow over thousands of images) to get Top-1 *and*
  Top-5 accuracy over the full 7575-image Food-101 val split -- Top-5 is a
  genuinely new metric, never computed during training. Results:

  | Backbone | Image size | Top-1 | Top-5 |
  |---|---|---|---|
  | ResNet-50 | 224 | 78.34% | 93.60% |
  | EfficientNet-B4 | 380 | 83.88% | 95.80% |

  ResNet-50's Top-1 (78.34%) exactly matches its training run's logged
  final val_acc -- a good sanity check that this is the same held-out set,
  evaluated correctly.
- `calorie_metrics.py` compares the USDA-lookup cache's Base_Cal values
  against `REFERENCE_KCAL_PER_100G`, a small hand-picked set of commonly
  cited nutrition figures for whichever foods happen to already be in
  `estimation/cache/base_calories.json` (currently 7). This is explicitly
  a small illustrative sanity check, not a rigorous benchmark -- see the
  module docstring for why (most reference figures are typical/approximate,
  not independently re-verified against a second database). Result: **MAE
  82.1 kcal/100g, MAPE 45.1%**, dominated by one bad outlier (`paella`:
  predicted 422 vs a reference ~158, 167% error -- the automated
  "top Foundation/SR Legacy match" strategy picked an unrepresentative
  record) and one plausible-but-off one (`apple_pie`: predicted 100 vs
  reference 237, because the top match was "Pie fillings, apple, canned"
  rather than a prepared/baked pie, both real USDA entries for the same
  query). This is a real, useful finding: a single top-search-result
  strategy is fragile to which specific record ranks first, and would
  benefit from either a curated food-name-to-fdcId mapping or a smarter
  re-ranking step if calorie accuracy needs to improve.

## App (`app/streamlit_app.py`, `app/components.py`)

- `load_pipeline()` is `@st.cache_resource`-wrapped so switching the
  sidebar's classifier selector doesn't reload+rebuild both models on every
  interaction; `PlateVisionPipeline.run_on_array()` (promoted from a
  private `_run_on_array()` once the app needed it) takes the BGR array
  straight from `cv2.imdecode()` on the uploaded file's bytes, no temp file
  needed.
- **Real bug caught while first running it**: `streamlit run
  app/streamlit_app.py` puts the script's own directory (`app/`) on
  `sys.path`, not the repo root, so the file's own `from app.components
  import ...` absolute import raised `ModuleNotFoundError: No module named
  'app'` on first launch. Fixed by inserting the repo root onto `sys.path`
  at the top of `streamlit_app.py` before those imports -- every other
  script in this project already ran via `python -u path/to/script.py`
  from the repo root (which doesn't have this problem, since there's no
  competing `path/to/` directory shadowing the top-level packages), so this
  is the first place it surfaced.
- Verified by actually launching the app and driving it with Playwright
  (Python) against headless Chromium -- no `chromium-cli` in this
  environment, so `pip install playwright && playwright install chromium`
  plus a short custom script (`page.goto()` -> `set_input_files()` the
  upload input -> `wait_for_selector("text=Total estimated calories")` ->
  `screenshot()`) stood in for it. Uploaded a real UECFOOD-256 photo
  (`9_832.jpg`, fried rice): the annotated image rendered with both boxes
  (green food box labeled with the classifier's guess + confidence, blue
  plate box), the per-item card showed a correct thumbnail crop +
  `Fried Rice` + 94% confidence + 380 kcal, and the total matched. Zero
  browser console errors. Since this took real package installs and a
  hand-written driver (no project run-skill existed yet), consider
  `/run-skill-generator` to capture this as a reusable project skill.
