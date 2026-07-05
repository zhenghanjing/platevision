"""End-to-end pipeline: detection -> plate localization -> classification -> calorie estimation."""

from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import requests

from estimation.calorie import estimate_calories
from estimation.fdc_client import FoodDataCentralClient
from estimation.plate_detector import detect_plate_bbox
from inference.schema import FoodItemResult, PlateResult
from models.classifier import FoodClassifier
from models.detector import BBox, YOLOv8Detector
from models.efficientnet_b4 import EfficientNetB4Classifier
from models.resnet50 import ResNet50Classifier

_CLASSIFIER_CLASSES: dict[str, type[FoodClassifier]] = {
    "resnet50": ResNet50Classifier,
    "efficientnet_b4": EfficientNetB4Classifier,
}


class PlateVisionPipeline:
    """Chains YOLOv8 food detection, classical-CV plate localization, a
    Food-101 classifier, and USDA-based calorie estimation for one photo.

    The detector's own class predictions are not used for identity -- only
    its boxes are, as region proposals for the classifier. The two models
    can (and in this project's smoke tests, do) have disjoint class
    vocabularies, since the detector is trained on UECFOOD-256 and the
    classifier on Food-101; that's a valid detect-then-classify split, not a
    bug, as long as callers know which model's classes are authoritative
    (the classifier's).

    Plate localization edge cases (by design, not left to the caller):
    - No plate boundary found at all: `estimation.plate_detector` already
      falls back through Hough circle -> largest round contour -> the whole
      image as a last resort, so this pipeline never receives "no plate."
      A whole-image proxy is more useful than skipping calorie estimation
      outright, as long as it's visible when that happened -- which is why
      `PlateResult.plate_detection_method` is carried through. Treat
      `"full_image"` as a low-confidence flag downstream.
    - Multiple candidate circles (e.g. a round food item competing with the
      actual plate): `plate_detector` already keeps the *largest* one, since
      the plate is expected to be the biggest round object in frame.
    """

    def __init__(
        self,
        detector: YOLOv8Detector,
        classifier: FoodClassifier,
        fdc_client: FoodDataCentralClient,
        conf_threshold: float = 0.25,
    ) -> None:
        self.detector = detector
        self.classifier = classifier
        self.fdc_client = fdc_client
        self.conf_threshold = conf_threshold

    @classmethod
    def from_checkpoints(
        cls,
        detector_weights: Path | str,
        classifier_weights: Path | str,
        classifier_backbone: Literal["resnet50", "efficientnet_b4"] = "resnet50",
        device: str = "cpu",
        fdc_api_key: str | None = None,
        conf_threshold: float = 0.25,
    ) -> "PlateVisionPipeline":
        """Build a ready-to-use pipeline from checkpoint paths.

        `classifier_backbone` picks which of the two trained classifiers
        `classifier_weights` belongs to -- e.g. "resnet50" for the 224x224
        checkpoint that currently scores highest, or "efficientnet_b4" for
        its 380x380 one, if that checkpoint is what you pass. `device`
        accepts either convention ("0", "cuda:0", or "cpu") and is
        normalized per component internally (Ultralytics and raw PyTorch
        disagree on the short form -- see CLAUDE.md).
        """
        detector = YOLOv8Detector(weights_path=detector_weights, device=_to_ultralytics_device(device))
        detector.load()

        classifier_cls = _CLASSIFIER_CLASSES[classifier_backbone]
        classifier = classifier_cls(weights_path=classifier_weights, device=_to_torch_device(device))
        classifier.load()

        fdc_client = FoodDataCentralClient(api_key=fdc_api_key)

        return cls(
            detector=detector,
            classifier=classifier,
            fdc_client=fdc_client,
            conf_threshold=conf_threshold,
        )

    def run(self, image_path: Path | str) -> PlateResult:
        """Run detection, plate localization, classification, and calorie
        estimation on the image at `image_path`."""
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return self.run_on_array(image_bgr)

    def run_on_array(self, image_bgr: np.ndarray) -> PlateResult:
        """Same as `run()`, but for an already-decoded BGR image array (e.g. from
        `cv2.imdecode` on an uploaded file) instead of a path on disk."""
        detections = self.detector.predict(image_bgr, conf_threshold=self.conf_threshold)
        plate_detection = detect_plate_bbox(image_bgr)

        items: list[FoodItemResult] = []
        for detection in detections:
            crop_rgb = self._crop_rgb(image_bgr, detection.bbox)
            if crop_rgb is None:
                continue  # degenerate bbox (e.g. rounds to zero width/height)

            classification = self.classifier.predict(crop_rgb)
            estimated_calories = self._estimate_calories(
                classification.class_name, detection.bbox, plate_detection.bbox
            )

            items.append(
                FoodItemResult(
                    detection=detection,
                    classification=classification,
                    estimated_calories=estimated_calories,
                )
            )

        return PlateResult(
            plate_bbox=plate_detection.bbox,
            plate_detection_method=plate_detection.method,
            items=items,
        )

    def _estimate_calories(self, class_name: str, food_bbox: BBox, plate_bbox: BBox) -> float | None:
        """Base_Cal lookup + area-ratio formula; None if either step fails (e.g. no
        USDA match, or a network error) so one bad lookup doesn't sink the whole plate."""
        try:
            base_cal = self.fdc_client.get_base_calories(class_name)
            return estimate_calories(base_cal, food_bbox, plate_bbox)
        except (ValueError, requests.RequestException) as exc:
            print(f"[pipeline] calorie estimation failed for {class_name!r}: {exc}")
            return None

    @staticmethod
    def _crop_rgb(image_bgr: np.ndarray, bbox: BBox) -> np.ndarray | None:
        """Crop `bbox` out of a BGR image and convert to RGB (classifiers expect RGB,
        matching how their training data was loaded via PIL). None for a degenerate bbox."""
        height, width = image_bgr.shape[:2]
        x1 = max(0, int(round(bbox.x1)))
        y1 = max(0, int(round(bbox.y1)))
        x2 = min(width, int(round(bbox.x2)))
        y2 = min(height, int(round(bbox.y2)))
        if x2 <= x1 or y2 <= y1:
            return None
        return cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)


def _to_ultralytics_device(device: str) -> str:
    """Ultralytics accepts "cpu"/"0"/"cuda:0"; normalize a bare "cuda:N" to "N"."""
    return device.split(":", 1)[1] if device.startswith("cuda:") else device


def _to_torch_device(device: str) -> str:
    """torch needs "cpu"/"cuda"/"cuda:N"; normalize a bare digit to "cuda:N"."""
    return f"cuda:{device}" if device.isdigit() else device
