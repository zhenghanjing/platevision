"""YOLOv8 food detector wrapper (Ultralytics), fine-tuned on UECFOOD-256.

UECFOOD-256 has no "plate"/tableware annotations, so this detector only
outputs food-item boxes. Plate bbox estimation (needed as the area
denominator in estimation/calorie.py) is handled separately by
estimation/plate_detector.py.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO


@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float

    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass
class Detection:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


class YOLOv8Detector:
    """Thin wrapper around an Ultralytics YOLOv8 model for food detection.

    `weights_path` can be a stock checkpoint name (e.g. "yolov8n.pt", which
    Ultralytics downloads and runs with COCO's 80 classes -- useful to smoke
    test the wrapper before a fine-tuned model exists) or a local path to a
    checkpoint fine-tuned on UECFOOD-256 (e.g. "runs/detect/train/weights/best.pt").
    """

    def __init__(self, weights_path: Path | str, device: str = "cpu") -> None:
        self.weights_path = weights_path
        self.device = device
        self.model: YOLO | None = None

    def load(self) -> None:
        """Load the YOLOv8 weights into `self.model`."""
        self.model = YOLO(str(self.weights_path))

    def predict(self, image: np.ndarray, conf_threshold: float = 0.25) -> list[Detection]:
        """Run inference on a single image and return detected boxes."""
        if self.model is None:
            raise RuntimeError("Call load() before predict().")

        results = self.model.predict(image, conf=conf_threshold, device=self.device, verbose=False)
        result = results[0]

        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        names = result.names
        for xyxy, confidence, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()
        ):
            class_id = int(cls)
            detections.append(
                Detection(
                    bbox=BBox(*xyxy),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=names[class_id],
                )
            )
        return detections
