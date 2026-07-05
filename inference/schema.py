"""Result types shared across the end-to-end inference pipeline."""

from dataclasses import dataclass

from models.classifier import ClassificationResult
from models.detector import BBox, Detection


@dataclass
class FoodItemResult:
    detection: Detection
    classification: ClassificationResult
    # None if calorie estimation failed for this item alone (e.g. no USDA
    # match, or a network error) -- one bad lookup shouldn't sink the whole
    # plate's results.
    estimated_calories: float | None


@dataclass
class PlateResult:
    plate_bbox: BBox
    # "hough_circle" | "contour" | "full_image" -- see estimation/plate_detector.py.
    # Check for "full_image" to flag/discount low-confidence calorie estimates:
    # that tier means no plate boundary was actually found in the photo.
    plate_detection_method: str
    items: list[FoodItemResult]

    def total_calories(self) -> float:
        return sum(item.estimated_calories for item in self.items if item.estimated_calories is not None)
