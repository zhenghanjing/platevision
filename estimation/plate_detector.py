"""Classical computer-vision fallback for locating the plate's bounding box.

UECFOOD-256 has no "plate"/tableware annotations (see data/uecfood256.py),
so the YOLOv8 detector can't be trained to output a plate box directly.
This module estimates one straight from the raw image instead, in three
tiers:

1. Hough circle transform on a blurred grayscale image. Plate photos in
   this kind of dataset are usually shot roughly top-down, so the plate rim
   tends to show up as close to a circle.
2. If no circle is found (e.g. an angled shot makes the rim elliptical),
   fall back to the largest sufficiently round contour from Canny edges.
3. If neither finds anything (cluttered background, plate edge not
   visible), fall back to the whole image as the plate bbox -- crude, but
   keeps `Calories = Base_Cal * (food_area / plate_area)` defined rather
   than dividing by an unknown quantity.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from models.detector import BBox


@dataclass
class PlateDetectionResult:
    bbox: BBox
    method: str  # "hough_circle" | "contour" | "full_image"


def detect_plate_bbox(image: np.ndarray) -> PlateDetectionResult:
    """Estimate the plate's bounding box in a BGR image (e.g. from cv2.imread)."""
    height, width = image.shape[:2]

    bbox = _detect_via_hough_circle(image)
    if bbox is not None:
        return PlateDetectionResult(bbox=bbox, method="hough_circle")

    bbox = _detect_via_largest_round_contour(image)
    if bbox is not None:
        return PlateDetectionResult(bbox=bbox, method="contour")

    return PlateDetectionResult(bbox=BBox(0.0, 0.0, float(width), float(height)), method="full_image")


def _detect_via_hough_circle(image: np.ndarray) -> BBox | None:
    height, width = image.shape[:2]
    min_dim = min(height, width)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    gray = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min_dim / 2,
        param1=100,
        param2=60,
        minRadius=int(min_dim * 0.2),
        maxRadius=int(min_dim * 0.49),
    )
    if circles is None:
        return None

    cx, cy, r = max(circles[0], key=lambda c: c[2])  # keep the largest (most plate-like) circle
    x1 = max(0.0, float(cx - r))
    y1 = max(0.0, float(cy - r))
    x2 = min(float(width), float(cx + r))
    y2 = min(float(height), float(cy + r))
    return BBox(x1, y1, x2, y2)


def _detect_via_largest_round_contour(image: np.ndarray) -> BBox | None:
    height, width = image.shape[:2]
    image_area = width * height

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_bbox: BBox | None = None
    best_circularity = 0.8  # a filled square scores ~0.785; require rounder than that
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.1:  # too small to plausibly be the plate
            continue
        perimeter = cv2.arcLength(contour, closed=True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter**2)  # 1.0 for a perfect circle
        if circularity > best_circularity:
            best_circularity = circularity
            x, y, w, h = cv2.boundingRect(contour)
            best_bbox = BBox(float(x), float(y), float(x + w), float(y + h))

    return best_bbox
