"""Reusable Streamlit UI pieces for rendering pipeline results."""

import cv2
import numpy as np
import streamlit as st

from inference.schema import FoodItemResult, PlateResult
from models.detector import BBox

PLATE_COLOR = (255, 80, 0)  # BGR blue-ish, drawn on the annotated image
FOOD_COLOR = (0, 200, 0)  # BGR green


def _bbox_to_int(bbox: BBox) -> tuple[int, int, int, int]:
    return int(round(bbox.x1)), int(round(bbox.y1)), int(round(bbox.x2)), int(round(bbox.y2))


def draw_detections(image_bgr: np.ndarray, result: PlateResult) -> np.ndarray:
    """Return a copy of `image_bgr` with the plate bbox and each food bbox drawn on it."""
    annotated = image_bgr.copy()

    px1, py1, px2, py2 = _bbox_to_int(result.plate_bbox)
    cv2.rectangle(annotated, (px1, py1), (px2, py2), PLATE_COLOR, 2)
    # Short label -- the full plate_detection_method (e.g. "full_image") is
    # already surfaced as a separate warning in render_summary() when it
    # matters; a long on-image label just runs off the edge of the photo.
    cv2.putText(annotated, "plate", (px1, max(15, py1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, PLATE_COLOR, 2)

    for i, item in enumerate(result.items, 1):
        x1, y1, x2, y2 = _bbox_to_int(item.detection.bbox)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), FOOD_COLOR, 2)
        label = f"#{i} {item.classification.class_name} {item.classification.confidence:.0%}"
        cv2.putText(annotated, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, FOOD_COLOR, 2)

    return annotated


def render_food_item_card(image_bgr: np.ndarray, index: int, item: FoodItemResult) -> None:
    """Render one food item's crop thumbnail alongside its classification + calorie estimate."""
    x1, y1, x2, y2 = _bbox_to_int(item.detection.bbox)
    height, width = image_bgr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    col1, col2 = st.columns([1, 3])
    with col1:
        if x2 > x1 and y2 > y1:
            crop_rgb = cv2.cvtColor(image_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            st.image(crop_rgb, width=120)
    with col2:
        name = item.classification.class_name.replace("_", " ").title()
        st.markdown(f"**#{index}. {name}**")
        st.caption(
            f"classifier confidence: {item.classification.confidence:.0%}  |  "
            f"detector confidence: {item.detection.confidence:.0%}"
        )
        if item.estimated_calories is not None:
            st.markdown(f"Estimated calories: **{item.estimated_calories:.0f} kcal**")
        else:
            st.markdown(":gray[Estimated calories: unavailable (calorie lookup failed)]")


def render_summary(image_bgr: np.ndarray, result: PlateResult) -> None:
    """Render the annotated plate image, a per-item breakdown, and the total calorie count."""
    annotated_rgb = cv2.cvtColor(draw_detections(image_bgr, result), cv2.COLOR_BGR2RGB)
    caption = f"Detected {len(result.items)} food item(s) -- plate localization: {result.plate_detection_method}"
    st.image(annotated_rgb, caption=caption, use_container_width=True)

    if result.plate_detection_method == "full_image":
        st.warning(
            "No plate boundary was found in this photo -- calorie estimates use the whole "
            "image as a rough proxy for portion size, so they're lower-confidence than usual."
        )

    if not result.items:
        st.info("No food items detected.")
        return

    st.subheader("Detected items")
    for i, item in enumerate(result.items, 1):
        render_food_item_card(image_bgr, i, item)
        st.divider()

    st.subheader(f"Total estimated calories: {result.total_calories():.0f} kcal")
    missing = sum(1 for item in result.items if item.estimated_calories is None)
    if missing:
        st.caption(f"{missing} item(s) excluded from the total (calorie lookup failed).")
