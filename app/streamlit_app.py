"""Streamlit demo: upload a plate photo, see detected food items and calories.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # `streamlit run app/streamlit_app.py` puts this file's own directory
    # (app/) on sys.path, not the repo root -- without this, the absolute
    # `app.*`/`inference.*`/etc. imports below fail with ModuleNotFoundError.
    sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

from app.components import render_summary
from inference.pipeline import PlateVisionPipeline

DETECTOR_WEIGHTS = REPO_ROOT / "runs" / "detect" / "smoke_test_food_only" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = {
    "resnet50": REPO_ROOT / "runs" / "classify" / "full101_resnet50.pt",
    "efficientnet_b4": REPO_ROOT / "runs" / "classify" / "full101_efficientnet_b4.pt",
}
CLASSIFIER_LABELS = {
    "resnet50": "ResNet-50 (224x224, 78.3% top-1)",
    "efficientnet_b4": "EfficientNet-B4 (380x380, 83.9% top-1)",
}


@st.cache_resource
def load_pipeline(classifier_backbone: str) -> PlateVisionPipeline:
    """Construct and cache the detector/classifier/estimator pipeline."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return PlateVisionPipeline.from_checkpoints(
        detector_weights=DETECTOR_WEIGHTS,
        classifier_weights=CLASSIFIER_WEIGHTS[classifier_backbone],
        classifier_backbone=classifier_backbone,
        device=device,
    )


def main() -> None:
    st.set_page_config(page_title="PlateVision", page_icon=":fork_and_knife:")
    st.title("PlateVision")
    st.write("Upload a photo of a plate to detect foods and estimate calories.")

    backbone = st.sidebar.selectbox(
        "Classifier",
        options=list(CLASSIFIER_WEIGHTS),
        format_func=lambda b: CLASSIFIER_LABELS[b],
    )
    conf_threshold = st.sidebar.slider("Detector confidence threshold", 0.0, 1.0, 0.25, 0.05)

    uploaded_file = st.file_uploader("Plate photo", type=["jpg", "jpeg", "png"])
    if uploaded_file is None:
        return

    file_bytes = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image_bgr is None:
        st.error("Could not read that image.")
        return

    pipeline = load_pipeline(backbone)
    pipeline.conf_threshold = conf_threshold

    with st.spinner("Running detection, classification, and calorie estimation..."):
        result = pipeline.run_on_array(image_bgr)

    render_summary(image_bgr, result)


if __name__ == "__main__":
    main()
