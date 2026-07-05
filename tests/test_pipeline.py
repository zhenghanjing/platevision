from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from estimation.fdc_client import FoodDataCentralClient
from inference.pipeline import PlateVisionPipeline, _to_torch_device, _to_ultralytics_device
from models.classifier import ClassificationResult
from models.detector import BBox, Detection


def _make_image() -> np.ndarray:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.circle(image, (100, 100), 80, (200, 200, 200), thickness=-1)
    return image


def _make_pipeline(detections, classifications, base_cal_side_effect=None) -> PlateVisionPipeline:
    detector = Mock()
    detector.predict.return_value = detections

    classifier = Mock()
    classifier.predict.side_effect = classifications

    fdc_client = Mock(spec=FoodDataCentralClient)
    if base_cal_side_effect is not None:
        fdc_client.get_base_calories.side_effect = base_cal_side_effect
    else:
        fdc_client.get_base_calories.return_value = 200.0

    return PlateVisionPipeline(detector=detector, classifier=classifier, fdc_client=fdc_client)


def test_run_on_array_happy_path():
    detections = [
        Detection(bbox=BBox(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="rice"),
        Detection(bbox=BBox(60, 60, 120, 120), confidence=0.8, class_id=1, class_name="sushi"),
    ]
    classifications = [
        ClassificationResult(class_id=3, class_name="apple_pie", confidence=0.7),
        ClassificationResult(class_id=5, class_name="beef_tartare", confidence=0.6),
    ]
    pipeline = _make_pipeline(detections, classifications)

    result = pipeline.run_on_array(_make_image())

    assert len(result.items) == 2
    assert result.items[0].classification.class_name == "apple_pie"
    assert result.items[0].estimated_calories is not None
    assert result.total_calories() == pytest.approx(
        result.items[0].estimated_calories + result.items[1].estimated_calories
    )
    assert result.plate_detection_method in ("hough_circle", "contour", "full_image")


def test_run_on_array_no_detections_returns_empty_items():
    pipeline = _make_pipeline([], [])
    result = pipeline.run_on_array(_make_image())
    assert result.items == []
    assert result.total_calories() == 0.0


def test_run_on_array_skips_degenerate_bbox():
    detections = [
        Detection(bbox=BBox(10, 10, 10, 10), confidence=0.9, class_id=0, class_name="rice"),  # zero area
        Detection(bbox=BBox(20, 20, 40, 40), confidence=0.9, class_id=0, class_name="rice"),
    ]
    classifications = [ClassificationResult(class_id=0, class_name="apple_pie", confidence=0.5)]
    pipeline = _make_pipeline(detections, classifications)

    result = pipeline.run_on_array(_make_image())

    assert len(result.items) == 1  # degenerate bbox skipped, classifier only called once
    pipeline.classifier.predict.assert_called_once()


def test_run_on_array_calorie_failure_does_not_sink_other_items():
    detections = [
        Detection(bbox=BBox(10, 10, 50, 50), confidence=0.9, class_id=0, class_name="rice"),
        Detection(bbox=BBox(60, 60, 120, 120), confidence=0.8, class_id=1, class_name="sushi"),
    ]
    classifications = [
        ClassificationResult(class_id=3, class_name="unknown_food", confidence=0.7),
        ClassificationResult(class_id=5, class_name="beef_tartare", confidence=0.6),
    ]
    pipeline = _make_pipeline(detections, classifications, base_cal_side_effect=[ValueError("no match"), 150.0])

    result = pipeline.run_on_array(_make_image())

    assert result.items[0].estimated_calories is None
    assert result.items[1].estimated_calories is not None
    assert result.total_calories() == pytest.approx(result.items[1].estimated_calories)


def test_run_raises_file_not_found_for_missing_image(tmp_path):
    pipeline = _make_pipeline([], [])
    with pytest.raises(FileNotFoundError):
        pipeline.run(tmp_path / "does_not_exist.jpg")


@pytest.mark.parametrize("device,expected", [("cpu", "cpu"), ("0", "0"), ("cuda:0", "0"), ("cuda:1", "1")])
def test_to_ultralytics_device(device, expected):
    assert _to_ultralytics_device(device) == expected


@pytest.mark.parametrize("device,expected", [("cpu", "cpu"), ("cuda", "cuda"), ("cuda:0", "cuda:0"), ("0", "cuda:0")])
def test_to_torch_device(device, expected):
    assert _to_torch_device(device) == expected
