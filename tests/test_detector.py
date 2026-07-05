import pytest

from models.detector import BBox, YOLOv8Detector


def test_detector_load_missing_weights_raises(tmp_path):
    detector = YOLOv8Detector(weights_path=tmp_path / "does_not_exist.pt")
    with pytest.raises(FileNotFoundError):
        detector.load()


def test_predict_before_load_raises():
    detector = YOLOv8Detector(weights_path="yolov8n.pt")
    with pytest.raises(RuntimeError):
        detector.predict(None)


def test_bbox_area():
    assert BBox(0, 0, 10, 10).area() == 100.0
    assert BBox(0, 0, 0, 0).area() == 0.0
    assert BBox(10, 10, 5, 5).area() == 0.0  # inverted coords clamp to zero, not negative
