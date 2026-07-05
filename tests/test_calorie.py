import pytest

from estimation.calorie import estimate_calories
from models.detector import BBox


def test_estimate_calories_scales_with_area_ratio():
    food_bbox = BBox(0, 0, 10, 10)  # area 100
    plate_bbox = BBox(0, 0, 20, 20)  # area 400
    assert estimate_calories(base_cal=200, food_bbox=food_bbox, plate_bbox=plate_bbox) == pytest.approx(50.0)


def test_estimate_calories_food_covers_whole_plate():
    food_bbox = BBox(0, 0, 20, 20)
    plate_bbox = BBox(0, 0, 20, 20)
    assert estimate_calories(base_cal=300, food_bbox=food_bbox, plate_bbox=plate_bbox) == pytest.approx(300.0)


def test_estimate_calories_zero_food_area_gives_zero_calories():
    food_bbox = BBox(5, 5, 5, 5)  # degenerate: zero width and height
    plate_bbox = BBox(0, 0, 20, 20)
    assert estimate_calories(base_cal=300, food_bbox=food_bbox, plate_bbox=plate_bbox) == 0.0


def test_estimate_calories_zero_plate_area_raises_value_error():
    food_bbox = BBox(0, 0, 10, 10)
    plate_bbox = BBox(5, 5, 5, 5)  # degenerate: zero area
    with pytest.raises(ValueError):
        estimate_calories(base_cal=300, food_bbox=food_bbox, plate_bbox=plate_bbox)


def test_estimate_calories_inverted_plate_coords_treated_as_zero_area():
    # x2 < x1 and y2 < y1 -- BBox.area() clamps negative extents to 0 rather
    # than returning a negative area, so this should raise the same as an
    # explicitly zero-area plate.
    food_bbox = BBox(0, 0, 10, 10)
    plate_bbox = BBox(10, 10, 5, 5)
    with pytest.raises(ValueError):
        estimate_calories(base_cal=300, food_bbox=food_bbox, plate_bbox=plate_bbox)


def test_estimate_calories_food_larger_than_plate_is_not_clamped():
    # A detection artifact (food bbox exceeding the plate bbox) intentionally
    # is not clamped to a ratio of 1 -- it's surfaced as-is so callers can
    # decide how to treat an implausible detection.
    food_bbox = BBox(0, 0, 30, 30)  # area 900
    plate_bbox = BBox(0, 0, 20, 20)  # area 400
    result = estimate_calories(base_cal=100, food_bbox=food_bbox, plate_bbox=plate_bbox)
    assert result == pytest.approx(225.0)


def test_estimate_calories_zero_base_cal_gives_zero():
    food_bbox = BBox(0, 0, 10, 10)
    plate_bbox = BBox(0, 0, 20, 20)
    assert estimate_calories(base_cal=0, food_bbox=food_bbox, plate_bbox=plate_bbox) == 0.0
