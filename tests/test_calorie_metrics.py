import json

import pytest

from evaluation.calorie_metrics import (
    evaluate_base_cal_lookup,
    mean_absolute_error,
    mean_absolute_percentage_error,
)


def test_mean_absolute_error_basic():
    assert mean_absolute_error([100.0, 200.0], [90.0, 220.0]) == pytest.approx(15.0)


def test_mean_absolute_error_zero_when_exact():
    assert mean_absolute_error([100.0, 50.0], [100.0, 50.0]) == 0.0


def test_mean_absolute_percentage_error_basic():
    # errors of 10% and 20% -> average 15%
    assert mean_absolute_percentage_error([100.0, 200.0], [90.0, 240.0]) == pytest.approx(15.0)


def test_mean_absolute_percentage_error_raises_on_zero_true_value():
    with pytest.raises(ValueError):
        mean_absolute_percentage_error([0.0, 100.0], [10.0, 100.0])


def test_mean_absolute_error_raises_on_empty():
    with pytest.raises(ValueError):
        mean_absolute_error([], [])


def test_mean_absolute_error_raises_on_mismatched_length():
    with pytest.raises(ValueError):
        mean_absolute_error([1.0, 2.0], [1.0])


def test_evaluate_base_cal_lookup_computes_overlap_only(tmp_path):
    cache_path = tmp_path / "base_calories.json"
    cache_path.write_text(json.dumps({"apple_pie": 200.0, "unknown_food": 999.0}), encoding="utf-8")

    report = evaluate_base_cal_lookup(cache_path, reference={"apple_pie": 237.0, "sushi": 150.0})

    assert report["foods"] == ["apple_pie"]
    assert report["reference_kcal"] == [237.0]
    assert report["predicted_kcal"] == [200.0]
    assert report["num_samples"] == 1
    assert report["mae"] == pytest.approx(37.0)


def test_evaluate_base_cal_lookup_raises_on_no_overlap(tmp_path):
    cache_path = tmp_path / "base_calories.json"
    cache_path.write_text(json.dumps({"unknown_food": 999.0}), encoding="utf-8")

    with pytest.raises(ValueError):
        evaluate_base_cal_lookup(cache_path, reference={"apple_pie": 237.0})
