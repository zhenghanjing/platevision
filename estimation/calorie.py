"""Portion-aware calorie estimation.

Calories = Base_Cal * (food_bbox_area / plate_bbox_area)

`Base_Cal` is the calorie value for a reference/full portion of a food
class, looked up via `estimation.fdc_client`.
"""

from models.detector import BBox


def estimate_calories(base_cal: float, food_bbox: BBox, plate_bbox: BBox) -> float:
    """Estimate calories for a food item given its and the plate's bounding boxes.

    Raises ValueError if `plate_bbox` has zero (or degenerate, e.g. x2<x1)
    area -- that denominator is undefined, not just "small", so this
    surfaces the bad plate detection rather than silently producing inf/nan.

    A `food_bbox` larger than `plate_bbox` (ratio > 1) is not clamped: it's
    a real signal that the plate or food detection was off, and callers are
    better placed to decide how to react than this function is.
    """
    plate_area = plate_bbox.area()
    if plate_area <= 0:
        raise ValueError(f"plate_bbox has non-positive area ({plate_area}); cannot compute a ratio")

    food_area = food_bbox.area()
    return base_cal * (food_area / plate_area)
