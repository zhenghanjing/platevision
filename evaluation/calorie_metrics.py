"""Error metrics for the calorie estimation stage.

`REFERENCE_KCAL_PER_100G` is a small, hand-picked set of commonly cited
nutrition reference figures (kcal per 100g) for the food names already
present in `estimation/cache/base_calories.json` -- i.e. foods this
project's USDA-lookup pipeline has already produced a Base_Cal for. It
exists to give `evaluate_base_cal_lookup()` an independent-ish number to
compare against, so the MAPE/MAE below are computed from something, not
nothing.

This is explicitly a small illustrative sanity check (as many samples as
happen to be in the cache -- currently 7), not a rigorous nutrition audit:
the reference figures are typical/commonly-cited approximations from
general nutrition knowledge, not all independently re-verified against a
second database. One of them (apple_pie) does happen to match a specific
USDA SR Legacy record ("Pie, apple, commercially prepared, enriched
flour" = 237 kcal/100g) found via a manual search earlier in this
project's development, which is a good illustration of *why* MAPE isn't
0%: the automated pipeline's top search match for "apple pie" landed on a
different, lower-calorie record ("Pie fillings, apple, canned" = 100
kcal/100g) -- both are real USDA entries for the same query, and picking
between them is exactly the kind of ambiguity a single-sample "any
Foundation/SR Legacy match" lookup strategy can't resolve on its own.
"""

import json
from pathlib import Path
from typing import Any

REFERENCE_KCAL_PER_100G: dict[str, float] = {
    "apple_pie": 237.0,
    "beef_carpaccio": 172.0,
    "bread_pudding": 153.0,
    "chocolate_mousse": 283.0,
    "fried_rice": 163.0,
    "lasagna": 135.0,
    "paella": 158.0,
}


def mean_absolute_error(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true:
        raise ValueError("y_true/y_pred must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    return sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)


def mean_absolute_percentage_error(y_true: list[float], y_pred: list[float]) -> float:
    """Returns a percentage (e.g. 12.5, not 0.125). Raises if any `y_true` entry is 0."""
    if not y_true:
        raise ValueError("y_true/y_pred must be non-empty")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if any(t == 0 for t in y_true):
        raise ValueError("mean_absolute_percentage_error is undefined when y_true contains 0")
    return sum(abs((t - p) / t) for t, p in zip(y_true, y_pred)) / len(y_true) * 100.0


def evaluate_base_cal_lookup(
    cache_path: Path | str,
    reference: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare the USDA-lookup cache's Base_Cal values against `reference`
    (default `REFERENCE_KCAL_PER_100G`) for whichever food names overlap,
    and report MAE/MAPE. Small-sample sanity check, not a rigorous benchmark
    -- see the module docstring."""
    reference = reference if reference is not None else REFERENCE_KCAL_PER_100G
    cache: dict[str, float] = json.loads(Path(cache_path).read_text(encoding="utf-8"))

    common = sorted(set(cache) & set(reference))
    if not common:
        raise ValueError(f"No overlap between {cache_path} and the reference set")

    y_true = [reference[name] for name in common]
    y_pred = [cache[name] for name in common]

    return {
        "foods": common,
        "reference_kcal": y_true,
        "predicted_kcal": y_pred,
        "mae": mean_absolute_error(y_true, y_pred),
        "mape_percent": mean_absolute_percentage_error(y_true, y_pred),
        "num_samples": len(common),
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"Base_Cal lookup evaluation ({report['num_samples']} sample(s), kcal/100g):")
    header = f"  {'food':<20} {'reference':>10} {'predicted':>10} {'abs err':>10} {'pct err':>10}"
    print(header)
    for name, ref, pred in zip(report["foods"], report["reference_kcal"], report["predicted_kcal"]):
        abs_err = abs(ref - pred)
        pct_err = abs_err / ref * 100.0
        print(f"  {name:<20} {ref:>10.1f} {pred:>10.1f} {abs_err:>10.1f} {pct_err:>9.1f}%")
    print(f"\n  MAE:  {report['mae']:.2f} kcal/100g")
    print(f"  MAPE: {report['mape_percent']:.1f}%")


if __name__ == "__main__":
    default_cache = Path(__file__).resolve().parent.parent / "estimation" / "cache" / "base_calories.json"
    print_report(evaluate_base_cal_lookup(default_cache))
