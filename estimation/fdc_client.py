"""Client for the USDA FoodData Central API, used to look up Base_Cal per food class.

API docs: https://fdc.nal.usda.gov/api-guide. Reads the API key from the
`FDC_API_KEY` environment variable, loaded from a `.env` file at the repo
root if present (falls back to api.data.gov's shared, rate-limited
"DEMO_KEY" if neither is set -- fine for occasional testing, get a personal
key for real use; never commit a real key -- `.env` is gitignored).

Base_Cal is taken as the "Energy" nutrient (USDA nutrient id 1008, reported
in kcal per 100g for Foundation/SR Legacy records) of the best search match.
That's a per-100g reference value, not a literal "one serving" -- consistent
with `estimation/calorie.py` treating Base_Cal as a reference amount scaled
by the food/plate area ratio, not an absolute serving size.
"""

import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from estimation.cache import BaseCalCache

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

DEFAULT_CACHE_PATH = Path(__file__).resolve().parent / "cache" / "base_calories.json"

# USDA nutrient id for "Energy" reported in kcal.
ENERGY_NUTRIENT_ID = 1008

# Prefer these curated data types (no branded-product noise) before falling
# back to an unrestricted search.
PREFERRED_DATA_TYPES = ["Foundation", "SR Legacy"]


class FoodDataCentralClient:
    BASE_URL = "https://api.nal.usda.gov/fdc/v1"

    def __init__(self, api_key: str | None = None, cache: BaseCalCache | None = None) -> None:
        self.api_key = api_key or os.environ.get("FDC_API_KEY", "DEMO_KEY")
        self.cache = cache if cache is not None else BaseCalCache(DEFAULT_CACHE_PATH)

    def search_food(self, query: str, data_types: list[str] | None = None) -> list[dict]:
        """Search FoodData Central for candidate matches to `query`."""
        params = {"query": query, "api_key": self.api_key, "pageSize": 10}
        if data_types:
            params["dataType"] = data_types

        response = requests.get(f"{self.BASE_URL}/foods/search", params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("foods", [])

    def get_base_calories(self, food_name: str) -> float:
        """Look up the reference calorie value (kcal per 100g) for `food_name`.

        `food_name` is expected in Food-101's underscored form (e.g.
        "apple_pie"); it's both the cache key and, space-separated, the
        search query.
        """
        cached = self.cache.get(food_name)
        if cached is not None:
            return cached

        query = food_name.replace("_", " ")
        foods = self.search_food(query, data_types=PREFERRED_DATA_TYPES)
        if not foods:
            foods = self.search_food(query)  # fall back to any data type, e.g. Branded
        if not foods:
            raise ValueError(f"No USDA FoodData Central match for {food_name!r}")

        base_cal = next(
            (cal for food in foods[:5] if (cal := self._extract_energy_kcal(food)) is not None),
            None,
        )
        if base_cal is None:
            raise ValueError(f"No Energy (KCAL) nutrient found for {food_name!r}")

        self.cache.set(food_name, base_cal)
        return base_cal

    @staticmethod
    def _extract_energy_kcal(food: dict) -> float | None:
        for nutrient in food.get("foodNutrients", []):
            if nutrient.get("nutrientId") == ENERGY_NUTRIENT_ID:
                return float(nutrient["value"])
        return None
