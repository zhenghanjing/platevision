"""Local disk cache for Base_Cal lookups, to avoid repeat USDA FDC API calls."""

import json
from pathlib import Path


class BaseCalCache:
    """A {food_name: base_cal_kcal} map persisted as a JSON file."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = Path(cache_path)
        self._data: dict[str, float] = {}
        self.load()

    def load(self) -> None:
        if self.cache_path.exists():
            self._data = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8"
        )

    def get(self, food_name: str) -> float | None:
        return self._data.get(food_name)

    def set(self, food_name: str, base_cal: float) -> None:
        self._data[food_name] = base_cal
        self.save()
