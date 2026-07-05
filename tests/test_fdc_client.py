from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from estimation.cache import BaseCalCache
from estimation.fdc_client import FoodDataCentralClient


def _mock_response(foods: list[dict]) -> Mock:
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"foods": foods}
    return response


def _food_with_energy(description: str, kcal: float) -> dict:
    return {
        "description": description,
        "foodNutrients": [{"nutrientId": 1008, "value": kcal, "unitName": "KCAL"}],
    }


@pytest.fixture
def client(tmp_path: Path) -> FoodDataCentralClient:
    return FoodDataCentralClient(api_key="test-key", cache=BaseCalCache(tmp_path / "base_calories.json"))


def test_get_base_calories_returns_energy_from_top_match(client):
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response([_food_with_energy("Apple pie", 237.0)])
        result = client.get_base_calories("apple_pie")
    assert result == 237.0


def test_get_base_calories_caches_and_skips_second_network_call(client):
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response([_food_with_energy("Apple pie", 237.0)])
        client.get_base_calories("apple_pie")
        client.get_base_calories("apple_pie")
    assert mock_get.call_count == 1


def test_get_base_calories_falls_back_when_preferred_data_types_empty(client):
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.side_effect = [
            _mock_response([]),  # preferred data types (Foundation/SR Legacy): no hits
            _mock_response([_food_with_energy("Apple Pie, Branded", 300.0)]),  # unrestricted fallback
        ]
        result = client.get_base_calories("apple_pie")
    assert result == 300.0
    assert mock_get.call_count == 2


def test_get_base_calories_raises_when_no_results(client):
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response([])
        with pytest.raises(ValueError):
            client.get_base_calories("not_a_real_food")


def test_get_base_calories_raises_when_no_energy_nutrient(client):
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response([{"description": "Mystery item", "foodNutrients": []}])
        with pytest.raises(ValueError):
            client.get_base_calories("mystery_item")


def test_cache_persists_across_client_instances(tmp_path):
    cache_path = tmp_path / "base_calories.json"

    client1 = FoodDataCentralClient(api_key="test-key", cache=BaseCalCache(cache_path))
    with patch("estimation.fdc_client.requests.get") as mock_get:
        mock_get.return_value = _mock_response([_food_with_energy("Sushi", 150.0)])
        client1.get_base_calories("sushi")

    client2 = FoodDataCentralClient(api_key="test-key", cache=BaseCalCache(cache_path))  # reloads from disk
    with patch("estimation.fdc_client.requests.get") as mock_get:
        result = client2.get_base_calories("sushi")
        mock_get.assert_not_called()
    assert result == 150.0
