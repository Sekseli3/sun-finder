from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.main import PlaceSearchStore, nominatim_to_place_results, places


class PlaceSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_known_venue_alias_is_instant_and_does_not_call_the_upstream_search(self) -> None:
        store = PlaceSearchStore()

        with patch("backend.main.fetch_nominatim_places") as fetch:
            results, source, cached = store.get("Bar Eerikinkulma")

        fetch.assert_not_called()
        self.assertFalse(cached)
        self.assertEqual(source, "curated")
        self.assertEqual(results[0]["name"], "Eerikin Kulma")
        self.assertEqual(results[0]["detail"], "Eerikinkatu 28")

    def test_nominatim_results_keep_only_helsinki_coordinates_and_clean_labels(self) -> None:
        results = nominatim_to_place_results(
            [
                {
                    "name": "Example Bar",
                    "lat": "60.166064",
                    "lon": "24.932312",
                    "category": "amenity",
                    "address": {
                        "road": "Eerikinkatu",
                        "house_number": "24",
                        "suburb": "Kamppi",
                        "city": "Helsinki",
                    },
                },
                {"name": "Outside Helsinki", "lat": "61.0", "lon": "24.9"},
            ]
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["name"], "Example Bar")
        self.assertEqual(results[0]["detail"], "Eerikinkatu 24, Kamppi, Helsinki")
        self.assertEqual(results[0]["kind"], "amenity")

    async def test_places_endpoint_returns_small_safe_results_for_the_browser(self) -> None:
        search_result = [
            {
                "name": "Buenos Aires Cafe/Bar",
                "detail": "Eerikinkatu 24",
                "latitude": 60.166064,
                "longitude": 24.932312,
                "kind": "bar",
            }
        ]
        with patch("backend.main.place_search_store.get", return_value=(search_result, "curated", False)):
            payload = await places(q="Buenos Aires")

        self.assertTrue(payload["meta"]["available"])
        self.assertFalse(payload["meta"]["cached"])
        self.assertEqual(payload["results"], search_result)


if __name__ == "__main__":
    unittest.main()
