from __future__ import annotations

import unittest
from urllib.error import URLError
from unittest.mock import patch

from backend.main import Bounds, BuildingStore, FALLBACK_FEATURES, wfs_to_features


class BuildingStoreTests(unittest.TestCase):
    def test_manual_retry_bypasses_a_cached_fallback(self) -> None:
        store = BuildingStore()
        bounds = Bounds(60.163, 24.929, 60.170, 24.941)

        with patch("backend.main.fetch_helsinki_buildings", side_effect=URLError("temporary outage")):
            _, source, cached = store.get(bounds)

        self.assertEqual(source, "fallback")
        self.assertFalse(cached)

        with patch("backend.main.fetch_helsinki_buildings", return_value=FALLBACK_FEATURES[:4]) as fetch:
            features, source, cached = store.get(bounds, retry_failed=True)

        fetch.assert_called_once_with(bounds)
        self.assertEqual(source, "helsinki-wfs")
        self.assertFalse(cached)
        self.assertEqual(len(features), 4)

    def test_wfs_features_keep_geometry_and_estimate_height_from_floors(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [{
                "id": "Rakennukset_alue_rekisteritiedot.123",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[24.931, 60.165], [24.932, 60.165], [24.932, 60.166], [24.931, 60.165]]],
                },
                "properties": {
                    "katunimi_suomi": "Eerikinkatu",
                    "osoitenumero": "28",
                    "i_kerrlkm": 4,
                },
            }],
        }

        features = wfs_to_features(payload)

        self.assertEqual(len(features), 1)
        self.assertEqual(features[0]["properties"]["name"], "Eerikinkatu 28")
        self.assertEqual(features[0]["properties"]["height"], 14.5)
        self.assertEqual(features[0]["geometry"]["coordinates"][0][0], [24.931, 60.165])


if __name__ == "__main__":
    unittest.main()
