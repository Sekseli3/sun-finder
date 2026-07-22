from __future__ import annotations

import json
import unittest
from urllib.error import URLError
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from backend.main import Bounds, BuildingStore, FALLBACK_FEATURES, fetch_helsinki_buildings, wfs_to_features


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

    def test_wfs_request_filters_the_mixed_layer_to_building_polygons(self) -> None:
        payload = {
            "type": "FeatureCollection",
            "features": [{
                "id": "building.1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[24.931, 60.165], [24.932, 60.165], [24.932, 60.166], [24.931, 60.165]]],
                },
                "properties": {"i_kerrlkm": 2},
            }],
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        bounds = Bounds(60.163, 24.929, 60.170, 24.941)
        with patch("backend.main.urlopen", return_value=Response()) as urlopen:
            features = fetch_helsinki_buildings(bounds)

        request = urlopen.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(len(features), 1)
        self.assertNotIn("bbox", query)
        self.assertIn("CQL_FILTER", query)
        self.assertIn("BBOX(geom,24.92900,60.16300,24.94100,60.17000,'EPSG:4326')", query["CQL_FILTER"][0])
        self.assertIn("geometryType(geom) = 'Polygon'", query["CQL_FILTER"][0])

    def test_valid_sparse_wfs_response_is_not_replaced_with_demo_buildings(self) -> None:
        store = BuildingStore()
        bounds = Bounds(60.163, 24.929, 60.170, 24.941)
        sparse_features = FALLBACK_FEATURES[:1]

        with patch("backend.main.fetch_helsinki_buildings", return_value=sparse_features):
            features, source, cached = store.get(bounds)

        self.assertEqual(features, sparse_features)
        self.assertEqual(source, "helsinki-wfs")
        self.assertFalse(cached)


if __name__ == "__main__":
    unittest.main()
