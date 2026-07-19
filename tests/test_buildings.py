from __future__ import annotations

import unittest
from urllib.error import URLError
from unittest.mock import patch

from backend.main import Bounds, BuildingStore, FALLBACK_FEATURES


class BuildingStoreTests(unittest.TestCase):
    def test_manual_retry_bypasses_a_cached_fallback(self) -> None:
        store = BuildingStore()
        bounds = Bounds(60.163, 24.929, 60.170, 24.941)

        with patch("backend.main.fetch_openstreetmap_buildings", side_effect=URLError("temporary outage")):
            _, source, cached = store.get(bounds)

        self.assertEqual(source, "fallback")
        self.assertFalse(cached)

        with patch("backend.main.fetch_openstreetmap_buildings", return_value=FALLBACK_FEATURES[:4]) as fetch:
            features, source, cached = store.get(bounds, retry_failed=True)

        fetch.assert_called_once_with(bounds)
        self.assertEqual(source, "openstreetmap")
        self.assertFalse(cached)
        self.assertEqual(len(features), 4)


if __name__ == "__main__":
    unittest.main()
