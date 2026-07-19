from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.main import FALLBACK_FEATURES, scene


class ScenePayloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_time_update_skips_buildings_but_returns_new_shadows(self) -> None:
        building_result = (FALLBACK_FEATURES[:4], "openstreetmap", True)
        with patch("backend.main.building_store.get", return_value=building_result):
            morning = await scene(
                bbox="60.163,24.929,60.170,24.941",
                at="2026-07-19T05:00:00Z",
                live=False,
                retry_buildings=False,
                include_buildings=True,
            )
            evening = await scene(
                bbox="60.163,24.929,60.170,24.941",
                at="2026-07-19T14:00:00Z",
                live=False,
                retry_buildings=False,
                include_buildings=False,
            )

        self.assertIn("buildings", morning)
        self.assertNotIn("buildings", evening)
        self.assertTrue(evening["meta"]["cached"])
        self.assertFalse(evening["meta"]["buildings_included"])
        self.assertNotEqual(morning["shadows"], evening["shadows"])


if __name__ == "__main__":
    unittest.main()
