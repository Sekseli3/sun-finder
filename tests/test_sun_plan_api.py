from __future__ import annotations

import unittest
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

from backend import main
from backend.sun_planner import AssistantSettings, SunPlanIntent, SunPlanRequest


class SunPlanApiTests(unittest.IsolatedAsyncioTestCase):
    async def test_status_is_disabled_by_default_for_the_public_service(self) -> None:
        with patch.object(
            main,
            "assistant_settings",
            AssistantSettings(False, "http://localhost:11434", "chat", "embed", 1),
        ):
            payload = await main.sun_planner_status()

        self.assertFalse(payload["enabled"])
        self.assertFalse(payload["ready"])

    async def test_plan_uses_deterministic_candidates_and_returns_local_model_copy(self) -> None:
        request = SunPlanRequest(
            message="Outdoor coffee near Eerikinkatu tomorrow after work",
            map_latitude=60.1657789,
            map_longitude=24.9313873,
            selected_time=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
        )
        settings = AssistantSettings(True, "http://localhost:11434", "chat", "embed", 1)
        weather = {
            "available": True,
            "applies_to_selected_time": False,
            "label": "Clear-sky potential",
            "cloud_cover": None,
            "note": "Future geometry only.",
            "nowcast": {"available": False, "probability": None, "uncertainty": {"lower": None, "upper": None}},
        }
        with (
            patch.object(main, "assistant_settings", settings),
            patch.object(main.assistant_client, "available_models", return_value={"chat", "embed"}),
            patch.object(
                main.assistant_client,
                "structured_intent",
                return_value=SunPlanIntent(anchor_query=None, requested_time=None),
            ),
            patch.object(main.building_store, "get", return_value=(main.FALLBACK_FEATURES[:4], "helsinki-wfs", True)),
            patch("backend.main.current_conditions", new=AsyncMock(return_value={"weather": weather})),
            patch.object(main.venue_retriever, "search", return_value=[]),
            patch.object(main.assistant_client, "write_answer", return_value="Try the top result."),
        ):
            payload = await main.sun_plans(request)

        self.assertEqual(payload["answer"], "Try the top result.")
        self.assertEqual(payload["request"]["window_minutes"], 60)
        self.assertTrue(payload["meta"]["building_geometry_available"])
        self.assertGreaterEqual(len(payload["recommendations"]), 1)
        self.assertIn("source", payload["recommendations"][0]["venue"])


if __name__ == "__main__":
    unittest.main()
