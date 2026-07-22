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
        self.assertNotIn("sun_score", payload["recommendations"][0])

    async def test_plan_uses_deterministic_copy_when_building_geometry_is_missing(self) -> None:
        request = SunPlanRequest(
            message="I want a beer in Eerikinkatu",
            map_latitude=60.1657789,
            map_longitude=24.9313873,
            selected_time=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
        )
        settings = AssistantSettings(True, "http://localhost:11434", "chat", "embed", 1)
        weather = {
            "available": True,
            "applies_to_selected_time": True,
            "label": "Overcast",
            "cloud_cover": 96,
            "note": "Cloudy now.",
            "nowcast": {"available": True, "probability": 71, "uncertainty": {"lower": 55, "upper": 83}},
        }
        with (
            patch.object(main, "assistant_settings", settings),
            patch.object(main.assistant_client, "available_models", return_value={"chat", "embed"}),
            patch.object(
                main.assistant_client,
                "structured_intent",
                return_value=SunPlanIntent(anchor_query=None, requested_time=None, venue_kind="bar"),
            ),
            patch.object(main.building_store, "get", return_value=(main.FALLBACK_FEATURES[:4], "fallback", False)),
            patch("backend.main.current_conditions", new=AsyncMock(return_value={"weather": weather})),
            patch.object(main.venue_retriever, "search", return_value=[]),
            patch.object(main.assistant_client, "write_answer") as writer,
        ):
            payload = await main.sun_plans(request)

        writer.assert_not_called()
        self.assertFalse(payload["meta"]["building_geometry_available"])
        self.assertIn("closest curated choices", payload["answer"])
        self.assertIn("71%", payload["answer"])
        self.assertNotIn("96%", payload["answer"])
        self.assertEqual(payload["recommendations"][0]["ranking_basis"], "distance only because building data is unavailable")

    def test_language_facts_do_not_present_cloud_cover_as_a_venue_score(self) -> None:
        facts = main.planner_language_facts(
            anchor={"name": "Eerikinkatu", "detail": "Kamppi", "latitude": 60.166, "longitude": 24.933},
            planned_time=datetime(2026, 7, 22, 15, 0, tzinfo=UTC),
            building_geometry_available=True,
            building_source="helsinki-wfs",
            weather={
                "applies_to_selected_time": True,
                "cloud_cover": 96,
                "nowcast": {"available": True, "probability": 71},
            },
            recommendations=[
                {
                    "venue": {"name": "Example Bar", "area": "Kamppi", "terrace_note": "Check availability."},
                    "distance_meters": 40,
                    "exposure": "sunny through the next hour",
                    "ranking_basis": "projected building shade over the next hour and distance",
                    "sun_coverage_percent": 100,
                    "ranking_score": 99,
                }
            ],
        )

        self.assertEqual(facts["weather"]["direct_sun_probability"], 71)
        self.assertNotIn("cloud_cover", facts["weather"])
        self.assertNotIn("sun_score", facts["recommendations"][0])
        self.assertNotIn("ranking_score", facts["recommendations"][0])


if __name__ == "__main__":
    unittest.main()
