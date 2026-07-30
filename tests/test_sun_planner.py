from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from backend.sun_planner import (
    AssistantSettings,
    VllmClient,
    Venue,
    VenueRetriever,
    build_assistant_client,
    cosine_similarity,
    fallback_anchor_hint,
    load_venues,
    planner_venues_near,
    point_in_feature,
    rank_venues_by_sun,
    time_relation_for_request,
    venue_preference_for_request,
    venues_near,
)


class FakeEmbeddingClient:
    def __init__(self) -> None:
        self.settings = AssistantSettings(
            enabled=True,
            ollama_base_url="http://example.invalid",
            chat_model="fake-chat",
            embedding_model="fake-embedding",
            timeout_seconds=1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            words = text.casefold().split()
            vectors.append(
                [
                    float(sum(word in {"coffee", "cafe", "bakery"} for word in words)),
                    float(sum(word in {"beer", "bar", "wine"} for word in words)),
                    float(len(words)),
                ]
            )
        return vectors


class SunPlannerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.venues = (
            Venue(
                venue_id="sunny-cafe",
                name="Sunny Cafe",
                area="Kamppi",
                kind="cafe",
                latitude=60.1700,
                longitude=24.9400,
                terrace_note="Coffee and a terrace.",
                source_label="Example",
                source_url="https://example.test/sunny",
            ),
            Venue(
                venue_id="shaded-bar",
                name="Shaded Bar",
                area="Kamppi",
                kind="bar",
                latitude=60.1705,
                longitude=24.9405,
                terrace_note="Beer and an outdoor table.",
                source_label="Example",
                source_url="https://example.test/shaded",
            ),
        )

    def test_catalogue_contains_a_seed_set_of_helsinki_venues(self) -> None:
        catalogue_path = Path(__file__).resolve().parents[1] / "backend" / "venue_data" / "helsinki_terraces.json"
        venues = load_venues(catalogue_path)

        self.assertGreaterEqual(len(venues), 30)
        self.assertEqual(venues[0].name, "Bar Mendocino")
        self.assertTrue(venues[0].source_url.startswith("https://"))

    def test_nearby_venues_are_sorted_and_limited_to_the_radius(self) -> None:
        nearby = venues_near(self.venues, 60.1700, 24.9400, radius_meters=100)

        self.assertEqual([venue.name for venue, _ in nearby], ["Sunny Cafe", "Shaded Bar"])
        self.assertLess(nearby[0][1], nearby[1][1])

    def test_cold_lager_prefers_local_beer_bars_not_cafes_or_wine_bars(self) -> None:
        wine_bar = Venue(
            venue_id="wine-bar",
            name="Wine Bar",
            area="Kamppi",
            kind="bakery and wine bar",
            latitude=60.1701,
            longitude=24.9401,
            terrace_note="Outdoor point.",
            source_label="Example",
            source_url="https://example.test/wine",
        )
        preference = venue_preference_for_request("All I want is a cold lager", "terrace_or_cafe")
        nearby = planner_venues_near(
            (*self.venues, wine_bar),
            60.1700,
            24.9400,
            preference=preference,
        )

        self.assertEqual(preference, "beer")
        self.assertEqual([venue.name for venue, _ in nearby], ["Shaded Bar"])

    def test_departure_phrase_keeps_the_location_and_before_deadline(self) -> None:
        request = "I am leaving Helsinki from Karhupuisto tomorrow at 2pm and want a lager before"

        self.assertEqual(fallback_anchor_hint(request), "Karhupuisto")
        self.assertEqual(time_relation_for_request(request, "at"), "before")
        self.assertIsNone(fallback_anchor_hint("I am in a bad mood and want a refreshment"))

    def test_ranker_prefers_unshaded_terrace_over_nearer_shaded_one(self) -> None:
        shadow = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[24.9403, 60.1703], [24.9407, 60.1703], [24.9407, 60.1707], [24.9403, 60.1707], [24.9403, 60.1703]]],
            },
        }
        start = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
        samples = [(start + timedelta(minutes=minute), [shadow], True) for minute in (0, 30, 60)]
        recommendations = rank_venues_by_sun(
            [(self.venues[0], 80), (self.venues[1], 20)],
            samples,
            building_geometry_available=True,
        )

        self.assertEqual(recommendations[0]["venue"]["id"], "sunny-cafe")
        self.assertEqual(recommendations[0]["exposure"], "sunny through the next hour")
        self.assertEqual(recommendations[0]["sun_coverage_percent"], 100)
        self.assertEqual(recommendations[0]["ranking_basis"], "projected building shade over the next hour and distance")
        self.assertEqual(recommendations[1]["exposure"], "in projected building shade")

    def test_ranker_marks_night_and_missing_geometry_clearly(self) -> None:
        samples = [(datetime(2026, 7, 22, 22, 0, tzinfo=UTC), [], False)]
        recommendations = rank_venues_by_sun(
            [(self.venues[0], 80)],
            samples,
            building_geometry_available=False,
        )

        self.assertEqual(recommendations[0]["exposure"], "after sunset")
        self.assertIsNone(recommendations[0]["sun_coverage_percent"])
        self.assertEqual(recommendations[0]["ranking_basis"], "distance only because the sun is below the horizon")
        self.assertIsNone(recommendations[0]["sample_details"][0]["building_shade"])

    def test_point_in_feature_uses_the_shadow_polygon(self) -> None:
        feature = {
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[24.0, 60.0], [25.0, 60.0], [25.0, 61.0], [24.0, 61.0], [24.0, 60.0]]],
            }
        }

        self.assertTrue(point_in_feature(24.5, 60.5, feature))
        self.assertFalse(point_in_feature(23.5, 60.5, feature))

    def test_local_retriever_builds_and_searches_a_small_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            retriever = VenueRetriever(
                venues=self.venues,
                index_path=Path(directory) / "venue_index.json",
                client=FakeEmbeddingClient(),
            )
            self.assertEqual(retriever.rebuild(), 2)
            matches = retriever.search("coffee cafe")

        self.assertEqual(matches[0].venue_id, "sunny-cafe")
        self.assertGreater(cosine_similarity([1, 0], [1, 0]), cosine_similarity([1, 0], [0, 1]))

    def test_vllm_client_uses_json_schema_and_disables_qwen_thinking(self) -> None:
        settings = AssistantSettings(
            enabled=True,
            ollama_base_url="http://unused.invalid",
            chat_model="Qwen/Qwen3-8B",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            timeout_seconds=1,
            provider="vllm",
            vllm_chat_base_url="http://vllm-chat.test/v1",
            vllm_embedding_base_url="http://vllm-embed.test/v1",
        )
        client = VllmClient(settings)
        model_response = {
            "choices": [
                {
                    "message": {
                        "content": '{"anchor_query":"Storyville","requested_time":"2026-07-30T13:00:00+03:00","time_relation":"at","venue_kind":"cafe"}'
                    }
                }
            ]
        }
        with patch.object(client, "_request", return_value=model_response) as request:
            intent = client.structured_intent(
                message="Coffee near Storyville tomorrow at 1pm",
                current_time=datetime.fromisoformat("2026-07-29T10:00:00+03:00"),
                selected_time=datetime.fromisoformat("2026-07-29T10:00:00+03:00"),
            )

        self.assertEqual(intent.anchor_query, "Storyville")
        self.assertEqual(intent.venue_kind, "cafe")
        body = request.call_args.args[2]
        self.assertEqual(request.call_args.args[:2], ("http://vllm-chat.test/v1", "/chat/completions"))
        self.assertEqual(body["response_format"]["type"], "json_schema")
        self.assertEqual(body["response_format"]["json_schema"]["name"], "sun_plan_intent")
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(body["max_tokens"], 256)

    def test_vllm_embeddings_are_sorted_by_api_index(self) -> None:
        settings = AssistantSettings(
            enabled=True,
            ollama_base_url="http://unused.invalid",
            chat_model="Qwen/Qwen3-8B",
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            timeout_seconds=1,
            provider="vllm",
            vllm_embedding_base_url="http://vllm-embed.test/v1",
        )
        client = VllmClient(settings)
        response = {
            "data": [
                {"index": 1, "embedding": [0.3, 0.4]},
                {"index": 0, "embedding": [0.1, 0.2]},
            ]
        }
        with patch.object(client, "_request", return_value=response) as request:
            vectors = client.embed(["first", "second"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(request.call_args.args[:2], ("http://vllm-embed.test/v1", "/embeddings"))
        self.assertEqual(request.call_args.args[2]["model"], "Qwen/Qwen3-Embedding-0.6B")

    def test_provider_factory_keeps_ollama_default_and_can_select_vllm(self) -> None:
        ollama_settings = AssistantSettings(True, "http://localhost:11434", "chat", "embed", 1)
        vllm_settings = AssistantSettings(True, "http://localhost:11434", "chat", "embed", 1, provider="vllm")

        self.assertEqual(type(build_assistant_client(ollama_settings)).__name__, "OllamaClient")
        self.assertIsInstance(build_assistant_client(vllm_settings), VllmClient)

    def test_vllm_environment_uses_hugging_face_model_defaults(self) -> None:
        with patch.dict("os.environ", {"SUNFINDER_LLM_PROVIDER": "vllm"}, clear=True):
            settings = AssistantSettings.from_environment()

        self.assertEqual(settings.provider, "vllm")
        self.assertEqual(settings.chat_model, "Qwen/Qwen3-8B-AWQ")
        self.assertEqual(settings.embedding_model, "Qwen/Qwen3-Embedding-0.6B")


if __name__ == "__main__":
    unittest.main()
