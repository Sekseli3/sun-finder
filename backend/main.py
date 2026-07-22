"""FastAPI service for Helsinki building shadows.

The browser handles interaction, visible building tiles, and fast local shadow
previews. This module owns solar geometry, weather, the direct-sun nowcast,
place search, and a City of Helsinki building fallback for API callers.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles

from backend.nowcast import direct_sun_nowcast, unavailable_direct_sun_nowcast
from backend.sun_planner import (
    AssistantSettings,
    OllamaClient,
    OllamaUnavailableError,
    RetrievedVenueDocument,
    SunPlanRequest,
    VenueRetriever,
    load_environment_file,
    load_venues,
    rank_venues_by_sun,
    venues_near,
)


APP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_ROOT / "frontend"
HELSINKI_LATITUDE = 60.1699
HELSINKI_LONGITUDE = 24.9384
HELSINKI_TIME_ZONE = ZoneInfo("Europe/Helsinki")
METERS_PER_DEGREE_LAT = 111_320
DEFAULT_BUILDING_HEIGHT = 15.0
MAX_SHADOW_METERS = 560.0
MAX_BUILDINGS_PER_REQUEST = 3_000
HELSINKI_WFS_ENDPOINT = "https://kartta.hel.fi/ws/geoserver/avoindata/wfs"
HELSINKI_WFS_BUILDINGS_LAYER = "avoindata:Rakennukset_alue_rekisteritiedot"
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
NOMINATIM_SEARCH_ENDPOINT = "https://nominatim.openstreetmap.org/search"
PHOTON_SUGGESTION_ENDPOINT = "https://photon.komoot.io/api/"
WEATHER_CACHE_SECONDS = 5 * 60
HELSINKI_WFS_HTTP_TIMEOUT_SECONDS = 12
FAILED_BUILDING_CACHE_SECONDS = 15
PLACE_SEARCH_CACHE_SECONDS = 60 * 60
FAILED_PLACE_SEARCH_CACHE_SECONDS = 2 * 60
PLACE_SEARCH_MIN_REQUEST_INTERVAL_SECONDS = 1.1
PLACE_SEARCH_HTTP_TIMEOUT_SECONDS = 8
MAX_PLACE_SEARCH_RESULTS = 5
PLACE_SUGGESTION_CACHE_SECONDS = 20 * 60
FAILED_PLACE_SUGGESTION_CACHE_SECONDS = 60
PLACE_SUGGESTION_MIN_REQUEST_INTERVAL_SECONDS = 0.5
PLACE_SUGGESTION_HTTP_TIMEOUT_SECONDS = 6
MAX_PLACE_SUGGESTIONS = 6
PLANNER_SAMPLE_MINUTES = (0, 30, 60)
PLANNER_BOUNDS_PADDING_DEGREES_LATITUDE = 0.007
PLANNER_BOUNDS_PADDING_DEGREES_LONGITUDE = 0.014

# This bounds check keeps the City of Helsinki building query firmly scoped to
# the product's Helsinki use case. It is intentionally broader than city centre.
HELSINKI_REGION = (59.95, 24.60, 60.38, 25.40)  # south, west, north, east
MAX_QUERY_LATITUDE_SPAN = 0.09
MAX_QUERY_LONGITUDE_SPAN = 0.14
VENUE_CATALOGUE_PATH = APP_ROOT / "backend" / "venue_data" / "helsinki_terraces.json"
LOCAL_VENUE_INDEX_PATH = APP_ROOT / ".sunfinder" / "venue_index.json"

load_environment_file(APP_ROOT / ".env")
assistant_settings = AssistantSettings.from_environment()
assistant_client = OllamaClient(assistant_settings)
venue_catalogue = load_venues(VENUE_CATALOGUE_PATH)
venue_retriever = VenueRetriever(
    venues=venue_catalogue,
    index_path=LOCAL_VENUE_INDEX_PATH,
    client=assistant_client,
)


@dataclass(frozen=True)
class Bounds:
    south: float
    west: float
    north: float
    east: float

    @property
    def cache_key(self) -> str:
        return ":".join(f"{value:.3f}" for value in (self.south, self.west, self.north, self.east))


@dataclass
class BuildingCacheEntry:
    expires_at: float
    features: list[dict[str, Any]]
    source: str


@dataclass
class WeatherCacheEntry:
    expires_at: float
    weather: dict[str, Any]


@dataclass
class PlaceSearchCacheEntry:
    expires_at: float
    results: list[dict[str, Any]]
    source: str


def feature(name: str, height: float, coordinates: list[list[float]]) -> dict[str, Any]:
    """Create a small, durable fallback footprint for central Helsinki."""
    return {
        "type": "Feature",
        "properties": {"name": name, "height": height, "source": "starter"},
        "geometry": {"type": "Polygon", "coordinates": [[*coordinates, coordinates[0]]]},
    }


FALLBACK_FEATURES = [
    feature("Senate Square block", 18, [[24.94849, 60.17005], [24.94906, 60.17018], [24.94927, 60.16990], [24.94869, 60.16977]]),
    feature("Sofiankatu block", 21, [[24.94692, 60.16968], [24.94758, 60.16985], [24.94786, 60.16953], [24.94718, 60.16935]]),
    feature("Kiseleff House", 20, [[24.94765, 60.16898], [24.94843, 60.16918], [24.94870, 60.16887], [24.94793, 60.16866]]),
    feature("Market Square block", 17, [[24.95410, 60.16778], [24.95512, 60.16802], [24.95553, 60.16763], [24.95449, 60.16740]]),
    feature("Esplanadi quarter", 24, [[24.94294, 60.16803], [24.94406, 60.16828], [24.94450, 60.16775], [24.94337, 60.16750]]),
    feature("Aleksanterinkatu block", 27, [[24.94017, 60.16993], [24.94123, 60.17018], [24.94155, 60.16971], [24.94048, 60.16947]]),
    feature("Kluuvi block", 32, [[24.94510, 60.17162], [24.94612, 60.17190], [24.94646, 60.17141], [24.94541, 60.17114]]),
    feature("Railway station block", 34, [[24.93831, 60.17125], [24.93944, 60.17154], [24.93984, 60.17098], [24.93865, 60.17068]]),
    feature("Kaisaniemi block", 22, [[24.94744, 60.17404], [24.94828, 60.17428], [24.94863, 60.17382], [24.94776, 60.17358]]),
    feature("Katajanokka block", 18, [[24.96223, 60.16985], [24.96313, 60.17008], [24.96348, 60.16963], [24.96258, 60.16941]]),
    feature("Töölönlahti block", 25, [[24.93018, 60.17643], [24.93144, 60.17675], [24.93183, 60.17623], [24.93055, 60.17594]]),
    feature("Kallio block", 23, [[24.95036, 60.18424], [24.95127, 60.18444], [24.95160, 60.18399], [24.95068, 60.18378]]),
]


# These are useful Helsinki venues that should work even if a public place
# index has not received a recent OpenStreetMap edit. They also make the first
# search feel instant for the suggested examples.
CURATED_PLACES = (
    {
        "name": "Bar Mendocino",
        "detail": "Eerikinkatu",
        "latitude": 60.1657789,
        "longitude": 24.9313873,
        "kind": "bar",
        "aliases": ("bar mendocino", "mendocino"),
    },
    {
        "name": "Eerikin Kulma",
        "detail": "Eerikinkatu 28",
        "latitude": 60.1658342,
        "longitude": 24.9316017,
        "kind": "bar",
        "aliases": ("eerikin kulma", "eerikinkulma", "bar eerikinkulma", "pub ek"),
    },
    {
        "name": "Buenos Aires Cafe/Bar",
        "detail": "Eerikinkatu 24",
        "latitude": 60.1660640,
        "longitude": 24.9323120,
        "kind": "bar",
        "aliases": ("buenos aires", "bar buenos aires", "buenos aires cafe bar"),
    },
)


class BuildingStore:
    """A small in-memory cache around Helsinki's official building WFS."""

    def __init__(self) -> None:
        self._cache: dict[str, BuildingCacheEntry] = {}
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.Lock()

    def get(self, bounds: Bounds, *, retry_failed: bool = False) -> tuple[list[dict[str, Any]], str, bool]:
        cache_key = bounds.cache_key
        while True:
            now = time.monotonic()
            with self._lock:
                cached = self._cache.get(cache_key)
                should_retry_fallback = bool(retry_failed and cached and cached.source == "fallback")
                if cached and cached.expires_at > now and not should_retry_fallback:
                    return cached.features, cached.source, True
                in_flight = self._inflight.get(cache_key)
                if in_flight is None:
                    in_flight = threading.Event()
                    self._inflight[cache_key] = in_flight
                    break
            # Several requests for the same visible area can arrive together.
            # Let them share one WFS request instead of piling up.
            in_flight.wait()

        try:
            try:
                features = fetch_helsinki_buildings(bounds)
                if len(features) < 4:
                    raise ValueError("Helsinki WFS returned too few building footprints")
                source = "helsinki-wfs"
                ttl_seconds = 12 * 60 * 60
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                # The map remains useful when the public building service is unavailable.
                print(f"Helsinki WFS building fetch failed: {error}")
                features = fallback_features_in(bounds)
                source = "fallback"
                ttl_seconds = FAILED_BUILDING_CACHE_SECONDS

            with self._lock:
                self._cache[cache_key] = BuildingCacheEntry(
                    expires_at=now + ttl_seconds,
                    features=features,
                    source=source,
                )
            return features, source, False
        finally:
            with self._lock:
                completed = self._inflight.pop(cache_key, None)
            if completed:
                completed.set()


building_store = BuildingStore()


class WeatherStore:
    """Cache a small current-sky payload so live refreshes stay inexpensive."""

    def __init__(self) -> None:
        self._entry: WeatherCacheEntry | None = None
        self._lock = threading.Lock()

    def get(self) -> tuple[dict[str, Any], bool]:
        now = time.monotonic()
        with self._lock:
            if self._entry and self._entry.expires_at > now:
                return self._entry.weather, True

        try:
            weather = fetch_current_weather()
            ttl_seconds = WEATHER_CACHE_SECONDS
        except (HTTPError, URLError, TimeoutError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            print(f"Current weather fetch failed: {error}")
            weather = unavailable_weather()
            ttl_seconds = 2 * 60

        with self._lock:
            self._entry = WeatherCacheEntry(expires_at=now + ttl_seconds, weather=weather)
        return weather, False


weather_store = WeatherStore()


class PlaceSearchStore:
    """A small cache and one-request-at-a-time gate for public place search.

    Search is only invoked after a person submits the form. The gate keeps
    requests to the public Nominatim service below its one request per second
    policy, while the cache makes repeated venue searches free.
    """

    def __init__(self) -> None:
        self._cache: dict[str, PlaceSearchCacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._upstream_lock = threading.Lock()
        self._last_upstream_request_at = 0.0

    def get(self, query: str) -> tuple[list[dict[str, Any]], str, bool]:
        cache_key = normalise_place_query(query)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return cached.results, cached.source, True

        curated = curated_place_results(query)
        if curated:
            return curated, "curated", False

        # Keep all outgoing requests serial so separate search terms do not
        # accidentally turn a few quick taps into a burst against Nominatim.
        with self._upstream_lock:
            now = time.monotonic()
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached and cached.expires_at > now:
                    return cached.results, cached.source, True

            wait_seconds = PLACE_SEARCH_MIN_REQUEST_INTERVAL_SECONDS - (now - self._last_upstream_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_upstream_request_at = time.monotonic()

            try:
                results = fetch_nominatim_places(query)
                source = "nominatim"
                ttl_seconds = PLACE_SEARCH_CACHE_SECONDS
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                print(f"Place search failed: {error}")
                results = []
                source = "unavailable"
                ttl_seconds = FAILED_PLACE_SEARCH_CACHE_SECONDS

            with self._cache_lock:
                self._cache[cache_key] = PlaceSearchCacheEntry(
                    expires_at=time.monotonic() + ttl_seconds,
                    results=results,
                    source=source,
                )
            return results, source, False


place_search_store = PlaceSearchStore()


def normalise_place_query(value: str) -> str:
    return " ".join("".join(character if character.isalnum() else " " for character in value.casefold()).split())


def curated_place_results(query: str) -> list[dict[str, Any]]:
    query_key = normalise_place_query(query)
    if not query_key:
        return []
    query_words = set(query_key.split())
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for index, place in enumerate(CURATED_PLACES):
        name_keys = [normalise_place_query(place["name"])]
        name_keys.extend(normalise_place_query(alias) for alias in place["aliases"])
        detail_key = normalise_place_query(place["detail"])
        searchable_words = set(" ".join((*name_keys, detail_key)).split())

        if query_key in name_keys:
            score = 0
        elif any(name_key.startswith(query_key) for name_key in name_keys):
            score = 1
        elif any(query_key in name_key for name_key in name_keys):
            score = 2
        elif query_key in detail_key:
            score = 3
        elif query_words.issubset(searchable_words):
            score = 4
        else:
            continue
        result = {key: place[key] for key in ("name", "detail", "latitude", "longitude", "kind")}
        matches.append((score, index, result))
    return [result for _, _, result in sorted(matches)]


class PlaceSuggestionStore:
    """Cache and gently pace Helsinki-only place suggestions.

    Photon supports search while someone is typing. Curated matches skip the
    network entirely, while the small server-side gate and cache keep the
    public suggestion service from receiving a request for every key press.
    """

    def __init__(self) -> None:
        self._cache: dict[str, PlaceSearchCacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._upstream_lock = threading.Lock()
        self._last_upstream_request_at = 0.0

    def get(self, query: str) -> tuple[list[dict[str, Any]], str, bool]:
        cache_key = normalise_place_query(query)
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            if cached and cached.expires_at > now:
                return cached.results, cached.source, True

        curated = curated_place_results(query)
        if curated:
            return curated[:MAX_PLACE_SUGGESTIONS], "curated", False

        with self._upstream_lock:
            now = time.monotonic()
            with self._cache_lock:
                cached = self._cache.get(cache_key)
                if cached and cached.expires_at > now:
                    return cached.results, cached.source, True

            wait_seconds = PLACE_SUGGESTION_MIN_REQUEST_INTERVAL_SECONDS - (now - self._last_upstream_request_at)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
            self._last_upstream_request_at = time.monotonic()

            try:
                results = fetch_photon_suggestions(query)
                source = "photon"
                ttl_seconds = PLACE_SUGGESTION_CACHE_SECONDS
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                print(f"Place suggestions failed: {error}")
                results = []
                source = "unavailable"
                ttl_seconds = FAILED_PLACE_SUGGESTION_CACHE_SECONDS

            with self._cache_lock:
                self._cache[cache_key] = PlaceSearchCacheEntry(
                    expires_at=time.monotonic() + ttl_seconds,
                    results=results,
                    source=source,
                )
            return results, source, False


place_suggestion_store = PlaceSuggestionStore()


def fetch_photon_suggestions(query_text: str) -> list[dict[str, Any]]:
    south, west, north, east = HELSINKI_REGION
    query = urlencode(
        {
            "q": query_text,
            "limit": str(MAX_PLACE_SUGGESTIONS),
            "bbox": f"{west:.2f},{south:.2f},{east:.2f},{north:.2f}",
            "lat": f"{HELSINKI_LATITUDE:.4f}",
            "lon": f"{HELSINKI_LONGITUDE:.4f}",
            "lang": "en",
        }
    )
    request = Request(
        f"{PHOTON_SUGGESTION_ENDPOINT}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "SunfinderHelsinki/1.0 (+https://sunfinder-helsinki.onrender.com/)",
        },
    )
    with urlopen(request, timeout=PLACE_SUGGESTION_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public geocoder behind a cache and gate
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("features"), list):
        raise ValueError("Place suggestion search returned an unexpected payload")
    return photon_to_place_results(payload["features"])


def photon_to_place_results(features: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[float, float]] = set()
    for feature in features:
        if not isinstance(feature, dict):
            continue
        geometry = feature.get("geometry")
        properties = feature.get("properties")
        if not isinstance(geometry, dict) or not isinstance(properties, dict):
            continue
        coordinates = geometry.get("coordinates")
        if not isinstance(coordinates, list) or len(coordinates) < 2:
            continue
        try:
            longitude = float(coordinates[0])
            latitude = float(coordinates[1])
        except (TypeError, ValueError):
            continue
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not is_in_helsinki_region(latitude, longitude):
            continue
        coordinate_key = (round(latitude, 6), round(longitude, 6))
        if coordinate_key in seen_coordinates:
            continue
        seen_coordinates.add(coordinate_key)
        name, detail = photon_result_name(properties)
        results.append(
            {
                "name": name,
                "detail": detail,
                "latitude": latitude,
                "longitude": longitude,
                "kind": str(properties.get("osm_value") or properties.get("osm_key") or properties.get("type") or "place"),
            }
        )
        if len(results) >= MAX_PLACE_SUGGESTIONS:
            break
    return results


def photon_result_name(properties: dict[str, Any]) -> tuple[str, str]:
    raw_name = str(properties.get("name") or "").strip()
    street = str(properties.get("street") or "").strip()
    house_number = str(properties.get("housenumber") or "").strip()
    street_address = " ".join(part for part in (street, house_number) if part)
    district = str(properties.get("district") or properties.get("locality") or "").strip()
    city = str(properties.get("city") or "Helsinki").strip()
    name = raw_name or street_address or district or city or "Helsinki place"
    detail_parts = [part for part in (street_address if raw_name else "", district, city) if part]
    detail = ", ".join(dict.fromkeys(detail_parts)) or "Helsinki"
    return name, detail


def fetch_nominatim_places(query_text: str) -> list[dict[str, Any]]:
    full_query = query_text if "helsinki" in query_text.casefold() else f"{query_text}, Helsinki"
    query = urlencode(
        {
            "q": full_query,
            "format": "jsonv2",
            "limit": str(MAX_PLACE_SEARCH_RESULTS),
            "countrycodes": "fi",
            "viewbox": "24.60,60.38,25.40,59.95",
            "bounded": "1",
            "addressdetails": "1",
            "dedupe": "1",
        }
    )
    request = Request(
        f"{NOMINATIM_SEARCH_ENDPOINT}?{query}",
        headers={
            "Accept": "application/json",
            "Accept-Language": "fi,en",
            "User-Agent": "SunfinderHelsinki/1.0 (+https://sunfinder-helsinki.onrender.com/)",
        },
    )
    with urlopen(request, timeout=PLACE_SEARCH_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public geocoder with rate limiting
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Place search returned an unexpected payload")
    return nominatim_to_place_results(payload)


def nominatim_to_place_results(payload: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[float, float]] = set()
    for item in payload:
        if not isinstance(item, dict):
            continue
        try:
            latitude = float(item["lat"])
            longitude = float(item["lon"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(latitude) or not math.isfinite(longitude) or not is_in_helsinki_region(latitude, longitude):
            continue
        coordinate_key = (round(latitude, 6), round(longitude, 6))
        if coordinate_key in seen_coordinates:
            continue
        seen_coordinates.add(coordinate_key)
        name, detail = nominatim_result_name(item)
        results.append(
            {
                "name": name,
                "detail": detail,
                "latitude": latitude,
                "longitude": longitude,
                "kind": str(item.get("category") or item.get("type") or "place"),
            }
        )
        if len(results) >= MAX_PLACE_SEARCH_RESULTS:
            break
    return results


def nominatim_result_name(item: dict[str, Any]) -> tuple[str, str]:
    raw_name = str(item.get("name") or "").strip()
    address = item.get("address") if isinstance(item.get("address"), dict) else {}
    road = str(address.get("road") or "").strip()
    house_number = str(address.get("house_number") or "").strip()
    street_address = " ".join(part for part in (road, house_number) if part)
    display_parts = [part.strip() for part in str(item.get("display_name") or "").split(",") if part.strip()]
    name = raw_name or street_address or ", ".join(display_parts[:2]) or "Helsinki place"
    neighbourhood = str(address.get("neighbourhood") or address.get("suburb") or "").strip()
    locality = str(address.get("city") or address.get("town") or address.get("municipality") or "Helsinki").strip()
    detail_parts = [part for part in (street_address if raw_name else "", neighbourhood, locality) if part]
    detail = ", ".join(dict.fromkeys(detail_parts)) or "Helsinki"
    return name, detail


def is_in_helsinki_region(latitude: float, longitude: float) -> bool:
    south, west, north, east = HELSINKI_REGION
    return south <= latitude <= north and west <= longitude <= east


def planner_bounds(latitude: float, longitude: float, candidate_coordinates: list[tuple[float, float]]) -> Bounds:
    """Return one WFS-safe box around the anchor and the nearby terrace points."""
    latitudes = [latitude, *(candidate_latitude for candidate_latitude, _ in candidate_coordinates)]
    longitudes = [longitude, *(candidate_longitude for _, candidate_longitude in candidate_coordinates)]
    region_south, region_west, region_north, region_east = HELSINKI_REGION
    return Bounds(
        south=max(region_south, min(latitudes) - PLANNER_BOUNDS_PADDING_DEGREES_LATITUDE),
        west=max(region_west, min(longitudes) - PLANNER_BOUNDS_PADDING_DEGREES_LONGITUDE),
        north=min(region_north, max(latitudes) + PLANNER_BOUNDS_PADDING_DEGREES_LATITUDE),
        east=min(region_east, max(longitudes) + PLANNER_BOUNDS_PADDING_DEGREES_LONGITUDE),
    )


def planner_timestamp(timestamp: datetime) -> datetime:
    """Interpret a bare LLM time as Helsinki wall time, then use UTC internally."""
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=HELSINKI_TIME_ZONE)
    return timestamp.astimezone(UTC)


def planner_weather_summary(weather: dict[str, Any]) -> dict[str, Any]:
    nowcast = weather.get("nowcast") if isinstance(weather.get("nowcast"), dict) else {}
    uncertainty = nowcast.get("uncertainty") if isinstance(nowcast.get("uncertainty"), dict) else {}
    return {
        "applies_to_selected_time": bool(weather.get("applies_to_selected_time")),
        "available": bool(weather.get("available")),
        "label": weather.get("label"),
        "cloud_cover": weather.get("cloud_cover"),
        "direct_sun_probability": nowcast.get("probability"),
        "direct_sun_range": {
            "lower": uncertainty.get("lower"),
            "upper": uncertainty.get("upper"),
        },
        "note": weather.get("note"),
    }


def planner_language_facts(
    *,
    anchor: dict[str, Any],
    planned_time: datetime,
    recommendations: list[dict[str, Any]],
    building_geometry_available: bool,
    building_source: str,
    weather: dict[str, Any],
) -> dict[str, Any]:
    """Give the writing model only plainly labelled facts it can safely phrase."""
    nowcast = weather.get("nowcast") if isinstance(weather.get("nowcast"), dict) else {}
    if weather.get("applies_to_selected_time") and nowcast.get("available"):
        weather_facts = {
            "description": "City-wide next-hour direct-sun estimate for an open point.",
            "direct_sun_probability": nowcast.get("probability"),
            "interpretation": "This is an average of the forecast for now, +30 minutes, and +60 minutes. It is not a venue-specific score or a statement about the sky at one exact moment.",
        }
    else:
        weather_facts = {
            "description": "Clear-sky potential for the selected map time.",
            "interpretation": "This is solar geometry only, not a local weather forecast.",
        }
    return {
        "anchor": {key: anchor.get(key) for key in ("name", "detail", "latitude", "longitude")},
        "planned_time": planned_time.isoformat(),
        "window_minutes": 60,
        "building_geometry_available": building_geometry_available,
        "building_source": building_source,
        "weather": weather_facts,
        "recommendations": [
            {
                "name": recommendation["venue"]["name"],
                "area": recommendation["venue"]["area"],
                "distance_meters": recommendation["distance_meters"],
                "exposure": recommendation["exposure"],
                "ranking_basis": recommendation["ranking_basis"],
                "outdoor_note": recommendation["venue"]["terrace_note"],
            }
            for recommendation in recommendations
        ],
    }


def deterministic_plan_answer(
    recommendations: list[dict[str, Any]],
    *,
    building_geometry_available: bool,
    weather: dict[str, Any],
) -> str:
    """Keep the planner useful if its second local model call fails."""
    if not recommendations:
        return "I could not find a curated terrace or café close to that spot. Try another Helsinki area."
    names = ", ".join(recommendation["venue"]["name"] for recommendation in recommendations)
    lead = recommendations[0]
    geometry_note = (
        f"{lead['venue']['name']} looks best for projected building sun over the next hour."
        if building_geometry_available
        else "I could not load building geometry for this area, so these are the closest curated choices rather than confirmed sun spots."
    )
    weather_note = (
        f" The city-wide next-hour direct-sun estimate is {weather['nowcast']['probability']}% for an open point, not for a specific venue."
        if weather.get("applies_to_selected_time") and weather.get("nowcast", {}).get("available")
        else " This is clear sky potential for the selected time, not a local weather forecast."
    )
    return f"{geometry_note} Nearby choices are {names}.{weather_note}"


def parse_bounds(raw_bounds: str) -> Bounds:
    try:
        values = [float(value) for value in raw_bounds.split(",")]
    except ValueError as error:
        raise HTTPException(status_code=422, detail="bbox must be south,west,north,east") from error
    if len(values) != 4 or not all(math.isfinite(value) for value in values):
        raise HTTPException(status_code=422, detail="bbox must contain four finite coordinates")
    bounds = Bounds(*values)
    if bounds.south >= bounds.north or bounds.west >= bounds.east:
        raise HTTPException(status_code=422, detail="bbox coordinates are not ordered correctly")
    region_south, region_west, region_north, region_east = HELSINKI_REGION
    if not (
        region_south <= bounds.south <= region_north
        and region_south <= bounds.north <= region_north
        and region_west <= bounds.west <= region_east
        and region_west <= bounds.east <= region_east
    ):
        raise HTTPException(status_code=422, detail="Sunfinder currently serves the Helsinki region only")
    if bounds.north - bounds.south > MAX_QUERY_LATITUDE_SPAN or bounds.east - bounds.west > MAX_QUERY_LONGITUDE_SPAN:
        raise HTTPException(status_code=422, detail="Zoom in a little before loading buildings")
    return bounds


def parse_timestamp(raw_timestamp: str | None) -> datetime:
    if not raw_timestamp:
        return datetime.now(UTC)
    try:
        timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
    except ValueError as error:
        raise HTTPException(status_code=422, detail="at must be an ISO-8601 timestamp") from error
    # Treat an unqualified timestamp as Helsinki wall-clock time; the frontend
    # normally sends an explicit UTC ISO string.
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=HELSINKI_TIME_ZONE)
    return timestamp.astimezone(UTC)


def fetch_helsinki_buildings(bounds: Bounds) -> list[dict[str, Any]]:
    """Fetch one viewport from Helsinki's maintained building register.

    The browser normally uses prebuilt vector tiles. This endpoint keeps the
    Python API useful as a compact fallback without relying on Overpass.
    """
    query = urlencode(
        {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": HELSINKI_WFS_BUILDINGS_LAYER,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "bbox": (
                f"{bounds.west:.5f},{bounds.south:.5f},"
                f"{bounds.east:.5f},{bounds.north:.5f},EPSG:4326"
            ),
            "count": str(MAX_BUILDINGS_PER_REQUEST),
        }
    )
    request = Request(
        f"{HELSINKI_WFS_ENDPOINT}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "SunfinderHelsinki/1.0 (building map)",
        },
    )
    with urlopen(request, timeout=HELSINKI_WFS_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed City of Helsinki endpoint
        payload = json.loads(response.read().decode("utf-8"))
    return wfs_to_features(payload)


def fetch_current_weather() -> dict[str, Any]:
    """Get a lightweight, no-key current sky estimate for Helsinki.

    Open-Meteo's current conditions are model data, not an on-site cloud
    observation. We expose that limitation directly in the returned copy.
    """
    query = urlencode(
        {
            "latitude": HELSINKI_LATITUDE,
            "longitude": HELSINKI_LONGITUDE,
            "current": "cloud_cover,weather_code,is_day",
            "hourly": (
                "cloud_cover,cloud_cover_low,weather_code,"
                "direct_radiation,direct_normal_irradiance,precipitation_probability"
            ),
            "past_hours": 1,
            "forecast_hours": 3,
            "timezone": "Europe/Helsinki",
        }
    )
    request = Request(
        f"{OPEN_METEO_ENDPOINT}?{query}",
        headers={"Accept": "application/json", "User-Agent": "SunfinderHelsinki/1.0 (local map prototype)"},
    )
    with urlopen(request, timeout=10) as response:  # noqa: S310 - fixed public weather endpoint
        payload = json.loads(response.read().decode("utf-8"))
    current = payload["current"]
    cloud_cover = max(0, min(100, int(round(float(current["cloud_cover"])))))
    weather_code = int(current["weather_code"])
    weather = weather_summary(
        cloud_cover=cloud_cover,
        weather_code=weather_code,
        observed_at=str(current.get("time", "")),
        is_day=bool(current.get("is_day", 0)),
    )
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        hourly = {}
    now = datetime.now(HELSINKI_TIME_ZONE)
    try:
        weather["nowcast"] = direct_sun_nowcast(
            now=now,
            current=current,
            hourly=hourly,
            solar_altitude_at=lambda candidate: solar_position(candidate.astimezone(UTC))["altitude"],
        )
    except (KeyError, TypeError, ValueError) as error:
        print(f"Direct-sun nowcast failed: {error}")
        weather["nowcast"] = unavailable_direct_sun_nowcast("Forecast data did not include a usable direct-sun estimate.")
    weather["source"] = "Open-Meteo weather model and hourly forecast"
    return weather


def weather_summary(
    *, cloud_cover: int, weather_code: int, observed_at: str, is_day: bool
) -> dict[str, Any]:
    """Turn cloud cover into deliberately conservative shadow messaging."""
    if weather_code in {45, 48}:
        label, visibility, opacity, note = (
            "Foggy",
            "unlikely",
            0.0,
            "Fog scatters direct light, so sharp building shadows are unlikely to be visible.",
        )
    elif weather_code >= 51:
        label, visibility, opacity, note = (
            weather_label(weather_code),
            "unlikely",
            0.0,
            "Precipitation or dense cloud makes direct building shadows unlikely to be visible.",
        )
    elif cloud_cover >= 85:
        label, visibility, opacity, note = (
            "Overcast",
            "unlikely",
            0.0,
            "Overcast cloud usually removes visible direct building shadows.",
        )
    elif cloud_cover >= 65:
        label, visibility, opacity, note = (
            "Cloudy",
            "very soft",
            0.15,
            "Cloud cover should make direct shadows very soft or intermittent.",
        )
    elif cloud_cover >= 35:
        label, visibility, opacity, note = (
            "Partly cloudy",
            "intermittent",
            0.32,
            "Shadows may appear briefly when direct sun breaks through cloud.",
        )
    elif cloud_cover >= 10:
        label, visibility, opacity, note = (
            "Mostly clear",
            "soft",
            0.50,
            "Some cloud may soften the otherwise projected direct shadows.",
        )
    else:
        label, visibility, opacity, note = (
            "Clear sky",
            "defined",
            0.58,
            "Direct shadows should be reasonably well defined where the sun reaches the ground.",
        )
    return {
        "available": True,
        "observed_at": observed_at,
        "cloud_cover": cloud_cover,
        "weather_code": weather_code,
        "is_day": is_day,
        "label": label,
        "shadow_visibility": visibility,
        "shadow_opacity": opacity,
        "note": note,
        "source": "Open-Meteo weather model",
    }


def unavailable_weather() -> dict[str, Any]:
    return {
        "available": False,
        "observed_at": None,
        "cloud_cover": None,
        "weather_code": None,
        "is_day": None,
        "label": "Cloud cover unavailable",
        "shadow_visibility": "potential",
        "shadow_opacity": 0.58,
        "note": "Showing clear-sky potential. Local clouds can soften or remove real shadows.",
        "source": "Weather data unavailable",
        "nowcast": unavailable_direct_sun_nowcast("Weather data is unavailable, so there is no direct-sun estimate."),
    }


def custom_time_weather() -> dict[str, Any]:
    return {
        "available": False,
        "observed_at": None,
        "cloud_cover": None,
        "weather_code": None,
        "is_day": None,
        "label": "Potential clear-sky shadows",
        "shadow_visibility": "potential",
        "shadow_opacity": 0.58,
        "note": "Current cloud cover only applies to live time. This selected time shows clear-sky potential.",
        "source": "Clear-sky geometry",
        "nowcast": unavailable_direct_sun_nowcast("Direct-sun nowcasts are only available for live time."),
    }


def weather_label(weather_code: int) -> str:
    if weather_code in {51, 53, 55, 56, 57}:
        return "Drizzle"
    if weather_code in {61, 63, 65, 66, 67, 80, 81, 82}:
        return "Rain / showers"
    if weather_code in {71, 73, 75, 77, 85, 86}:
        return "Snow showers"
    if weather_code in {95, 96, 99}:
        return "Thunderstorms"
    return "Poor sky conditions"


def wfs_to_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    raw_features = payload.get("features", [])
    if not isinstance(raw_features, list):
        return features
    for item in raw_features:
        if not isinstance(item, dict):
            continue
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            continue
        polygons = geometry.get("coordinates")
        if geometry.get("type") == "Polygon":
            polygons = [polygons]
        if geometry.get("type") not in {"Polygon", "MultiPolygon"} or not isinstance(polygons, list):
            continue

        properties = item.get("properties")
        if not isinstance(properties, dict):
            properties = {}
        for polygon_index, polygon in enumerate(polygons):
            if not isinstance(polygon, list) or not polygon:
                continue
            ring = polygon[0]
            if not isinstance(ring, list) or len(ring) < 3:
                continue
            normalised_ring = [[point[0], point[1]] for point in ring if isinstance(point, list) and len(point) >= 2]
            if len(normalised_ring) < 3 or any(
                not all(isinstance(value, (int, float)) for value in point) for point in normalised_ring
            ):
                continue
            if normalised_ring[0] != normalised_ring[-1]:
                normalised_ring.append(normalised_ring[0].copy())
            item_id = item.get("id") or properties.get("id")
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "id": f"{item_id}:{polygon_index}",
                        "name": wfs_building_name(properties),
                        "height": height_from_wfs_properties(properties),
                        "source": "City of Helsinki building register",
                    },
                    "geometry": {"type": "Polygon", "coordinates": [normalised_ring]},
                }
            )
            if len(features) >= MAX_BUILDINGS_PER_REQUEST:
                return features
    return features


def height_from_wfs_properties(properties: dict[str, Any]) -> float:
    try:
        levels = float(properties.get("i_kerrlkm", ""))
    except (TypeError, ValueError):
        levels = 0
    if levels > 0:
        return max(4.0, min(140.0, levels * 3.25 + 1.5))
    building_type = str(properties.get("tyyppi", "")).lower()
    if "kirkko" in building_type:
        return 28.0
    if "liike" in building_type:
        return 20.0
    return DEFAULT_BUILDING_HEIGHT


def wfs_building_name(properties: dict[str, Any]) -> str:
    street = str(properties.get("katunimi_suomi") or "").strip()
    number = str(properties.get("osoitenumero") or "").strip()
    if street and number:
        return f"{street} {number}"
    if street:
        return street
    building_type = str(properties.get("tyyppi") or "").strip()
    return building_type or "Helsinki building"


def fallback_features_in(bounds: Bounds) -> list[dict[str, Any]]:
    def overlaps(feature_item: dict[str, Any]) -> bool:
        ring = feature_item["geometry"]["coordinates"][0]
        longitudes = [coordinate[0] for coordinate in ring]
        latitudes = [coordinate[1] for coordinate in ring]
        return not (
            max(longitudes) < bounds.west
            or min(longitudes) > bounds.east
            or max(latitudes) < bounds.south
            or min(latitudes) > bounds.north
        )

    return [feature_item for feature_item in FALLBACK_FEATURES if overlaps(feature_item)]


def solar_position(timestamp: datetime) -> dict[str, float]:
    """Return NOAA-style solar altitude and azimuth for Helsinki."""
    local_time = timestamp.astimezone(HELSINKI_TIME_ZONE)
    day_of_year = local_time.timetuple().tm_yday
    days_in_year = 366 if is_leap_year(local_time.year) else 365
    hour = local_time.hour + local_time.minute / 60 + local_time.second / 3_600
    gamma = 2 * math.pi / days_in_year * (day_of_year - 1 + (hour - 12) / 24)
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    utc_offset_minutes = local_time.utcoffset().total_seconds() / 60
    true_solar_time = (hour * 60 + equation_of_time + 4 * HELSINKI_LONGITUDE - utc_offset_minutes) % 1_440
    hour_angle = math.radians(true_solar_time / 4 - 180)
    latitude_radians = math.radians(HELSINKI_LATITUDE)
    cosine_zenith = max(
        -1,
        min(
            1,
            math.sin(latitude_radians) * math.sin(declination)
            + math.cos(latitude_radians) * math.cos(declination) * math.cos(hour_angle),
        ),
    )
    zenith = math.acos(cosine_zenith)
    altitude = 90 - math.degrees(zenith)
    azimuth = (
        math.degrees(
            math.atan2(
                math.sin(hour_angle),
                math.cos(hour_angle) * math.sin(latitude_radians)
                - math.tan(declination) * math.cos(latitude_radians),
            )
        )
        + 180
    ) % 360
    return {
        "altitude": round(altitude, 5),
        "azimuth": round(azimuth, 5),
        "declination": round(math.degrees(declination), 5),
    }


@lru_cache(maxsize=800)
def sun_times_for_date(local_date: date) -> dict[str, str]:
    start = datetime.combine(local_date, datetime.min.time(), tzinfo=HELSINKI_TIME_ZONE)
    previous_altitude = solar_position(start.astimezone(UTC))["altitude"]
    sunrise: str | None = None
    sunset: str | None = None
    for minute in range(4, 1_445, 4):
        candidate = start + timedelta(minutes=minute)
        altitude = solar_position(candidate.astimezone(UTC))["altitude"]
        if previous_altitude < -0.83 <= altitude and sunrise is None:
            sunrise = candidate.strftime("%H:%M")
        if previous_altitude >= -0.83 > altitude and sunset is None:
            sunset = candidate.strftime("%H:%M")
        previous_altitude = altitude
    return {"sunrise": sunrise or "No sunrise", "sunset": sunset or "No sunset"}


def create_shadows(features: list[dict[str, Any]], sun: dict[str, float]) -> list[dict[str, Any]]:
    if sun["altitude"] <= 0 or sun["altitude"] > 88:
        return []
    shadows: list[dict[str, Any]] = []
    for building in features:
        shadow = create_shadow(building, sun)
        if shadow:
            shadows.append(shadow)
    return shadows


def create_shadow(building: dict[str, Any], sun: dict[str, float]) -> dict[str, Any] | None:
    try:
        ring = building["geometry"]["coordinates"][0]
    except (KeyError, IndexError, TypeError):
        return None
    if len(ring) < 4:
        return None
    height = normalise_height(building.get("properties", {}).get("height"))
    distance = min(MAX_SHADOW_METERS, height / math.tan(math.radians(sun["altitude"])))
    shadow_bearing = (sun["azimuth"] + 180) % 360
    footprint = ring[:-1]
    projected = [shift_coordinate(point, distance, shadow_bearing) for point in footprint]
    hull = convex_hull([*footprint, *projected])
    if len(hull) < 3:
        return None
    return {
        "type": "Feature",
        "properties": {
            "building": building.get("properties", {}).get("name", "Building"),
            "height": height,
            "length": round(distance),
        },
        "geometry": {"type": "Polygon", "coordinates": [[*hull, hull[0]]]},
    }


def shift_coordinate(point: list[float], distance: float, bearing: float) -> list[float]:
    longitude, latitude = point
    radians = math.radians(bearing)
    north = math.cos(radians) * distance
    east = math.sin(radians) * distance
    return [
        longitude + east / (METERS_PER_DEGREE_LAT * math.cos(math.radians(latitude))),
        latitude + north / METERS_PER_DEGREE_LAT,
    ]


def convex_hull(points: list[list[float]]) -> list[list[float]]:
    """Andrew's monotone-chain hull, sufficient for a projected footprint."""
    sorted_points = sorted(points, key=lambda point: (point[0], point[1]))
    if len(sorted_points) <= 1:
        return sorted_points

    def cross(origin: list[float], point_a: list[float], point_b: list[float]) -> float:
        return (
            (point_a[0] - origin[0]) * (point_b[1] - origin[1])
            - (point_a[1] - origin[1]) * (point_b[0] - origin[0])
        )

    lower: list[list[float]] = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[list[float]] = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def normalise_height(value: Any) -> float:
    try:
        height = float(value)
    except (TypeError, ValueError):
        return DEFAULT_BUILDING_HEIGHT
    return height if height > 0 else DEFAULT_BUILDING_HEIGHT


def is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def feature_collection(features: list[dict[str, Any]]) -> dict[str, Any]:
    return {"type": "FeatureCollection", "features": features}


app = FastAPI(
    title="Sunfinder Helsinki API",
    version="1.0.0",
    description="Solar, sky, and building fallback API for Helsinki.",
)
app.add_middleware(GZipMiddleware, minimum_size=1_000)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "time_zone": "Europe/Helsinki"}


@app.get("/api/solar")
async def solar(at: str | None = Query(default=None, description="ISO-8601 timestamp")) -> dict[str, Any]:
    timestamp = parse_timestamp(at)
    local_time = timestamp.astimezone(HELSINKI_TIME_ZONE)
    return {
        "at": timestamp.isoformat().replace("+00:00", "Z"),
        "local_time": local_time.isoformat(),
        "solar": solar_position(timestamp),
        "sun_times": sun_times_for_date(local_time.date()),
    }


async def current_conditions(timestamp: datetime, live: bool) -> dict[str, Any]:
    """Return fast sun/sky data independently from the slower building lookup."""
    is_live_time = live and abs((datetime.now(UTC) - timestamp).total_seconds()) <= 15 * 60
    if is_live_time:
        weather, weather_cached = await asyncio.to_thread(weather_store.get)
    else:
        weather = custom_time_weather()
        weather_cached = False
    local_time = timestamp.astimezone(HELSINKI_TIME_ZONE)
    return {
        "at": timestamp.isoformat().replace("+00:00", "Z"),
        "local_time": local_time.isoformat(),
        "solar": solar_position(timestamp),
        "sun_times": sun_times_for_date(local_time.date()),
        "weather": {**weather, "applies_to_selected_time": is_live_time, "cached": weather_cached},
    }


@app.get("/api/conditions")
async def conditions(
    at: str | None = Query(default=None, description="ISO-8601 timestamp"),
    live: bool = Query(default=True, description="Whether current weather should be applied"),
) -> dict[str, Any]:
    """Return current sky state without waiting for building footprints."""
    return await current_conditions(parse_timestamp(at), live)


@app.get("/api/sun-planner/status")
async def sun_planner_status() -> dict[str, Any]:
    """Report whether the optional local Ollama planner is ready for this process."""
    if not assistant_settings.enabled:
        return {
            "enabled": False,
            "ready": False,
            "reason": "The local outing planner is disabled on this server.",
        }
    try:
        installed_models = await asyncio.to_thread(assistant_client.available_models)
    except OllamaUnavailableError:
        return {
            "enabled": True,
            "ready": False,
            "reason": "Start Ollama locally to use the outing planner.",
        }
    required_models = (assistant_settings.chat_model, assistant_settings.embedding_model)
    missing_models = [model for model in required_models if model not in installed_models]
    if missing_models:
        return {
            "enabled": True,
            "ready": False,
            "reason": "Pull the local planner models before using it.",
            "missing_models": missing_models,
        }
    return {
        "enabled": True,
        "ready": True,
        "catalogue_size": len(venue_catalogue),
        "chat_model": assistant_settings.chat_model,
        "embedding_model": assistant_settings.embedding_model,
    }


@app.post("/api/sun-plans")
async def sun_plans(request: SunPlanRequest) -> dict[str, Any]:
    """Recommend nearby curated terraces using local LLM parsing and map geometry."""
    if not assistant_settings.enabled:
        raise HTTPException(status_code=404, detail="The local outing planner is disabled on this server")

    status = await sun_planner_status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status["reason"])

    selected_time = planner_timestamp(request.selected_time)
    if not is_in_helsinki_region(request.map_latitude, request.map_longitude):
        raise HTTPException(status_code=422, detail="Choose a point in the Helsinki map before planning an outing")

    try:
        intent = await asyncio.to_thread(
            assistant_client.structured_intent,
            message=request.message,
            selected_time=selected_time.astimezone(HELSINKI_TIME_ZONE),
            current_time=datetime.now(HELSINKI_TIME_ZONE),
        )
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    planned_time = planner_timestamp(intent.requested_time or selected_time)
    anchor = {
        "name": "Current map view",
        "detail": "Map centre",
        "latitude": request.map_latitude,
        "longitude": request.map_longitude,
    }
    if intent.anchor_query:
        place_results, place_source, _ = await asyncio.to_thread(place_search_store.get, intent.anchor_query)
        if not place_results:
            if place_source == "unavailable":
                raise HTTPException(status_code=503, detail="Place search is unavailable. Try again shortly.")
            raise HTTPException(status_code=422, detail=f"I could not find {intent.anchor_query} in Helsinki")
        anchor = place_results[0]

    nearby_venues = venues_near(venue_catalogue, anchor["latitude"], anchor["longitude"])
    if not nearby_venues:
        raise HTTPException(status_code=422, detail="No curated terrace or café is within 2 km of that spot yet")

    bounds = planner_bounds(
        anchor["latitude"],
        anchor["longitude"],
        [(venue.latitude, venue.longitude) for venue, _ in nearby_venues],
    )
    building_result, condition_data = await asyncio.gather(
        asyncio.to_thread(building_store.get, bounds),
        current_conditions(planned_time, live=True),
    )
    building_features, building_source, building_cached = building_result
    building_geometry_available = building_source == "helsinki-wfs"
    shadow_samples: list[tuple[datetime, list[dict[str, Any]], bool]] = []
    for minutes_ahead in PLANNER_SAMPLE_MINUTES:
        sample_time = planned_time + timedelta(minutes=minutes_ahead)
        solar = solar_position(sample_time)
        daylight = solar["altitude"] > 0
        shadows = create_shadows(building_features, solar) if daylight and building_geometry_available else []
        shadow_samples.append((sample_time, shadows, daylight))

    recommendations = rank_venues_by_sun(
        nearby_venues,
        shadow_samples,
        building_geometry_available=building_geometry_available,
    )
    try:
        retrieved_documents = await asyncio.to_thread(venue_retriever.search, request.message)
    except OllamaUnavailableError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    documents_by_id = {document.venue_id: document for document in retrieved_documents}
    for recommendation in recommendations:
        venue = recommendation["venue"]
        if venue["id"] not in documents_by_id:
            documents_by_id[venue["id"]] = RetrievedVenueDocument(
                venue_id=venue["id"],
                score=1.0,
                text="\n".join(
                    (
                        venue["name"],
                        f"Area: {venue['area']}",
                        f"Outdoor note: {venue['terrace_note']}",
                    )
                ),
                source_label=venue["source"]["label"],
                source_url=venue["source"]["url"],
            )
    response_facts = planner_language_facts(
        anchor=anchor,
        planned_time=planned_time,
        recommendations=recommendations,
        building_geometry_available=building_geometry_available,
        building_source=building_source,
        weather=condition_data["weather"],
    )
    if not building_geometry_available:
        answer = deterministic_plan_answer(
            recommendations,
            building_geometry_available=building_geometry_available,
            weather=condition_data["weather"],
        )
    else:
        try:
            answer = await asyncio.to_thread(
                assistant_client.write_answer,
                request=request.message,
                facts=response_facts,
                retrieved_documents=list(documents_by_id.values()),
            )
        except OllamaUnavailableError:
            answer = deterministic_plan_answer(
                recommendations,
                building_geometry_available=building_geometry_available,
                weather=condition_data["weather"],
            )

    return {
        "answer": answer,
        "request": {
            "message": request.message,
            "anchor": anchor,
            "at": planned_time.isoformat(),
            "window_minutes": 60,
            "venue_kind": intent.venue_kind,
        },
        "recommendations": recommendations,
        "weather": planner_weather_summary(condition_data["weather"]),
        "retrieved_sources": [
            {
                "venue_id": document.venue_id,
                "score": document.score,
                "source": {"label": document.source_label, "url": document.source_url},
            }
            for document in retrieved_documents
        ],
        "meta": {
            "building_source": building_source,
            "building_cached": building_cached,
            "building_geometry_available": building_geometry_available,
            "catalogue_size": len(venue_catalogue),
        },
    }


@app.get("/api/places")
async def places(
    q: str = Query(description="A Helsinki venue, park, landmark, or address", max_length=100),
) -> dict[str, Any]:
    """Find a submitted place query without exposing the geocoder to the browser."""
    query = " ".join(q.split())
    if len(normalise_place_query(query)) < 2:
        raise HTTPException(status_code=422, detail="Type at least two letters to search for a place")
    results, source, from_cache = await asyncio.to_thread(place_search_store.get, query)
    return {
        "results": results,
        "meta": {
            "cached": from_cache,
            "available": source != "unavailable",
        },
    }


@app.get("/api/place-suggestions")
async def place_suggestions(
    q: str = Query(description="At least two letters from a Helsinki place name", max_length=100),
) -> dict[str, Any]:
    """Return a short, debounced-friendly list of Helsinki place suggestions."""
    query = " ".join(q.split())
    if len(normalise_place_query(query)) < 2:
        return {"results": [], "meta": {"cached": False, "available": True}}
    results, source, from_cache = await asyncio.to_thread(place_suggestion_store.get, query)
    return {
        "results": results,
        "meta": {
            "cached": from_cache,
            "available": source != "unavailable",
        },
    }


@app.get("/api/buildings")
async def buildings(
    bbox: str = Query(description="south,west,north,east"),
    retry_buildings: bool = Query(default=False, description="Bypass a cached fallback building response"),
) -> dict[str, Any]:
    """Return a Helsinki WFS building fallback for one map viewport."""
    bounds = parse_bounds(bbox)
    features, source, from_cache = await asyncio.to_thread(
        building_store.get,
        bounds,
        retry_failed=retry_buildings,
    )
    return {
        "buildings": feature_collection(features),
        "meta": {
            "building_count": len(features),
            "source": source,
            "cached": from_cache,
        },
    }


@app.get("/api/scene")
async def scene(
    bbox: str = Query(description="south,west,north,east"),
    at: str | None = Query(default=None, description="ISO-8601 timestamp"),
    live: bool = Query(default=True, description="Whether current weather should be applied"),
    retry_buildings: bool = Query(default=False, description="Retry a failed building-data lookup"),
    include_buildings: bool = Query(default=True, description="Whether to include unchanged building geometry"),
) -> dict[str, Any]:
    """Return data for one map viewport at one point in Helsinki time."""
    bounds = parse_bounds(bbox)
    timestamp = parse_timestamp(at)
    building_result, condition_data = await asyncio.gather(
        asyncio.to_thread(building_store.get, bounds, retry_failed=retry_buildings),
        current_conditions(timestamp, live),
    )
    features, source, from_cache = building_result
    shadows = create_shadows(features, condition_data["solar"])
    payload = {
        **condition_data,
        "shadows": feature_collection(shadows),
        "meta": {
            "building_count": len(features),
            "shadow_count": len(shadows),
            "source": source,
            "cached": from_cache,
            "buildings_included": include_buildings,
        },
    }
    if include_buildings:
        payload["buildings"] = feature_collection(features)
    return payload


# Mounting static files last means /api routes always take precedence.
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
