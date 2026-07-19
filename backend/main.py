"""FastAPI service for Helsinki building shadows.

The browser is responsible for interaction and WebGL rendering.  This module
owns the data and solar geometry: it fetches/caches building footprints,
calculates Helsinki's sun position, and returns ready-to-render shadow shapes.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
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


APP_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = APP_ROOT / "frontend"
HELSINKI_LATITUDE = 60.1699
HELSINKI_LONGITUDE = 24.9384
HELSINKI_TIME_ZONE = ZoneInfo("Europe/Helsinki")
METERS_PER_DEGREE_LAT = 111_320
DEFAULT_BUILDING_HEIGHT = 15.0
MAX_SHADOW_METERS = 560.0
MAX_BUILDINGS_PER_REQUEST = 3_000
OVERPASS_ENDPOINTS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"
WEATHER_CACHE_SECONDS = 5 * 60
OVERPASS_HTTP_TIMEOUT_SECONDS = 10
FAILED_BUILDING_CACHE_SECONDS = 15

# This bounds check keeps the public Overpass request firmly scoped to the
# product's Helsinki use case. It is intentionally broader than city centre.
HELSINKI_REGION = (59.95, 24.60, 60.38, 25.40)  # south, west, north, east
MAX_QUERY_LATITUDE_SPAN = 0.09
MAX_QUERY_LONGITUDE_SPAN = 0.14


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


class BuildingStore:
    """A deliberately small, in-memory cache around OpenStreetMap Overpass."""

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
            # A time scrub can make several identical requests before Overpass
            # responds. Let them share the same fetch instead of piling up.
            in_flight.wait()

        try:
            try:
                features = fetch_openstreetmap_buildings(bounds)
                if len(features) < 4:
                    raise ValueError("OpenStreetMap returned too few building footprints")
                source = "openstreetmap"
                ttl_seconds = 12 * 60 * 60
            except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as error:
                # The map remains useful when a public API is rate-limited or offline.
                print(f"OpenStreetMap building fetch failed: {error}")
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


def fetch_openstreetmap_buildings(bounds: Bounds) -> list[dict[str, Any]]:
    query = (
        "[out:json][timeout:8];"
        f'way["building"]({bounds.south:.5f},{bounds.west:.5f},{bounds.north:.5f},{bounds.east:.5f});'
        "out tags geom;"
    )
    body = urlencode({"data": query}).encode("utf-8")
    last_error: Exception | None = None
    for endpoint in OVERPASS_ENDPOINTS:
        request = Request(
            endpoint,
            data=body,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "SunfinderHelsinki/1.0 (local map prototype)",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=OVERPASS_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310 - fixed public OSM endpoint
                payload = json.loads(response.read().decode("utf-8"))
            return overpass_to_features(payload)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            print(f"Overpass endpoint unavailable ({endpoint}): {error}")
    if last_error:
        raise last_error
    raise RuntimeError("No Overpass endpoints are configured")


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


def overpass_to_features(payload: dict[str, Any]) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        if element.get("type") != "way" or not isinstance(element.get("geometry"), list):
            continue
        ring = [[node.get("lon"), node.get("lat")] for node in element["geometry"]]
        if len(ring) < 3 or any(not all(isinstance(value, (int, float)) for value in point) for point in ring):
            continue
        if ring[0] != ring[-1]:
            ring.append(ring[0].copy())
        tags = element.get("tags") or {}
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": element.get("id"),
                    "name": tags.get("name") or tags.get("addr:street") or "Helsinki building",
                    "height": height_from_tags(tags),
                    "source": "OpenStreetMap",
                },
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
        if len(features) >= MAX_BUILDINGS_PER_REQUEST:
            break
    return features


def height_from_tags(tags: dict[str, str]) -> float:
    value = str(tags.get("height", "")).replace(",", ".")
    match = re.search(r"-?\d+(?:\.\d+)?", value)
    if match:
        height = float(match.group())
        if 2 < height < 400:
            return height
    try:
        levels = float(tags.get("building:levels", ""))
    except (TypeError, ValueError):
        levels = 0
    if levels > 0:
        return max(4.0, min(140.0, levels * 3.25 + 1.5))
    if tags.get("building") in {"church", "cathedral"}:
        return 28.0
    if tags.get("building") == "commercial":
        return 20.0
    return DEFAULT_BUILDING_HEIGHT


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
    description="Server-side solar and building-shadow calculations for Helsinki.",
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
