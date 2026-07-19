"""Short-horizon direct-sun estimates from forecast weather inputs.

This is intentionally a transparent baseline, not a claim that a local camera
has observed sunlight. It combines the weather model's cloud and radiation
fields into a probability that direct sun reaches an *open* point in Helsinki.
Building obstruction remains the map's geometric calculation.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping


DIRECT_SUN_NOWCAST_VERSION = "weather-baseline-v1"
DIRECT_SUN_WINDOW_MINUTES = 60
NOWCAST_HORIZONS_MINUTES = (0, 30, 60)
WMO_SUNSHINE_DNI_THRESHOLD = 120
DIRECT_SUN_NOWCAST_SCOPE = (
    "One city level estimate for an open point, based on a fixed Helsinki forecast location. "
    "It is not a separate forecast for each neighbourhood."
)
MODEL_FEATURE_NAMES = (
    "cloud_cover",
    "low_cloud_cover",
    "cloud_low_interaction",
    "precipitation_signal",
    "precipitation_weather",
    "direct_radiation_fraction",
    "sun_altitude_sine",
    "season_sine",
    "season_cosine",
)
TRAINED_MODEL_PATH = Path(__file__).with_name("model_data") / "direct_sun_logistic.json"


def direct_sun_nowcast(
    *,
    now: datetime,
    current: Mapping[str, Any],
    hourly: Mapping[str, Any],
    solar_altitude_at: Callable[[datetime], float],
) -> dict[str, Any]:
    """Estimate direct sun over the next hour.

    The hourly weather values are nearest-hour forecast values. The current
    observation is used at zero minutes so the display reacts to the newest
    available cloud estimate. The result is deliberately phrased as a
    probability at an open point: it does not replace local building shadows.
    """
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")

    samples: list[dict[str, Any]] = []
    for minutes_ahead in NOWCAST_HORIZONS_MINUTES:
        target = now + timedelta(minutes=minutes_ahead)
        forecast = hourly_conditions_near(hourly, target) or {}
        if minutes_ahead == 0:
            forecast = {**forecast, **current}
        altitude = solar_altitude_at(target)
        probability = direct_sun_probability(
            sun_altitude=altitude,
            day_of_year=target.timetuple().tm_yday,
            cloud_cover=forecast.get("cloud_cover"),
            low_cloud_cover=forecast.get("cloud_cover_low"),
            weather_code=forecast.get("weather_code"),
            precipitation_probability=forecast.get("precipitation_probability"),
            direct_radiation=forecast.get("direct_radiation"),
            direct_normal_irradiance=forecast.get("direct_normal_irradiance"),
        )
        samples.append(
            {
                "minutes_ahead": minutes_ahead,
                "forecast_at": target.isoformat(),
                "probability": probability,
                "sun_altitude": round(altitude, 1),
                "cloud_cover": integer_or_none(forecast.get("cloud_cover")),
            }
        )

    probability = round(sum(sample["probability"] for sample in samples) / len(samples))
    return {
        "available": True,
        "probability": probability,
        "window_minutes": DIRECT_SUN_WINDOW_MINUTES,
        "label": probability_label(probability),
        "samples": samples,
        "model": direct_sun_model_metadata(),
        "scope": DIRECT_SUN_NOWCAST_SCOPE,
        "note": "Estimate for an open point. Nearby buildings, trees, and a changing local sky can still block direct sun.",
    }


def unavailable_direct_sun_nowcast(reason: str) -> dict[str, Any]:
    return {
        "available": False,
        "probability": None,
        "window_minutes": DIRECT_SUN_WINDOW_MINUTES,
        "label": "Direct-sun estimate unavailable",
        "samples": [],
        "model": direct_sun_model_metadata(),
        "scope": DIRECT_SUN_NOWCAST_SCOPE,
        "note": reason,
    }


def direct_sun_probability(
    *,
    sun_altitude: float,
    day_of_year: int | None = None,
    cloud_cover: Any,
    low_cloud_cover: Any,
    weather_code: Any,
    precipitation_probability: Any,
    direct_radiation: Any,
    direct_normal_irradiance: Any,
) -> int:
    """Return a conservative 0–100 direct-sun probability.

    A trained logistic-regression artifact is used when present. The fallback
    coefficients remain deliberately inspectable, so a deployed service still
    returns an honest estimate before a calibration artifact has been trained.
    """
    if not math.isfinite(sun_altitude) or sun_altitude <= 0:
        return 0

    cloud = fraction(cloud_cover, fallback=0.65)
    low_cloud = fraction(low_cloud_cover, fallback=cloud)
    precipitation = fraction(precipitation_probability, fallback=0.0)
    code = integer_or_none(weather_code)
    radiation_signal = direct_radiation_fraction(
        sun_altitude=sun_altitude,
        direct_radiation=direct_radiation,
        direct_normal_irradiance=None,
        fallback=1 - cloud,
    )
    trained_probability = trained_model_probability(
        cloud=cloud,
        low_cloud=low_cloud,
        precipitation=precipitation,
        weather_code=code,
        radiation_signal=radiation_signal,
        sun_altitude=sun_altitude,
        day_of_year=day_of_year,
    )
    if trained_probability is not None:
        return conservative_probability_cap(
            trained_probability,
            code,
            cloud=cloud,
            radiation_signal=radiation_signal,
        )

    radiation_signal = direct_radiation_fraction(
        sun_altitude=sun_altitude,
        direct_radiation=direct_radiation,
        direct_normal_irradiance=direct_normal_irradiance,
        fallback=radiation_signal,
    )

    # A small logistic model whose inputs are all available in the live
    # forecast. The values are deliberately conservative under low cloud.
    logit = 2.35 - 3.0 * cloud - 1.25 * low_cloud - 1.2 * precipitation
    logit += 1.6 * (radiation_signal - 0.5)
    probability = sigmoid(logit) * 100
    return conservative_probability_cap(probability, code, cloud=cloud, radiation_signal=radiation_signal)


def direct_sun_model_metadata() -> dict[str, Any]:
    model = load_trained_model()
    if model is None:
        return {
            "version": DIRECT_SUN_NOWCAST_VERSION,
            "kind": "weather-model baseline",
            "trained": False,
            "definition": f"Direct normal irradiance above {WMO_SUNSHINE_DNI_THRESHOLD} W/m².",
        }
    return {
        "version": model["version"],
        "kind": model["kind"],
        "trained": True,
        "definition": model["definition"],
        "training_source": model.get("training", {}).get("source"),
    }


@lru_cache(maxsize=1)
def load_trained_model() -> dict[str, Any] | None:
    """Load a checked-in, dependency-free logistic-regression artifact."""
    try:
        model = json.loads(TRAINED_MODEL_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(model, dict) or model.get("features") != list(MODEL_FEATURE_NAMES):
        return None
    weights = model.get("weights")
    if not isinstance(weights, list) or len(weights) != len(MODEL_FEATURE_NAMES) + 1:
        return None
    try:
        model["weights"] = [float(weight) for weight in weights]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(weight) for weight in model["weights"]):
        return None
    if not isinstance(model.get("version"), str) or not isinstance(model.get("kind"), str):
        return None
    if not isinstance(model.get("definition"), str):
        return None
    return model


def trained_model_probability(
    *,
    cloud: float,
    low_cloud: float,
    precipitation: float,
    weather_code: int | None,
    radiation_signal: float,
    sun_altitude: float,
    day_of_year: int | None,
) -> float | None:
    model = load_trained_model()
    if model is None:
        return None
    features = logistic_features(
        cloud=cloud,
        low_cloud=low_cloud,
        precipitation=precipitation,
        weather_code=weather_code,
        radiation_signal=radiation_signal,
        sun_altitude=sun_altitude,
        day_of_year=day_of_year,
    )
    weights = model["weights"]
    return sigmoid(weights[0] + sum(weight * feature for weight, feature in zip(weights[1:], features))) * 100


def logistic_features(
    *,
    cloud: float,
    low_cloud: float,
    precipitation: float,
    weather_code: int | None,
    radiation_signal: float,
    sun_altitude: float,
    day_of_year: int | None,
) -> list[float]:
    season_sine, season_cosine = seasonal_features(day_of_year)
    return [
        cloud,
        low_cloud,
        cloud * low_cloud,
        precipitation,
        1.0 if weather_code is not None and weather_code >= 51 else 0.0,
        radiation_signal,
        max(0.0, math.sin(math.radians(sun_altitude))),
        season_sine,
        season_cosine,
    ]


def seasonal_features(day_of_year: int | None) -> tuple[float, float]:
    """Encode the calendar as a continuous annual cycle, including leap years."""
    day = 172 if day_of_year is None else max(1, min(366, day_of_year))
    angle = 2 * math.pi * (day - 1) / 365.2425
    return math.sin(angle), math.cos(angle)


def conservative_probability_cap(
    probability: float,
    weather_code: int | None,
    *,
    cloud: float | None = None,
    radiation_signal: float | None = None,
) -> int:
    if weather_code in {45, 48}:  # fog
        probability = min(probability, 4)
    elif weather_code is not None and weather_code >= 51:  # drizzle, rain, snow, storms
        probability = min(probability, 12)
    # A grid-cell model can occasionally pair high cloud cover with a little
    # direct radiation. Keep the user-facing estimate conservative unless the
    # radiation signal itself supports a sun break.
    elif cloud is not None and radiation_signal is not None and cloud >= 0.85 and radiation_signal < 0.1:
        probability = min(probability, 8)
    elif cloud is not None and radiation_signal is not None and cloud >= 0.65 and radiation_signal < 0.2:
        probability = min(probability, 25)
    return max(0, min(100, round(probability)))


def sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-max(-12.0, min(12.0, value))))


def hourly_conditions_near(hourly: Mapping[str, Any], target: datetime) -> dict[str, Any] | None:
    """Read the closest forecast row from Open-Meteo's parallel arrays."""
    raw_times = hourly.get("time")
    if not isinstance(raw_times, list) or not raw_times:
        return None

    parsed: list[tuple[int, datetime]] = []
    for index, raw_time in enumerate(raw_times):
        if not isinstance(raw_time, str):
            continue
        try:
            candidate = datetime.fromisoformat(raw_time)
        except ValueError:
            continue
        if candidate.tzinfo is None:
            candidate = candidate.replace(tzinfo=target.tzinfo)
        else:
            candidate = candidate.astimezone(target.tzinfo)
        parsed.append((index, candidate))
    if not parsed:
        return None

    index, _ = min(parsed, key=lambda item: abs((item[1] - target).total_seconds()))
    return {
        key: values[index]
        for key, values in hourly.items()
        if key != "time" and isinstance(values, list) and index < len(values)
    }


def direct_radiation_fraction(
    *,
    sun_altitude: float,
    direct_radiation: Any,
    direct_normal_irradiance: Any,
    fallback: float,
) -> float:
    dni = positive_float(direct_normal_irradiance)
    if dni is not None:
        # 750 W/m² is a conservative bright-sky reference, not a clear-sky
        # radiation model. This merely normalises the provider's signal.
        return clamp(dni / 750)

    horizontal = positive_float(direct_radiation)
    if horizontal is not None:
        expected = max(25.0, 750 * math.sin(math.radians(sun_altitude)))
        return clamp(horizontal / expected)
    return clamp(fallback)


def probability_label(probability: int) -> str:
    if probability < 15:
        return "Direct sun unlikely"
    if probability < 40:
        return "Direct sun possible"
    if probability < 70:
        return "Direct sun plausible"
    return "Direct sun likely"


def fraction(value: Any, *, fallback: float) -> float:
    numeric = positive_float(value)
    if numeric is None:
        return fallback
    return clamp(numeric / 100)


def positive_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric >= 0 else None


def integer_or_none(value: Any) -> int | None:
    numeric = positive_float(value)
    return None if numeric is None else round(numeric)


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
