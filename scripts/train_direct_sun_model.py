#!/usr/bin/env python3
"""Train Sunfinder's dependency-free direct-sun logistic-regression model.

The target is the WMO sunshine definition: direct normal irradiance greater
than 120 W/m². The script uses Open-Meteo's historical weather archive for a
repeatable first calibration. Replacing its training data with FMI station
observations is the next accuracy upgrade, not a required runtime dependency.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import HELSINKI_LATITUDE, HELSINKI_LONGITUDE, solar_position
from backend.nowcast import (
    MODEL_FEATURE_NAMES,
    TRAINED_MODEL_PATH,
    WMO_SUNSHINE_DNI_THRESHOLD,
    direct_radiation_fraction,
    logistic_features,
    sigmoid,
)


ARCHIVE_ENDPOINT = "https://archive-api.open-meteo.com/v1/archive"
HELSINKI_TIME_ZONE = ZoneInfo("Europe/Helsinki")


def fetch_training_rows(start_date: date, end_date: date) -> list[tuple[list[float], int]]:
    query = urlencode(
        {
            "latitude": HELSINKI_LATITUDE,
            "longitude": HELSINKI_LONGITUDE,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "hourly": "cloud_cover,cloud_cover_low,weather_code,precipitation,direct_radiation,direct_normal_irradiance",
            "timezone": "Europe/Helsinki",
        }
    )
    request = Request(
        f"{ARCHIVE_ENDPOINT}?{query}",
        headers={"Accept": "application/json", "User-Agent": "SunfinderHelsinki/1.0 (model training)"},
    )
    with urlopen(request, timeout=60) as response:  # noqa: S310 - fixed public weather archive
        payload = json.loads(response.read().decode("utf-8"))
    hourly = payload["hourly"]

    rows: list[tuple[list[float], int]] = []
    for index, raw_time in enumerate(hourly.get("time", [])):
        try:
            timestamp = datetime.fromisoformat(raw_time).replace(tzinfo=HELSINKI_TIME_ZONE)
            altitude = solar_position(timestamp.astimezone(UTC))["altitude"]
            cloud = percent(hourly["cloud_cover"][index])
            low_cloud = percent(hourly["cloud_cover_low"][index])
            weather_code = int(hourly["weather_code"][index])
            precipitation = precipitation_signal(hourly["precipitation"][index])
            direct_radiation = hourly["direct_radiation"][index]
            direct_normal_irradiance = float(hourly["direct_normal_irradiance"][index])
        except (IndexError, KeyError, TypeError, ValueError):
            continue
        if altitude <= 0 or direct_normal_irradiance < 0:
            continue
        features = logistic_features(
            cloud=cloud,
            low_cloud=low_cloud,
            precipitation=precipitation,
            weather_code=weather_code,
            radiation_signal=direct_radiation_fraction(
                sun_altitude=altitude,
                direct_radiation=direct_radiation,
                direct_normal_irradiance=None,
                fallback=1 - cloud,
            ),
            sun_altitude=altitude,
        )
        target = int(direct_normal_irradiance >= WMO_SUNSHINE_DNI_THRESHOLD)
        rows.append((features, target))
    if not rows:
        raise RuntimeError("The weather archive returned no usable daylight rows.")
    return rows


def percent(value: Any) -> float:
    return max(0.0, min(1.0, float(value) / 100))


def precipitation_signal(value: Any) -> float:
    # Archive precipitation is mm in the preceding hour. At runtime this maps
    # to precipitation probability, which is also normalised to [0, 1].
    return max(0.0, min(1.0, float(value) / 0.3))


def train_logistic(rows: list[tuple[list[float], int]], *, iterations: int, learning_rate: float) -> list[float]:
    weights = [0.0] * (len(MODEL_FEATURE_NAMES) + 1)
    positive_rate = sum(target for _, target in rows) / len(rows)
    weights[0] = math.log(max(0.001, positive_rate) / max(0.001, 1 - positive_rate))
    regularisation = 0.001

    for _ in range(iterations):
        gradient = [0.0] * len(weights)
        for features, target in rows:
            prediction = sigmoid(weights[0] + sum(weight * feature for weight, feature in zip(weights[1:], features)))
            error = prediction - target
            gradient[0] += error
            for index, feature in enumerate(features, start=1):
                gradient[index] += error * feature
        sample_count = len(rows)
        weights[0] -= learning_rate * gradient[0] / sample_count
        for index in range(1, len(weights)):
            weights[index] -= learning_rate * (gradient[index] / sample_count + regularisation * weights[index])
    return weights


def metrics(rows: list[tuple[list[float], int]], weights: list[float]) -> dict[str, float]:
    predictions = [sigmoid(weights[0] + sum(weight * feature for weight, feature in zip(weights[1:], features))) for features, _ in rows]
    targets = [target for _, target in rows]
    brier = sum((prediction - target) ** 2 for prediction, target in zip(predictions, targets)) / len(rows)
    accuracy = sum((prediction >= 0.5) == bool(target) for prediction, target in zip(predictions, targets)) / len(rows)
    return {"brier_score": round(brier, 4), "accuracy_at_50_percent": round(accuracy, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=365, help="Historical days to fetch (default: 365).")
    parser.add_argument("--iterations", type=int, default=1800, help="Gradient-descent iterations (default: 1800).")
    parser.add_argument("--learning-rate", type=float, default=0.7, help="Gradient-descent learning rate (default: 0.7).")
    parser.add_argument("--output", type=Path, default=TRAINED_MODEL_PATH, help="Output JSON model path.")
    args = parser.parse_args()
    if args.days < 30:
        parser.error("--days must be at least 30")

    end_date = datetime.now(UTC).date() - timedelta(days=6)
    start_date = end_date - timedelta(days=args.days - 1)
    rows = fetch_training_rows(start_date, end_date)
    split = max(1, round(len(rows) * 0.8))
    training_rows, validation_rows = rows[:split], rows[split:]
    weights = train_logistic(training_rows, iterations=args.iterations, learning_rate=args.learning_rate)
    model = {
        "version": "helsinki-archive-logistic-v1",
        "kind": "logistic regression",
        "definition": f"Direct normal irradiance above {WMO_SUNSHINE_DNI_THRESHOLD} W/m².",
        "features": list(MODEL_FEATURE_NAMES),
        "weights": [round(weight, 10) for weight in weights],
        "training": {
            "source": "Open-Meteo Historical Weather API reanalysis",
            "location": "Helsinki, Finland",
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daylight_samples": len(rows),
            "validation": metrics(validation_rows or training_rows, weights),
            "generated_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output} from {len(training_rows)} training and {len(validation_rows)} validation rows.")
    print(json.dumps(model["training"]["validation"], indent=2))


if __name__ == "__main__":
    main()
