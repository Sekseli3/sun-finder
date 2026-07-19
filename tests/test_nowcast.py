from __future__ import annotations

import unittest
from datetime import datetime, timedelta
import json
from unittest.mock import patch
from zoneinfo import ZoneInfo

from backend.main import fetch_current_weather
from backend.nowcast import direct_sun_nowcast, direct_sun_probability, hourly_conditions_near, seasonal_features


HELSINKI = ZoneInfo("Europe/Helsinki")


class DirectSunNowcastTests(unittest.TestCase):
    def test_clear_sky_is_far_more_likely_than_overcast(self) -> None:
        clear = direct_sun_probability(
            sun_altitude=42,
            cloud_cover=0,
            low_cloud_cover=0,
            weather_code=0,
            precipitation_probability=0,
            direct_radiation=430,
            direct_normal_irradiance=720,
        )
        overcast = direct_sun_probability(
            sun_altitude=42,
            cloud_cover=100,
            low_cloud_cover=100,
            weather_code=3,
            precipitation_probability=0,
            direct_radiation=0,
            direct_normal_irradiance=0,
        )

        self.assertGreaterEqual(clear, 85)
        self.assertLessEqual(overcast, 15)
        self.assertGreater(clear, overcast)

    def test_sun_below_horizon_is_never_direct_sun(self) -> None:
        probability = direct_sun_probability(
            sun_altitude=-2,
            cloud_cover=0,
            low_cloud_cover=0,
            weather_code=0,
            precipitation_probability=0,
            direct_radiation=500,
            direct_normal_irradiance=800,
        )

        self.assertEqual(probability, 0)

    def test_nowcast_uses_current_conditions_and_hourly_horizons(self) -> None:
        now = datetime(2026, 7, 19, 12, 10, tzinfo=HELSINKI)
        hourly = {
            "time": ["2026-07-19T12:00", "2026-07-19T13:00"],
            "cloud_cover": [10, 95],
            "cloud_cover_low": [5, 90],
            "weather_code": [0, 3],
            "direct_radiation": [500, 10],
            "direct_normal_irradiance": [750, 20],
            "precipitation_probability": [0, 20],
        }
        nowcast = direct_sun_nowcast(
            now=now,
            current={"cloud_cover": 5, "weather_code": 0},
            hourly=hourly,
            solar_altitude_at=lambda _: 38.0,
        )

        self.assertTrue(nowcast["available"])
        self.assertEqual(nowcast["window_minutes"], 60)
        self.assertEqual([sample["minutes_ahead"] for sample in nowcast["samples"]], [0, 30, 60])
        self.assertGreater(nowcast["samples"][0]["probability"], nowcast["samples"][-1]["probability"])

    def test_hourly_lookup_chooses_closest_forecast_time(self) -> None:
        target = datetime(2026, 7, 19, 12, 44, tzinfo=HELSINKI)
        result = hourly_conditions_near(
            {"time": ["2026-07-19T12:00", "2026-07-19T13:00"], "cloud_cover": [20, 80]},
            target,
        )

        self.assertEqual(result, {"cloud_cover": 80})

    def test_season_features_wrap_across_new_year(self) -> None:
        new_year = seasonal_features(1)
        year_end = seasonal_features(366)
        summer = seasonal_features(172)
        winter = seasonal_features(355)

        self.assertLess(sum((left - right) ** 2 for left, right in zip(new_year, year_end)), 0.001)
        self.assertLess(sum(left * right for left, right in zip(summer, winter)), -0.9)

    def test_weather_fetch_exposes_the_trained_nowcast(self) -> None:
        now = datetime.now(HELSINKI).replace(minute=0, second=0, microsecond=0)
        payload = {
            "current": {"time": now.strftime("%Y-%m-%dT%H:%M"), "cloud_cover": 100, "weather_code": 3, "is_day": 1},
            "hourly": {
                "time": [now.strftime("%Y-%m-%dT%H:%M"), (now.replace(minute=0) + timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M")],
                "cloud_cover": [100, 100],
                "cloud_cover_low": [100, 100],
                "weather_code": [3, 3],
                "direct_radiation": [0, 0],
                "direct_normal_irradiance": [0, 0],
                "precipitation_probability": [0, 0],
            },
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with patch("backend.main.urlopen", return_value=Response()):
            weather = fetch_current_weather()

        self.assertTrue(weather["nowcast"]["available"])
        self.assertTrue(weather["nowcast"]["model"]["trained"])
        self.assertLessEqual(weather["nowcast"]["probability"], 8)


if __name__ == "__main__":
    unittest.main()
