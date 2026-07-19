# Sunfinder Helsinki

[**Hosted version here → sunfinder-helsinki.onrender.com**](https://sunfinder-helsinki.onrender.com/)

Hosted on Render's free tier. It may take about a minute to wake up after 15
minutes without visitors.

An interactive 2D / 3D map for finding sunlight in Helsinki.

The frontend is a small MapLibre browser app. The Python/FastAPI backend owns
the solar calculation, building-footprint lookup and shadow projection.

## Run it

Create a virtual environment if you want to keep the dependencies isolated,
then install and run the Python service:

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 4173
```

Open [http://localhost:4173](http://localhost:4173).

`make install`, `make run`, and `make check` provide the same common actions.

## How it works

- The time control uses the `Europe/Helsinki` time zone, including daylight
  saving transitions.
- `GET /api/scene` accepts a Helsinki map bounding box and time. It returns the
  solar position, sunrise/sunset, building features, and Python-projected
  shadow polygons in one response.
- `GET /api/conditions` returns the quicker sun, current-sky, and direct-sun
  nowcast data without waiting for building footprints.
- The backend calculates sun direction in the `Europe/Helsinki` time zone,
  including daylight-saving transitions.
- In live mode, the backend also retrieves Helsinki cloud cover and fades or
  suppresses projected shadows when direct sun is unlikely. It also estimates
  the chance of direct sun over the next hour. These are weather-model
  estimates, not a local sky observation. For any manually selected time, the
  map deliberately shows **clear-sky potential**.
- Building footprints are fetched server-side from OpenStreetMap’s public
  Overpass API and cached in memory for 12 hours. A small fallback set is used
  if that public service is unavailable.
- Shadows are projected from building height and sun altitude. They are useful
  planning visualisations, not survey-grade measurements.

## Direct-sun estimate (beta)

The optional live-map estimate is the average probability for now, +30 minutes,
and +60 minutes. It predicts whether direct sunlight is likely to reach an
**unobstructed point** during the next hour.

### Scope

This is one Helsinki-wide weather signal, calculated from the fixed forecast
location `60.1699, 24.9384` in central Helsinki. It is **not** a separate
forecast for each street or neighbourhood, and it does not mean every part of
Helsinki will see sun at once. The map separately uses building geometry to
show whether a given location is blocked; trees and locally changing cloud are
not modelled. The UI labels this feature **BETA** for that reason.

### Live update path

In live mode, the browser asks `/api/conditions` when the map refreshes. The
selected cadence is one, five, or ten minutes (one minute by default). The
Python service caches the upstream weather response for five minutes, so it
does not call Open-Meteo more often than that.

Each upstream request asks Open-Meteo for the current Helsinki cloud condition
and three hours of hourly forecast fields: total and low cloud, weather code,
precipitation probability, direct radiation, and direct normal irradiance. At
the zero-minute sample the newest current cloud/weather-code values take
precedence; the +30 and +60 minute samples use the closest hourly forecast.
The three direct-sun probabilities are averaged and rounded into the number in
the UI. It is weather-model data, not a camera or local sunshine sensor.

### Training data and target

The checked-in model is a small logistic regression trained on three years of
[Helsinki weather reanalysis](https://open-meteo.com/en/docs/historical-weather-api),
from 2023-07-15 through 2026-07-13. A daylight row is positive when its direct
normal irradiance is at least 120 W/m², the WMO-style sunshine threshold.

The 13,257 daylight rows are ordered in time: 8,838 (the first two thirds) are
used for training and 4,419 (the final third) are held out for validation. This
keeps future observations out of the fitted model and gives the holdout a full
annual cycle. On that holdout, the model reached 95.2% threshold accuracy and
a 0.0395 Brier score; a constant climatology baseline reached 60.5% and
0.2397. These are **reanalysis-to-reanalysis** metrics: the target and the
strong direct-radiation feature come from the same data source, so they are not
a claim of equivalent accuracy for a particular Helsinki street or a true
forecast-as-issued benchmark.

### Model math and prior assumptions

For a feature vector `x`, the model calculates `p = sigmoid(z)`, where
`sigmoid(z) = 1 / (1 + exp(-z))` and the current fitted log-odds are:

```text
z = -0.9866
    - 1.4096 * total_cloud
    - 0.1914 * low_cloud
    - 0.6662 * total_cloud * low_cloud
    - 0.7982 * precipitation_signal
    - 0.3032 * rainy_weather_code
    + 8.5981 * direct_radiation_fraction
    + 2.0229 * sin(sun_altitude)
    - 0.2483 * sin(season)
    - 0.2357 * cos(season)
```

Cloud, low cloud, and precipitation are scaled to `0..1`.
`rainy_weather_code` is `1` for a WMO weather code of 51 or higher.
`direct_radiation_fraction` is the horizontal direct-radiation forecast divided
by `max(25, 750 * sin(sun_altitude))`, then clamped to `0..1`.
The season pair is `sin(2π(day - 1)/365.2425)` and
`cos(2π(day - 1)/365.2425)`, so late December and early January remain close
on the yearly cycle.

This is a deterministic regularised logistic regression, not a Bayesian model:
it has no posterior distribution or credible intervals. The intercept starts
at the training set's base-rate log-odds, feature weights start at zero, and
full-batch gradient descent runs for 1,200 iterations with learning rate `0.7`.
An L2 penalty of `0.001` applies to non-intercept weights; in a MAP
interpretation this acts like a weak, zero-centred Gaussian prior that shrinks
unneeded weights towards zero. After the logistic output, fog, precipitation,
and high-cloud/low-radiation rules cap the displayed probability conservatively.

Retrain the model with the latest historical data using only the Python
standard library:

```sh
python3 scripts/train_direct_sun_model.py --days 1095
```

The useful next DS upgrade is to train and evaluate it against local
[FMI solar-radiation observations](https://en.ilmatieteenlaitos.fi/weather-observations),
using rolling forecast-time splits. That would turn this weather-model
calibration into a proper local nowcast benchmark.

## How a shadow is calculated

For a building of height `H` and a sun altitude of `α`, the projected shadow
length is `H / tan(α)`. The app caps that length at 560 metres so very low sun
does not create enormous map geometry.

```python
shadow_length_m = min(560, building_height_m / tan(radians(sun_altitude_deg)))
shadow_bearing_deg = (sun_azimuth_deg + 180) % 360  # opposite the sun
```

Each point in the building footprint is shifted by that distance and bearing.
The original and shifted footprints are then combined into one convex-hull
polygon. For example, a 20 m building with a 30°-high sun casts a roughly
34.6 m shadow; if the sun is at 135°, the shadow points towards 315°.
