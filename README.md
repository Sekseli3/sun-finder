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

## Direct-sun nowcast

The live map shows the average probability from now, +30 minutes, and +60
minutes. It means direct sun at an **open point** in Helsinki; the building
geometry separately decides whether a specific map location is blocked.

The checked-in model is a small logistic regression trained on one year of
[Helsinki weather reanalysis](https://open-meteo.com/en/docs/historical-weather-api).
Its target is the WMO-style sunshine threshold: direct normal irradiance above
120 W/m². It uses total/low cloud cover,
precipitation signals, direct radiation, and sun altitude. Overcast, fog, and
precipitation are capped conservatively so the interface does not imply sharp
shadows under a uniformly grey sky.

Retrain the model with the latest historical data using only the Python
standard library:

```sh
python3 scripts/train_direct_sun_model.py --days 365
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
