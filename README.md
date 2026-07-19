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
  for planning, not survey work.

## Direct sun estimate

The BETA switch is there for a simple question. Is direct sun likely to reach
an open spot in Helsinki in the next hour? The app makes one estimate for now,
+30 minutes, and +60 minutes, then averages them.

### What it covers

The weather comes from one Open Meteo forecast location at `60.1699, 24.9384`.
Think of it as a city level clue, not a forecast for every street. A sunny
result does not mean every part of Helsinki is sunny. A local cloud, a tree, or
a taller building can still keep a particular spot in shade.

The map handles buildings separately. It uses the sun angle and building shape
to draw the shadow. The direct sun number only says something about the sky.

### How it updates

The browser asks `/api/conditions` every one, five, or ten minutes. One minute
is the default. The Python service keeps the weather response for five minutes
so Open Meteo does not get called on every screen refresh.

Each request gets the current cloud cover and weather code, plus three hours of
cloud, rain chance, direct radiation, and direct normal irradiance. The now
sample uses the newest cloud and weather code. The +30 and +60 samples use the
closest hourly forecast. The app turns all three into probabilities and shows
their average.

### What it learned from

The model learned from three years of
[Helsinki weather reanalysis](https://open-meteo.com/en/docs/historical-weather-api),
from 2023-07-15 to 2026-07-13. It only keeps daylight rows. A row counts as
direct sun when direct normal irradiance reaches 120 W/m².

There are 13,257 daylight rows. The first 8,838 rows teach the model. The last
4,419 rows are kept aside for checking it later. That last section includes a
full year, so the check sees winter, spring, summer, and autumn too.

On that held out data it gets 95.2% threshold accuracy and a 0.0395 Brier
score. A simple always use the average baseline gets 60.5% and 0.2397. Those
numbers are useful, but they are not a promise about a particular Helsinki
street. Both the answer and one very strong input come from the same reanalysis
data. A proper next step is to use observed FMI sunlight data and old forecast
runs.

### The actual calculation

The model makes a score called `z`, then turns it into a chance with the
sigmoid curve.

```text
chance = 1 / (1 + exp(-z))

z = -0.9866
    - 1.4096 * total_cloud
    - 0.1914 * low_cloud
    - 0.6662 * total_cloud * low_cloud
    - 0.7982 * precipitation_signal
    - 0.3032 * rain_code
    + 8.5981 * direct_radiation_fraction
    + 2.0229 * sin(sun_altitude)
    - 0.2483 * sin(season)
    - 0.2357 * cos(season)
```

Cloud, low cloud, and rain chance are numbers from 0 to 1. For example, 60%
cloud becomes `0.60`. `rain_code` is 1 when the weather code says drizzle,
rain, snow, or a storm. This is the `rainy_weather_code` flag in the code.
The radiation fraction compares the direct radiation forecast with a bright sky
value for the current sun height.

```text
direct_radiation_fraction = clamp(direct_radiation / max(25, 750 * sin(sun_altitude)))
season_sin = sin(2π * (day_of_year - 1) / 365.2425)
season_cos = cos(2π * (day_of_year - 1) / 365.2425)
```

The season values wrap the year into a circle, so the end of December sits next
to the start of January.

Here is a small example. Say it is late July, with 60% total cloud, 40% low
cloud, 20% rain chance, no rain code, a 35° high sun, and a direct radiation
fraction of 0.25. The season values for that day are about `-0.278` and
`-0.961`.

```text
z = -0.9866 - 1.4096(0.60) - 0.1914(0.40) - 0.6662(0.60)(0.40)
    - 0.7982(0.20) + 8.5981(0.25) + 2.0229(0.574)
    - 0.2483(-0.278) - 0.2357(-0.961)
  = 1.3769

chance = 1 / (1 + exp(-1.3769)) = 0.798
```

That becomes an 80% chance. The app then has a last sanity check. Fog, rain,
and very heavy cloud with almost no radiation can pull that number down before
it reaches the screen.

### The Bayesian bit

The model running today is not Bayesian. It learns one fixed weight for each
clue and returns one number. The trainer starts the feature weights at zero and
the intercept at the average direct sun rate in the training data. It reads the
full training set on each of 1,200 steps, with a learning rate of `0.7`.

The L2 setting of `0.001` is close to a simple Bayesian idea. It gently prefers
weights close to zero until the data gives a reason to move them. You can
picture a Bayesian version starting with a prior like this:

```text
cloud_weight ~ Normal(0, sigma²)
```

Instead of keeping one cloud weight, it would keep a range of believable cloud
weights. New observations would narrow or move that range. The final answer
could then say both that direct sun looks likely and how unsure the model is.
For example, it could show a middle chance with a wide range on a day where the
forecast inputs disagree.

That is not what the app does yet. The L2 penalty is a handy first step, but it
does not create Bayesian uncertainty or a confidence interval.

To train again with the latest three years of data, run:

```sh
python3 scripts/train_direct_sun_model.py --days 1095
```

The next useful upgrade is to train against local
[FMI solar radiation observations](https://en.ilmatieteenlaitos.fi/weather-observations)
and test with old forecast runs. That would make the result much closer to a
real nowcast check.

## How a shadow is calculated

For a building of height `H` and a sun altitude of `α`, the projected shadow
length is `H / tan(α)`. The app caps that length at 560 metres so very low sun
does not create enormous map geometry.

```python
shadow_length_m = min(560, building_height_m / tan(radians(sun_altitude_deg)))
shadow_bearing_deg = (sun_azimuth_deg + 180) % 360  # opposite the sun
```

Each point in the building footprint is shifted by that distance and bearing.
The original and shifted footprints are then combined into one convex hull
polygon. For example, a 20 m building with a 30° high sun casts a roughly
34.6 m shadow. If the sun is at 135°, the shadow points towards 315°.
