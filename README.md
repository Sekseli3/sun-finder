# Sunfinder Helsinki
Have you ever had the problem where you want to catch some sweet sweet sunrays on a terrace or in a park, but builings seem to block everything? Don't worry I have the solution for you: Sunfinder! It shows where to catch those precious rays from an interactive map. Find it in the link below!
[**Hosted version here → sunfinder-helsinki.onrender.com**](https://sunfinder-helsinki.onrender.com/)

Hosted on Render's free tier. It may take about a minute to wake up after 15
minutes without visitors.

An interactive 2D / 3D map for finding sunlight in Helsinki.

The frontend is a small MapLibre browser app. Python handles the solar
calculation, sky estimate, and Bayesian nowcast. The browser streams visible
building tiles and projects their shadows locally.

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
- `GET /api/conditions` returns the sun position, current-sky estimate, and
  direct-sun nowcast. It stays small while the time slider moves.
- The browser loads prebuilt building vector tiles for the visible map and
  turns the cached footprints into shadow polygons immediately. It does not
  wait for a new server shadow response on every slider move.
- `GET /api/buildings` is a Python fallback. It reads Helsinki's official WFS
  building layer if the browser tile source is unavailable.
- `GET /api/place-suggestions` returns debounced Helsinki matches while a
  place name is being typed. It checks local favourites first, then uses
  cached Photon search limited to Helsinki.
- `GET /api/places` resolves a submitted Helsinki venue, landmark, or address.
  It uses cached OpenStreetMap Nominatim search with a one-request-per-second
  server-side gate.
- `GET /api/scene` remains available for a complete server-side scene response.
- The backend calculates sun direction in the `Europe/Helsinki` time zone,
  including daylight-saving transitions.
- In live mode, the backend retrieves Helsinki cloud cover and estimates the
  chance of direct sun over the next hour. The map starts with
  **clear-sky potential** so a pessimistic cloud estimate does not hide a
  possible sunny gap. Turn it off to fade or suppress shadows using the live
  sky estimate. These are weather-model estimates, not a local sky
  observation. A manually selected time always shows clear-sky potential.
- The normal building path uses compact OpenFreeMap/OpenMapTiles building tiles
  from Helsinki's map tile service. MapLibre keeps nearby tiles in its browser
  cache as you pan. The Python fallback uses Helsinki's official WFS building
  layer and keeps a viewport response in memory for 12 hours.
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

On that held out data it gets 96.6% threshold accuracy and a 0.0258 Brier
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

z = -2.8533
    - 2.6012 * total_cloud
    + 0.4081 * low_cloud
    - 1.2060 * total_cloud * low_cloud
    - 1.2874 * precipitation_signal
    + 0.0263 * rain_code
    + 25.3626 * direct_radiation_fraction
    + 3.4459 * sin(sun_altitude)
    - 0.3300 * sin(season)
    + 0.0689 * cos(season)
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

The weights work together. For example, low cloud has a small positive weight
on its own, but the total cloud and cloud interaction terms pull the score down.
That is a correlation in this dataset, not a claim that low cloud brings sun.

Here is a small example. Say it is late July, with 60% total cloud, 40% low
cloud, 20% rain chance, no rain code, a 35° high sun, and 108 W/m² of direct
radiation. That gives a radiation fraction of `0.2511`. The season values for
that day are about `-0.278` and `-0.961`.

```text
z = -2.8533 - 2.6012(0.60) + 0.4081(0.40) - 1.2060(0.60)(0.40)
    - 1.2874(0.20) + 25.3626(0.2511) + 3.4459(0.574)
    - 0.3300(-0.278) + 0.0689(-0.961)
  = 3.5729

chance = 1 / (1 + exp(-3.5729)) = 0.973
```

That becomes a 97% chance. The app then has a last sanity check. Fog, rain,
and very heavy cloud with almost no radiation can pull that number down before
it reaches the screen.

### The Bayesian bit

The model is now Bayesian logistic regression. It starts with these priors:

```text
intercept ~ Normal(0.3130, 2.5²)
each feature weight ~ Normal(0, 2.5²)
```

`0.3130` is the log odds of direct sun in the training rows. The other prior
means are zero, so the model does not begin by assuming a huge effect from any
one clue.

The trainer finds the most likely weights after seeing the data. That point is
called MAP. It then looks at the curvature around that point and turns it into
a Gaussian approximation of the posterior.

```text
weights | data ≈ Normal(MAP weights, inverse negative Hessian)
```

This is called a Laplace approximation. It is much lighter than running a long
MCMC chain, which makes it a good fit for this small Python app. Training took
10 Newton steps on the current three year data set.

For every live forecast row, the app gets a middle probability and a 90% model
range from that posterior. For the example above, the feature vector `x` gives:

```text
xᵀΣx = 0.0664
sd(z) = sqrt(0.0664) = 0.2577
z | data ≈ Normal(3.5729, 0.2577²)

90% range for chance
= sigmoid(3.5729 ± 1.645 * 0.2577)
= 0.9589 to 0.9820
```

So the middle result is 97.3% and the 90% model range is 95.9% to 98.2%.

The number on screen averages now, +30 minutes, and +60 minutes. Those three
rows share the same learned weights, so the app carries the shared covariance
through that average with a small delta-method calculation. It does not just
average three separate ranges.

That range only covers uncertainty in the learned weights. It does not know
whether a new small cloud will arrive above one street, so it is not a full
weather confidence interval.

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
