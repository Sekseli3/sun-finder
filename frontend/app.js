/* global maplibregl */

/*
 * The map deliberately stays light: it only owns interaction and rendering.
 * /api/scene is the Python service that calculates solar geometry, obtains
 * building footprints, and projects the shadow geometry.
 */

const HELSINKI = { lat: 60.1699, lng: 24.9384, timeZone: 'Europe/Helsinki' };
const DEFAULT_BUILDING_HEIGHT = 15;
const BUILDING_RETRY_DELAY_MS = 1_800;
const MAX_BUILDING_RETRIES = 2;

const state = {
  date: new Date(),
  live: true,
  refreshMinutes: 1,
  is3d: true,
  map: null,
  solar: null,
  sunTimes: null,
  weather: null,
  showClearSkyPotential: false,
  showDirectSunNowcast: true,
  buildings: emptyFeatureCollection(),
  shadows: emptyFeatureCollection(),
  buildingsBoundsKey: null,
  hasLiveBuildingData: false,
  buildingsDirty: true,
  shadowsDirty: true,
  liveTimer: null,
  refreshDebounce: null,
  abortController: null,
  sceneRequestId: 0,
  conditionsAbortController: null,
  conditionsRequestId: 0,
  mapMoveRequestId: 0,
  buildingRetryTimer: null,
  buildingRetryBounds: null,
  buildingRetryCount: 0
};

const elements = {};
let toastTimer;

window.addEventListener('DOMContentLoaded', initialise);

function initialise() {
  [
    'date-time', 'time-slider', 'timeline-date', 'timeline-time', 'live-toggle',
    'refresh-rate', 'now-button', 'mobile-controls-toggle', 'control-panel', 'calendar-reset', 'sun-orb', 'sun-orbit', 'sun-pointer',
    'solar-state', 'solar-altitude', 'solar-detail', 'sunrise', 'sunset',
    'weather-callout', 'weather-glyph', 'weather-title', 'weather-detail', 'clear-sky-toggle',
    'nowcast-control', 'nowcast-toggle', 'nowcast-info', 'nowcast-dialog', 'nowcast-range', 'close-nowcast', 'legend-shadow-label',
    'header-status', 'place-select', 'load-buildings', 'building-status', 'data-note',
    'map-loading', 'map-loading-title', 'map-loading-detail', 'inspector', 'inspector-title',
    'inspector-detail', 'close-inspector', 'locate-button', 'about-button', 'about-dialog',
    'close-about', 'toast'
  ].forEach((id) => { elements[id] = document.getElementById(id); });

  wireControls();
  syncInputsFromDate();
  initialiseMap();
  scheduleLiveRefresh();
}

function initialiseMap() {
  if (!window.maplibregl) {
    showToast('The map library could not load. Check your internet connection.');
    return;
  }

  const map = new maplibregl.Map({
    container: 'map',
    center: [HELSINKI.lng, HELSINKI.lat],
    zoom: 14.15,
    pitch: 54,
    bearing: -18,
    antialias: true,
    attributionControl: false,
    style: {
      version: 8,
      sources: {
        osm: {
          type: 'raster',
          tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
          tileSize: 256,
          maxzoom: 19,
          attribution: '© OpenStreetMap contributors'
        }
      },
      layers: [{
        id: 'osm',
        type: 'raster',
        source: 'osm',
        paint: {
          'raster-saturation': -0.58,
          'raster-brightness-min': 0.62,
          'raster-brightness-max': 0.92
        }
      }]
    }
  });

  state.map = map;
  map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');
  map.on('load', () => {
    installMapLayers();
    refreshScene({ quiet: true, showProgress: true });
    map.on('mouseenter', 'building-extrusions', () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', 'building-extrusions', () => { map.getCanvas().style.cursor = ''; });
  });
  map.on('click', inspectBuilding);
  map.on('error', (event) => {
    if (event?.error?.message?.includes('Failed to fetch')) return;
    console.warn('Map error', event.error);
  });
}

function installMapLayers() {
  const { map } = state;
  map.addSource('building-footprints', { type: 'geojson', data: emptyFeatureCollection() });
  map.addSource('building-shadows', { type: 'geojson', data: emptyFeatureCollection() });

  map.addLayer({
    id: 'building-shadows',
    type: 'fill',
    source: 'building-shadows',
    paint: { 'fill-color': '#152d3d', 'fill-opacity': 0.57, 'fill-antialias': true }
  });
  map.addLayer({
    id: 'building-footprints-outline',
    type: 'line',
    source: 'building-footprints',
    paint: { 'line-color': '#716c61', 'line-opacity': 0.42, 'line-width': 0.65 }
  });
  map.addLayer({
    id: 'building-extrusions',
    type: 'fill-extrusion',
    source: 'building-footprints',
    minzoom: 12,
    paint: {
      'fill-extrusion-color': '#c9934d',
      'fill-extrusion-height': ['get', 'height'],
      'fill-extrusion-base': 0,
      'fill-extrusion-opacity': 0.9,
      'fill-extrusion-vertical-gradient': true
    }
  });
}

function wireControls() {
  elements['date-time'].addEventListener('change', () => {
    const selected = zonedInputToDate(elements['date-time'].value, HELSINKI.timeZone);
    if (!selected) return;
    state.date = selected;
    setLive(false);
    refreshScene({ syncDateInput: true });
  });

  elements['time-slider'].addEventListener('input', () => {
    const minutes = Number(elements['time-slider'].value);
    const day = getZonedParts(state.date, HELSINKI.timeZone);
    state.date = zonedInputToDate(
      `${day.year}-${pad(day.month)}-${pad(day.day)}T${pad(Math.floor(minutes / 60))}:${pad(minutes % 60)}`,
      HELSINKI.timeZone
    );
    setLive(false);
    syncInputsFromDate();
    scheduleSceneRefresh();
  });

  elements['live-toggle'].addEventListener('change', () => {
    setLive(elements['live-toggle'].checked);
    if (state.live) setDateToNow();
  });
  elements['refresh-rate'].addEventListener('change', () => {
    state.refreshMinutes = Number(elements['refresh-rate'].value);
    scheduleLiveRefresh();
  });
  elements['now-button'].addEventListener('click', setDateToNow);
  elements['mobile-controls-toggle'].addEventListener('click', () => {
    setMobileControlsOpen(!elements['control-panel'].classList.contains('mobile-expanded'));
  });
  elements['calendar-reset'].addEventListener('click', setDateToNow);

  document.querySelectorAll('.dimension-option').forEach((button) => {
    button.addEventListener('click', () => setMapDimension(button.dataset.dimension === '3d'));
  });

  elements['clear-sky-toggle'].addEventListener('click', () => {
    state.showClearSkyPotential = !state.showClearSkyPotential;
    applyMapData();
    updateWeatherPanel();
  });
  elements['nowcast-toggle'].addEventListener('change', () => {
    state.showDirectSunNowcast = elements['nowcast-toggle'].checked;
    updateWeatherPanel();
    updateHeaderStatus();
  });

  elements['place-select'].addEventListener('change', () => {
    const [lat, lng, zoom, pitch] = elements['place-select'].value.split(',').map(Number);
    if (!state.map) return;
    flyToAndLoad({ center: [lng, lat], zoom, pitch: state.is3d ? pitch : 0, bearing: -18, duration: 1150, essential: true });
  });
  elements['load-buildings'].addEventListener('click', () => refreshScene({ retryBuildings: true }));
  elements['close-inspector'].addEventListener('click', () => { elements['inspector'].hidden = true; });
  elements['locate-button'].addEventListener('click', locateUser);
  elements['about-button'].addEventListener('click', () => elements['about-dialog'].showModal());
  elements['close-about'].addEventListener('click', () => elements['about-dialog'].close());
  elements['about-dialog'].addEventListener('click', (event) => {
    if (event.target === elements['about-dialog']) elements['about-dialog'].close();
  });
  elements['nowcast-info'].addEventListener('click', () => elements['nowcast-dialog'].showModal());
  elements['close-nowcast'].addEventListener('click', () => elements['nowcast-dialog'].close());
  elements['nowcast-dialog'].addEventListener('click', (event) => {
    if (event.target === elements['nowcast-dialog']) elements['nowcast-dialog'].close();
  });
}

function flyToAndLoad(target) {
  const { map } = state;
  if (!map) return;
  const requestId = ++state.mapMoveRequestId;
  window.clearTimeout(state.refreshDebounce);
  setBuildingButtonLoading(true);
  setBuildingLoadStatus('Loading visible area…', 'Fetching building footprints…');
  setMapLoading(true, 'Loading visible area…', 'Fetching building data…');

  const [targetLng, targetLat] = target.center;
  const currentCenter = map.getCenter();
  const alreadyAtTarget = Math.abs(currentCenter.lng - targetLng) < 0.00001
    && Math.abs(currentCenter.lat - targetLat) < 0.00001
    && Math.abs(map.getZoom() - target.zoom) < 0.01;
  const loadCurrentViewport = () => {
    if (requestId !== state.mapMoveRequestId) return;
    refreshScene({ quiet: true, showProgress: true });
  };

  if (alreadyAtTarget) {
    loadCurrentViewport();
    return;
  }
  map.once('moveend', loadCurrentViewport);
  map.flyTo(target);
}

function setDateToNow() {
  state.date = new Date();
  setLive(true);
  syncInputsFromDate();
  refreshScene();
}

function setMobileControlsOpen(isOpen) {
  const panel = elements['control-panel'];
  const button = elements['mobile-controls-toggle'];
  panel.classList.toggle('mobile-expanded', isOpen);
  button.setAttribute('aria-expanded', String(isOpen));
  button.textContent = isOpen ? 'Done' : 'Controls';
  if (state.map) window.setTimeout(() => state.map.resize(), 260);
}

function setLive(value) {
  state.live = value;
  elements['live-toggle'].checked = value;
  updateNowcastControl();
  if (!value) {
    state.showClearSkyPotential = false;
    state.weather = clearSkyPotentialWeather();
    updateWeatherPanel();
    applyMapData();
  }
  updateHeaderStatus();
}

function scheduleLiveRefresh() {
  window.clearInterval(state.liveTimer);
  state.liveTimer = window.setInterval(() => {
    if (state.live) {
      state.date = new Date();
      syncInputsFromDate();
      refreshScene({ quiet: true });
    }
  }, state.refreshMinutes * 60_000);
  setLive(state.live);
}

function scheduleSceneRefresh() {
  window.clearTimeout(state.refreshDebounce);
  state.refreshDebounce = window.setTimeout(() => refreshScene({ quiet: true }), 120);
}

function visibleBoundsKey() {
  const bounds = state.map?.getBounds();
  if (!bounds) return null;
  return [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()]
    .map((value) => value.toFixed(5))
    .join(',');
}

function clearBuildingRetry() {
  window.clearTimeout(state.buildingRetryTimer);
  state.buildingRetryTimer = null;
  state.buildingRetryBounds = null;
  state.buildingRetryCount = 0;
}

function scheduleBuildingRetry(boundsKey) {
  if (state.buildingRetryBounds !== boundsKey) {
    clearBuildingRetry();
    state.buildingRetryBounds = boundsKey;
  }
  if (state.buildingRetryCount >= MAX_BUILDING_RETRIES) return false;

  state.buildingRetryCount += 1;
  window.clearTimeout(state.buildingRetryTimer);
  state.buildingRetryTimer = window.setTimeout(() => {
    state.buildingRetryTimer = null;
    if (visibleBoundsKey() !== boundsKey) {
      clearBuildingRetry();
      return;
    }
    refreshScene({ quiet: true, showProgress: true, retryBuildings: true, fromAutoRetry: true });
  }, BUILDING_RETRY_DELAY_MS);
  return true;
}

async function refreshScene({
  quiet = false,
  syncDateInput = false,
  showProgress = !quiet,
  retryBuildings = false,
  fromAutoRetry = false
} = {}) {
  if (syncDateInput) syncInputsFromDate();
  if (!fromAutoRetry) clearBuildingRetry();
  refreshConditions();
  if (!state.map?.isStyleLoaded() || !state.map.getSource('building-footprints')) return;

  const requestId = ++state.sceneRequestId;
  state.abortController?.abort();
  state.abortController = new AbortController();
  const boundsKey = visibleBoundsKey();
  if (!boundsKey) return;
  const includeBuildings = retryBuildings
    || state.buildingsBoundsKey !== boundsKey
    || !state.hasLiveBuildingData;
  if (showProgress) {
    setBuildingButtonLoading(true);
    setBuildingLoadStatus(
      'Loading visible area…',
      'Fetching building footprints…'
    );
    setMapLoading(true, 'Loading visible area…', 'Fetching building data…');
  }
  const parameters = new URLSearchParams({
    bbox: boundsKey,
    at: state.date.toISOString(),
    live: String(state.live),
    retry_buildings: String(retryBuildings),
    include_buildings: String(includeBuildings)
  });

  try {
    const response = await fetch(`/api/scene?${parameters}`, { signal: state.abortController.signal });
    const scene = await response.json();
    if (!response.ok) throw new Error(scene.detail || `The API returned ${response.status}`);
    if (requestId !== state.sceneRequestId) return;
    applyScene(scene, { boundsKey, includeBuildings });
    if (scene.meta.source === 'fallback') {
      const retryScheduled = scheduleBuildingRetry(boundsKey);
      if (!quiet) {
        showToast(retryScheduled
          ? 'Building data is taking a moment. Trying again…'
          : 'Building data is unavailable right now. Try refresh again shortly.');
      }
    } else {
      clearBuildingRetry();
      if (!quiet) showToast(`Loaded ${scene.meta.building_count.toLocaleString()} buildings for the visible map. Shadows updated.`);
    }
  } catch (error) {
    if (error.name === 'AbortError') return;
    if (requestId !== state.sceneRequestId) return;
    console.warn('Scene request failed', error);
    setBuildingLoadStatus('Couldn’t load visible area', 'Refresh to try again.');
    if (!quiet) showToast(error.message || 'Could not calculate this scene.');
  } finally {
    if (requestId === state.sceneRequestId) {
      setBuildingButtonLoading(false);
      setMapLoading(false);
    }
  }
}

function applyScene(scene, { boundsKey, includeBuildings }) {
  applyConditions(scene, { render: false });
  const hasLiveBuildingData = scene.meta.source === 'openstreetmap';
  const buildingsIncluded = scene.meta.buildings_included ?? includeBuildings;
  if (buildingsIncluded || !hasLiveBuildingData) {
    state.buildings = hasLiveBuildingData ? scene.buildings || emptyFeatureCollection() : emptyFeatureCollection();
    state.buildingsBoundsKey = hasLiveBuildingData ? boundsKey : null;
    state.hasLiveBuildingData = hasLiveBuildingData;
    state.buildingsDirty = true;
  }
  state.shadows = hasLiveBuildingData ? scene.shadows || emptyFeatureCollection() : emptyFeatureCollection();
  state.shadowsDirty = true;
  applyMapData();

  const count = scene.meta.building_count;
  if (hasLiveBuildingData) {
    setBuildingLoadStatus(
      count ? `${count.toLocaleString()} buildings ready` : 'No buildings in this view',
      buildingsIncluded
        ? scene.meta.cached ? 'OpenStreetMap cache · visible area' : 'OpenStreetMap · visible area'
        : 'Cached building geometry · shadows updated'
    );
  } else {
    setBuildingLoadStatus(
      'Building data is taking a moment…',
      'Trying OpenStreetMap again…'
    );
  }
}

async function refreshConditions() {
  const requestId = ++state.conditionsRequestId;
  state.conditionsAbortController?.abort();
  state.conditionsAbortController = new AbortController();
  const parameters = new URLSearchParams({ at: state.date.toISOString(), live: String(state.live) });

  try {
    const response = await fetch(`/api/conditions?${parameters}`, { signal: state.conditionsAbortController.signal });
    const conditions = await response.json();
    if (!response.ok) throw new Error(conditions.detail || `The API returned ${response.status}`);
    if (requestId !== state.conditionsRequestId) return;
    applyConditions(conditions);
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== state.conditionsRequestId) return;
    console.warn('Conditions request failed', error);
    state.weather = state.live ? unavailableWeather() : clearSkyPotentialWeather();
    updateWeatherPanel();
    updateHeaderStatus();
  }
}

function applyConditions(conditions, { render = true } = {}) {
  state.solar = conditions.solar;
  state.sunTimes = conditions.sun_times;
  state.weather = conditions.weather || null;
  updateSunPanel();
  updateWeatherPanel();
  updateHeaderStatus();
  if (render) applyMapData();
}

function applyMapData() {
  if (!state.map?.isStyleLoaded() || !state.map.getSource('building-footprints')) return;
  if (state.buildingsDirty) {
    state.map.getSource('building-footprints').setData(state.buildings);
    state.buildingsDirty = false;
  }
  if (state.shadowsDirty) {
    state.map.getSource('building-shadows').setData(state.shadows);
    state.shadowsDirty = false;
  }
  state.map.setPaintProperty('building-shadows', 'fill-opacity', weatherAdjustedShadowOpacity());
  state.map.setPaintProperty('building-extrusions', 'fill-extrusion-opacity', 0.9);
  state.map.setPaintProperty('building-extrusions', 'fill-extrusion-color', buildingSunColor());
  state.map.setPaintProperty('osm', 'raster-saturation', -0.58);
  state.map.setPaintProperty('osm', 'raster-brightness-max', 0.92);
}

function buildingSunColor() {
  const sunIsUp = state.solar?.altitude > 0;
  return sunIsUp ? '#c9934d' : '#536474';
}

function updateSunPanel() {
  if (!state.solar) return;
  const { altitude, azimuth } = state.solar;
  const daylight = altitude > -0.83;
  const directSun = altitude > 0;
  const compass = bearingToCompass(azimuth);
  const glow = directSun ? Math.max(0.25, Math.min(1, altitude / 44)) : 0;

  elements['solar-state'].textContent = directSun ? `Sun from ${compass}` : daylight ? 'Civil twilight' : 'Sun below horizon';
  elements['solar-altitude'].textContent = `${formatDegrees(altitude)}°`;
  elements['solar-detail'].textContent = directSun ? `Azimuth ${Math.round(azimuth)}° ${compass}` : 'Sun altitude';
  elements['sun-pointer'].style.transform = `rotate(${azimuth}deg)`;
  elements['sun-orb'].style.opacity = daylight ? String(0.35 + glow * 0.65) : '0.15';
  elements['sun-orb'].style.transform = `scale(${0.75 + glow * 0.45})`;
  elements['sunrise'].textContent = state.sunTimes?.sunrise || '--:--';
  elements['sunset'].textContent = state.sunTimes?.sunset || '--:--';
}

function updateWeatherPanel() {
  const weather = state.weather;
  updateClearSkyToggle();
  updateNowcastControl();
  updateNowcastDialog();
  if (!weather) {
    elements['weather-callout'].dataset.state = 'loading';
    elements['weather-glyph'].textContent = '◌';
    elements['weather-title'].textContent = 'Checking sky…';
    elements['weather-detail'].textContent = 'Fetching current cloud cover.';
    finishWeatherPanel();
    return;
  }

  if (!weather.applies_to_selected_time) {
    elements['weather-callout'].dataset.state = 'potential';
    elements['weather-glyph'].textContent = '◌';
    elements['weather-title'].textContent = 'Clear-sky potential';
    elements['weather-detail'].textContent = weather.note;
    finishWeatherPanel();
    return;
  }

  if (!weather.available) {
    elements['weather-callout'].dataset.state = 'unavailable';
    elements['weather-glyph'].textContent = '?';
    elements['weather-title'].textContent = 'Sky estimate unavailable';
    elements['weather-detail'].textContent = weather.note;
    finishWeatherPanel();
    return;
  }

  const stateName = weather.shadow_visibility === 'unlikely'
    ? 'overcast'
    : weather.shadow_visibility === 'very soft' || weather.shadow_visibility === 'intermittent'
      ? 'cloudy'
      : 'clear';
  elements['weather-callout'].dataset.state = stateName;
  elements['weather-glyph'].textContent = weatherGlyph(weather);
  const nowcastProbability = visibleDirectSunNowcastProbability(weather);
  const shadowSummary = state.showClearSkyPotential
    ? 'clear-sky potential'
    : weather.shadow_visibility === 'unlikely'
      ? 'shadows hidden'
      : `${weather.shadow_visibility} shadows`;
  elements['weather-title'].textContent = nowcastProbability === null
    ? `${weather.label} · ${shadowSummary}`
    : `${nowcastProbability}% direct sun next hour`;
  const potentialSuffix = state.showClearSkyPotential
    ? ' Showing clear-sky geometry only — not an actual visible shadow.'
    : '';
  const nowcastDetail = nowcastProbability === null
    ? ''
    : ` Next-hour nowcast: ${nowcastProbability}% chance of direct sun at an open point. ${weather.nowcast.note}`;
  elements['weather-detail'].textContent = `${weather.note} ${weather.cloud_cover}% cloud cover.${nowcastDetail}${potentialSuffix}`;
  finishWeatherPanel();
}

function finishWeatherPanel() {
  updateLegendShadowLabel();
}

function updateClearSkyToggle() {
  const toggle = elements['clear-sky-toggle'];
  const weather = state.weather;
  const canShowPotential = weather?.applies_to_selected_time
    && weather.available
    && Number(weather.shadow_opacity) === 0;
  if (!canShowPotential) state.showClearSkyPotential = false;
  toggle.hidden = !canShowPotential;
  toggle.setAttribute('aria-pressed', String(state.showClearSkyPotential));
  toggle.textContent = state.showClearSkyPotential
    ? 'Hide clear-sky potential'
    : 'Show clear-sky potential';
}

function updateNowcastControl() {
  const toggle = elements['nowcast-toggle'];
  const control = elements['nowcast-control'];
  const availableInThisMode = state.live;
  toggle.checked = state.showDirectSunNowcast;
  toggle.disabled = !availableInThisMode;
  control.dataset.live = String(availableInThisMode);
  control.title = availableInThisMode ? '' : 'The direct-sun estimate is available in live mode only.';
}

function updateNowcastDialog() {
  const range = directSunNowcastRange(state.weather);
  const element = elements['nowcast-range'];
  if (!range) {
    element.hidden = true;
    element.textContent = '';
    return;
  }
  element.hidden = false;
  element.textContent = `For the next hour, the model range is ${range.lower}% to ${range.upper}%.`;
}

function updateHeaderStatus() {
  const status = elements['header-status'];
  const dot = document.querySelector('.live-dot');
  const weather = state.weather;
  if (!state.live) {
    status.textContent = 'Clear-sky potential · custom time';
    dot.style.background = '#d68a2c';
    return;
  }
  if (!weather) {
    status.textContent = 'Checking cloud cover';
    dot.style.background = '#d68a2c';
    return;
  }
  if (!weather.available) {
    status.textContent = 'Clear-sky potential · cloud data unavailable';
    dot.style.background = '#d68a2c';
    return;
  }
  const nowcastProbability = visibleDirectSunNowcastProbability(weather);
  if (nowcastProbability !== null) {
    status.textContent = `Nowcast beta · ${nowcastProbability}% direct sun next hour`;
    dot.style.background = nowcastProbability >= 70 ? '#46a276' : nowcastProbability >= 35 ? '#d68a2c' : '#788a94';
    return;
  }
  if (weather.shadow_visibility === 'unlikely') {
    status.textContent = `${weather.label} · direct shadows unlikely`;
    dot.style.background = '#788a94';
    return;
  }
  status.textContent = `${weather.label} · ${weather.shadow_visibility} shadows`;
  dot.style.background = weather.shadow_visibility === 'defined' ? '#46a276' : '#d68a2c';
}

function directSunNowcastProbability(weather) {
  const probability = Number(weather?.nowcast?.probability);
  return weather?.nowcast?.available && Number.isFinite(probability)
    ? Math.round(probability)
    : null;
}

function directSunNowcastRange(weather) {
  const uncertainty = weather?.nowcast?.uncertainty;
  const lower = Number(uncertainty?.lower);
  const upper = Number(uncertainty?.upper);
  if (!uncertainty?.available || !Number.isFinite(lower) || !Number.isFinite(upper)) return null;
  return { lower: Math.round(lower), upper: Math.round(upper) };
}

function visibleDirectSunNowcastProbability(weather) {
  return state.showDirectSunNowcast ? directSunNowcastProbability(weather) : null;
}

function weatherAdjustedShadowOpacity() {
  if (state.solar?.altitude <= 0) return 0;
  const weather = state.weather;
  if (state.showClearSkyPotential && weather?.applies_to_selected_time && weather.available && Number(weather.shadow_opacity) === 0) {
    return 0.58;
  }
  if (weather?.applies_to_selected_time && weather.available) return Number(weather.shadow_opacity);
  return 0.58;
}

function updateLegendShadowLabel() {
  const label = elements['legend-shadow-label'];
  if (state.showClearSkyPotential) {
    label.textContent = 'clear-sky potential';
    return;
  }
  const opacity = weatherAdjustedShadowOpacity();
  if (opacity < 0.08) {
    label.textContent = 'direct shadows unlikely';
  } else if (opacity < 0.3) {
    label.textContent = 'very soft shadow';
  } else if (!state.weather?.applies_to_selected_time || !state.weather?.available) {
    label.textContent = 'clear-sky potential';
  } else {
    label.textContent = 'weather-adjusted shadow';
  }
}

function weatherGlyph(weather) {
  if (weather.shadow_visibility === 'unlikely') return '☁';
  if (weather.shadow_visibility === 'very soft' || weather.shadow_visibility === 'intermittent') return '☁';
  return '☀';
}

function clearSkyPotentialWeather() {
  return {
    available: false,
    applies_to_selected_time: false,
    shadow_opacity: 0.58,
    note: 'Current cloud cover is not applied to a selected time. This view shows clear-sky potential.'
  };
}

function unavailableWeather() {
  return {
    available: false,
    applies_to_selected_time: true,
    shadow_opacity: 0.58,
    note: 'Sky estimate unavailable. Showing clear-sky potential.'
  };
}

function setBuildingLoadStatus(title, detail) {
  elements['building-status'].textContent = title;
  elements['data-note'].textContent = detail;
}

function setMapLoading(isLoading, title = '', detail = '') {
  elements['map-loading'].hidden = !isLoading;
  if (!isLoading) return;
  elements['map-loading-title'].textContent = title;
  elements['map-loading-detail'].textContent = detail;
}

function setBuildingButtonLoading(isLoading) {
  const button = elements['load-buildings'];
  button.disabled = isLoading;
  button.innerHTML = isLoading
    ? '<span class="loader"></span> Loading this map area…'
    : 'Load buildings here';
}

function inspectBuilding(event) {
  if (!state.map?.getLayer('building-extrusions') || !state.solar) return;
  const features = state.map.queryRenderedFeatures(event.point, { layers: ['building-extrusions'] });
  if (!features.length) return;
  const selected = features[0];
  elements['inspector'].hidden = false;
  const height = normaliseHeight(selected.properties.height);
  elements['inspector-title'].textContent = selected.properties.name || 'Selected building';
  if (state.solar.altitude <= 0) {
    elements['inspector-detail'].textContent = `${Math.round(height)} m high · the sun is below Helsinki’s horizon.`;
    return;
  }
  const length = Math.min(560, height / Math.tan(toRadians(state.solar.altitude)));
  const weather = state.weather;
  const realWorldNote = weather?.applies_to_selected_time && weather.available
    ? weather.shadow_visibility === 'unlikely'
      ? `Under ${weather.label.toLowerCase()} sky, it is unlikely to be visible.`
      : `Current sky suggests ${weather.shadow_visibility} shadows.`
    : 'This is a clear-sky projection.';
  elements['inspector-detail'].textContent = `${Math.round(height)} m high · clear-sky length ${Math.round(length)} m toward ${bearingToCompass((state.solar.azimuth + 180) % 360)}. ${realWorldNote}`;
}

function setMapDimension(is3d) {
  state.is3d = is3d;
  document.querySelectorAll('.dimension-option').forEach((button) => {
    const isActive = (button.dataset.dimension === '3d') === is3d;
    button.classList.toggle('active', isActive);
    button.setAttribute('aria-pressed', String(isActive));
  });
  if (!state.map) return;
  state.map.easeTo({ pitch: is3d ? 54 : 0, bearing: is3d ? -18 : 0, duration: 700, essential: true });
}

function locateUser() {
  if (!navigator.geolocation || !state.map) {
    showToast('Location is not available in this browser.');
    return;
  }
  navigator.geolocation.getCurrentPosition(
    ({ coords }) => {
      const withinHelsinki = coords.latitude >= 59.95 && coords.latitude <= 60.38 && coords.longitude >= 24.6 && coords.longitude <= 25.4;
      if (!withinHelsinki) {
        showToast('This prototype currently calculates shadows for the Helsinki region only.');
        return;
      }
      flyToAndLoad({ center: [coords.longitude, coords.latitude], zoom: 16, pitch: state.is3d ? 54 : 0, bearing: -18, duration: 850, essential: true });
    },
    () => showToast('Location permission was not granted.'),
    { enableHighAccuracy: true, timeout: 8_000, maximumAge: 60_000 }
  );
}

function syncInputsFromDate() {
  const parts = getZonedParts(state.date, HELSINKI.timeZone);
  elements['date-time'].value = `${parts.year}-${pad(parts.month)}-${pad(parts.day)}T${pad(parts.hour)}:${pad(parts.minute)}`;
  elements['time-slider'].value = parts.hour * 60 + parts.minute;
  elements['timeline-date'].textContent = readableDate(parts);
  elements['timeline-time'].textContent = `${pad(parts.hour)}:${pad(parts.minute)}`;
}

function getZonedParts(date, timeZone) {
  const formatter = new Intl.DateTimeFormat('en-CA', {
    timeZone,
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit', hourCycle: 'h23'
  });
  const values = Object.fromEntries(formatter.formatToParts(date)
    .filter((part) => part.type !== 'literal')
    .map((part) => [part.type, part.value]));
  return {
    year: Number(values.year), month: Number(values.month), day: Number(values.day),
    hour: Number(values.hour), minute: Number(values.minute), second: Number(values.second)
  };
}

function zonedInputToDate(value, timeZone) {
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})$/.exec(value);
  if (!match) return null;
  const [, yearText, monthText, dayText, hourText, minuteText] = match;
  const expectedAsUtc = Date.UTC(Number(yearText), Number(monthText) - 1, Number(dayText), Number(hourText), Number(minuteText));
  let candidate = new Date(expectedAsUtc);
  const offset = getTimeZoneOffset(candidate, timeZone);
  candidate = new Date(expectedAsUtc - offset * 60_000);
  const correctedOffset = getTimeZoneOffset(candidate, timeZone);
  return correctedOffset === offset ? candidate : new Date(expectedAsUtc - correctedOffset * 60_000);
}

function getTimeZoneOffset(date, timeZone) {
  const local = getZonedParts(date, timeZone);
  const localAsUtc = Date.UTC(local.year, local.month - 1, local.day, local.hour, local.minute, local.second);
  return Math.round((localAsUtc - date.getTime()) / 60_000);
}

function readableDate(parts) {
  const current = getZonedParts(new Date(), HELSINKI.timeZone);
  if (parts.year === current.year && parts.month === current.month && parts.day === current.day) return 'Today';
  return new Intl.DateTimeFormat('en-GB', { timeZone: HELSINKI.timeZone, day: 'numeric', month: 'short' }).format(state.date);
}

function normaliseHeight(value) {
  const parsed = Number.parseFloat(value);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_BUILDING_HEIGHT;
}

function bearingToCompass(bearing) {
  return ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round(bearing / 45) % 8];
}

function emptyFeatureCollection() { return { type: 'FeatureCollection', features: [] }; }
function toRadians(value) { return value * Math.PI / 180; }
function pad(value) { return String(value).padStart(2, '0'); }
function formatDegrees(value) { return value >= 0 ? Math.round(value) : `−${Math.abs(Math.round(value))}`; }

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('show'), 4_000);
}
