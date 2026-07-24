/* global maplibregl */

/*
 * The map owns interaction, visible building tiles, and quick local shadows.
 * Python supplies solar geometry, sky conditions, and the Bayesian nowcast.
 */

const HELSINKI = { lat: 60.1699, lng: 24.9384, timeZone: 'Europe/Helsinki' };
const DEFAULT_MAP_VIEW = {
  lat: 60.1657789,
  lng: 24.9313873,
  zoom: 16.35,
  pitch: 58,
  bearing: -18
};
const DEFAULT_PLACE = {
  name: 'Bar Mendocino',
  detail: 'Eerikinkatu',
  latitude: DEFAULT_MAP_VIEW.lat,
  longitude: DEFAULT_MAP_VIEW.lng,
  kind: 'bar'
};
const DEFAULT_BUILDING_HEIGHT = 15;
const MAX_SHADOW_METERS = 560;
const METERS_PER_DEGREE_LAT = 111_320;
const BUILDING_TILE_SOURCE = 'helsinki-building-tiles';
const BUILDING_TILE_LAYER = 'building';
const BUILDING_TILE_URL = 'https://maptiles.api.hel.fi/data/helsinki.json';
const CONDITIONS_DEBOUNCE_MS = 260;
const PLACE_SUGGESTION_DEBOUNCE_MS = 280;

const state = {
  date: new Date(),
  live: true,
  refreshMinutes: 1,
  is3d: true,
  map: null,
  solar: null,
  sunTimes: null,
  weather: null,
  // Keep possible sunshine visible unless someone explicitly opts into the
  // live cloud adjustment. Missing a sunny gap is more costly than showing
  // a possible shadow that clouds may hide.
  showClearSkyPotential: true,
  showDirectSunNowcast: true,
  buildings: emptyFeatureCollection(),
  shadows: emptyFeatureCollection(),
  buildingsBoundsKey: null,
  buildingTileFingerprint: null,
  buildingTilesFailed: false,
  buildingsDirty: true,
  shadowsDirty: true,
  liveTimer: null,
  refreshDebounce: null,
  shadowPreviewFrame: null,
  buildingTileSyncTimer: null,
  buildingFallbackAbortController: null,
  buildingFallbackRequestId: 0,
  placeMarker: null,
  placeSearchAbortController: null,
  placeSearchRequestId: 0,
  placeSuggestionAbortController: null,
  placeSuggestionRequestId: 0,
  placeSuggestionTimer: null,
  sunPlannerStatus: null,
  sunPlannerAbortController: null,
  sunPlannerRequestId: 0,
  conditionsAbortController: null,
  conditionsRequestId: 0,
  mapMoveRequestId: 0
};

const elements = {};
let toastTimer;

window.addEventListener('DOMContentLoaded', initialise);

function initialise() {
  [
    'date-time', 'time-slider', 'timeline-date', 'timeline-time', 'live-toggle',
    'refresh-rate', 'now-button', 'mobile-controls-toggle', 'control-panel', 'close-control-panel', 'open-control-panel', 'calendar-reset', 'sun-orb', 'sun-orbit', 'sun-pointer',
    'solar-state', 'solar-altitude', 'solar-detail', 'sunrise', 'sunset',
    'weather-callout', 'weather-glyph', 'weather-title', 'weather-detail', 'clear-sky-toggle',
    'nowcast-control', 'nowcast-toggle', 'nowcast-info', 'nowcast-dialog', 'nowcast-range', 'close-nowcast', 'legend-shadow-label',
    'header-status', 'place-panel', 'close-place-panel', 'open-place-panel', 'place-search-form', 'place-search', 'place-search-submit', 'place-search-results', 'place-select', 'load-buildings', 'building-status', 'data-note',
    'sun-planner', 'close-sun-planner', 'open-sun-planner', 'sun-planner-form', 'sun-planner-prompt', 'sun-planner-submit', 'sun-planner-status', 'sun-planner-response', 'sun-planner-window', 'sun-planner-answer', 'sun-planner-results', 'sun-planner-note',
    'map-loading', 'map-loading-title', 'map-loading-detail', 'inspector', 'inspector-title',
    'inspector-detail', 'close-inspector', 'locate-button', 'about-button', 'about-dialog',
    'close-about', 'toast'
  ].forEach((id) => { elements[id] = document.getElementById(id); });

  wireControls();
  syncInputsFromDate();
  initialiseMap();
  scheduleLiveRefresh();
  loadSunPlannerStatus();
}

function initialiseMap() {
  if (!window.maplibregl) {
    showToast('The map library could not load. Check your internet connection.');
    return;
  }

  const map = new maplibregl.Map({
    container: 'map',
    center: [DEFAULT_MAP_VIEW.lng, DEFAULT_MAP_VIEW.lat],
    zoom: DEFAULT_MAP_VIEW.zoom,
    pitch: DEFAULT_MAP_VIEW.pitch,
    bearing: DEFAULT_MAP_VIEW.bearing,
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
    showPlaceMarker(DEFAULT_PLACE);
    refreshScene({ quiet: true, showProgress: true });
    map.on('mouseenter', 'building-extrusions', () => { map.getCanvas().style.cursor = 'pointer'; });
    map.on('mouseleave', 'building-extrusions', () => { map.getCanvas().style.cursor = ''; });
  });
  map.on('click', inspectBuilding);
  map.on('sourcedata', (event) => {
    if (event.sourceId !== BUILDING_TILE_SOURCE || !event.isSourceLoaded) return;
    state.buildingTilesFailed = false;
    scheduleBuildingTileSync();
  });
  map.on('moveend', () => scheduleBuildingTileSync());
  map.on('idle', () => scheduleBuildingTileSync());
  map.on('error', (event) => {
    if (event?.sourceId === BUILDING_TILE_SOURCE) {
      handleBuildingTileError(event.error);
      return;
    }
    if (event?.error?.message?.includes('Failed to fetch')) return;
    console.warn('Map error', event.error);
  });
}

function installMapLayers() {
  const { map } = state;
  map.addSource(BUILDING_TILE_SOURCE, { type: 'vector', url: BUILDING_TILE_URL });
  map.addSource('building-footprints', { type: 'geojson', data: emptyFeatureCollection() });
  map.addSource('building-shadows', { type: 'geojson', data: emptyFeatureCollection() });

  // This transparent layer tells MapLibre to stream just the visible building
  // tiles. Their geometry is then reused for local shadow projection.
  map.addLayer({
    id: 'building-tile-loader',
    type: 'fill',
    source: BUILDING_TILE_SOURCE,
    'source-layer': BUILDING_TILE_LAYER,
    minzoom: 13,
    paint: { 'fill-opacity': 0 }
  });

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
  elements['close-control-panel'].addEventListener('click', () => setControlPanelVisible(false));
  elements['open-control-panel'].addEventListener('click', () => setControlPanelVisible(true));
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

  elements['place-search-form'].addEventListener('submit', searchPlaces);
  elements['close-place-panel'].addEventListener('click', () => setPlacePanelVisible(false));
  elements['open-place-panel'].addEventListener('click', () => setPlacePanelVisible(true));
  elements['place-search'].addEventListener('input', schedulePlaceSuggestions);
  elements['place-search'].addEventListener('keydown', (event) => {
    if (event.key === 'Escape') {
      cancelPlaceSuggestions();
      hidePlaceSearchResults();
    }
  });
  elements['sun-planner-form'].addEventListener('submit', requestSunPlan);
  elements['close-sun-planner'].addEventListener('click', () => setSunPlannerVisible(false));
  elements['open-sun-planner'].addEventListener('click', () => setSunPlannerVisible(true));
  elements['place-select'].addEventListener('change', () => {
    const [lat, lng, zoom, pitch] = elements['place-select'].value.split(',').map(Number);
    if (!state.map) return;
    const [name, detail] = elements['place-select'].selectedOptions[0].textContent.split(' · ');
    moveToPlace({ name, detail: detail || 'Helsinki', latitude: lat, longitude: lng, kind: 'place' }, {
      zoom,
      pitch: state.is3d ? pitch : 0,
      duration: 1150,
      announce: true
    });
  });
  elements['load-buildings'].addEventListener('click', () => refreshVisibleBuildingTiles({ showProgress: true, reload: true }));
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

async function searchPlaces(event) {
  event.preventDefault();
  const query = elements['place-search'].value.trim();
  if (query.length < 2) {
    showToast('Type at least two letters to search for a place.');
    elements['place-search'].focus();
    return;
  }

  const requestId = ++state.placeSearchRequestId;
  cancelPlaceSuggestions();
  state.placeSearchAbortController?.abort();
  state.placeSearchAbortController = new AbortController();
  setPlaceSearchLoading(true);
  hidePlaceSearchResults();

  try {
    const parameters = new URLSearchParams({ q: query });
    const response = await fetch(`/api/places?${parameters}`, { signal: state.placeSearchAbortController.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `The API returned ${response.status}`);
    if (requestId !== state.placeSearchRequestId || query !== elements['place-search'].value.trim()) return;

    const results = (Array.isArray(payload.results) ? payload.results : [])
      .map(normalisePlaceSearchResult)
      .filter(Boolean);
    if (!results.length) {
      showToast(payload.meta?.available === false
        ? 'Place search is unavailable. Try again shortly.'
        : 'No Helsinki place found. Try the street address.');
      return;
    }

    renderPlaceSearchResults(results, 0);
    moveToPlace(results[0]);
    showToast(results.length > 1
      ? `Showing ${results[0].name}. Choose another match below if needed.`
      : `Showing ${results[0].name}.`);
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== state.placeSearchRequestId) return;
    console.warn('Place search failed', error);
    showToast('Couldn’t search for that place. Try again shortly.');
  } finally {
    if (requestId === state.placeSearchRequestId) setPlaceSearchLoading(false);
  }
}

async function loadSunPlannerStatus() {
  try {
    const response = await fetch('/api/sun-planner/status');
    const payload = await response.json();
    if (!response.ok) throw new Error(`The API returned ${response.status}`);
    state.sunPlannerStatus = payload;
    elements['open-sun-planner'].hidden = !payload.enabled;
    if (!payload.enabled) {
      setSunPlannerVisible(false);
      return;
    }
    setSunPlannerStatus(payload.ready
      ? 'Local Ollama planner is ready.'
      : String(payload.reason || 'Local planner is not ready yet.'), payload.ready ? 'ready' : 'waiting');
  } catch (error) {
    console.info('Local sun planner is not available on this server.', error);
    state.sunPlannerStatus = { enabled: false, ready: false };
    elements['open-sun-planner'].hidden = true;
    setSunPlannerVisible(false);
  }
}

async function requestSunPlan(event) {
  event.preventDefault();
  const message = elements['sun-planner-prompt'].value.trim();
  if (message.length < 2) {
    setSunPlannerStatus('Tell me where and when you would like to sit outside.', 'waiting');
    elements['sun-planner-prompt'].focus();
    return;
  }
  if (!state.sunPlannerStatus?.ready) {
    setSunPlannerStatus(String(state.sunPlannerStatus?.reason || 'Start Ollama and pull the planner models first.'), 'waiting');
    return;
  }
  const center = state.map?.getCenter();
  if (!center) {
    setSunPlannerStatus('The map is still loading. Try again in a moment.', 'waiting');
    return;
  }

  const requestId = ++state.sunPlannerRequestId;
  state.sunPlannerAbortController?.abort();
  state.sunPlannerAbortController = new AbortController();
  setSunPlannerLoading(true);
  setSunPlannerStatus('Checking nearby terraces, building shade, and local notes…', 'loading');

  try {
    const response = await fetch('/api/sun-plans', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: state.sunPlannerAbortController.signal,
      body: JSON.stringify({
        message,
        map_latitude: center.lat,
        map_longitude: center.lng,
        selected_time: state.date.toISOString()
      })
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `The API returned ${response.status}`);
    if (requestId !== state.sunPlannerRequestId) return;
    renderSunPlan(payload);
    setSunPlannerStatus(
      (payload?.meta?.plan_mode || 'sun') === 'sun'
        ? 'Sun plan ready. Tap a place to move the map there.'
        : 'Nearby choices ready. Tap a place to move the map there.',
      'ready'
    );
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== state.sunPlannerRequestId) return;
    console.warn('Sun planner failed', error);
    setSunPlannerStatus(error.message || 'Could not make a sun plan. Try again shortly.', 'error');
  } finally {
    if (requestId === state.sunPlannerRequestId) {
      setSunPlannerLoading(false);
      if (state.sunPlannerAbortController?.signal.aborted === false) state.sunPlannerAbortController = null;
    }
  }
}

function renderSunPlan(payload) {
  const results = Array.isArray(payload.recommendations) ? payload.recommendations : [];
  const planMode = String(payload?.meta?.plan_mode || 'sun');
  const windowLabel = String(payload?.request?.window_label || '');
  const anchorName = String(payload?.request?.anchor?.name || 'Current map view');
  const distanceOrigin = anchorName === 'Current map view' ? 'map centre' : anchorName;
  const anchorLabel = `From ${distanceOrigin}`;
  elements['sun-planner-window'].textContent = windowLabel
    ? `${anchorLabel} · ${windowLabel}`
    : anchorLabel;
  elements['sun-planner-window'].hidden = false;
  elements['sun-planner-answer'].textContent = String(payload.answer || 'Here are the closest sunny options.');
  const container = elements['sun-planner-results'];
  container.replaceChildren();

  results.forEach((result, index) => {
    const venue = result?.venue;
    if (!venue || !Number.isFinite(Number(venue.latitude)) || !Number.isFinite(Number(venue.longitude))) return;
    const card = document.createElement('article');
    card.className = planMode === 'sun' ? 'sun-plan-result' : 'sun-plan-result sun-plan-result--fallback';

    const choose = document.createElement('button');
    choose.type = 'button';
    choose.className = 'sun-plan-result-main';
    choose.setAttribute('aria-label', `Show ${venue.name} on the map`);
    if (planMode === 'sun') {
      const rank = document.createElement('span');
      rank.className = 'sun-plan-rank';
      rank.textContent = String(index + 1);
      choose.append(rank);
    } else {
      choose.classList.add('sun-plan-result-main--fallback');
    }
    const copy = document.createElement('span');
    const name = document.createElement('strong');
    name.textContent = String(venue.name || 'Helsinki venue');
    const detail = document.createElement('small');
    const exposure = planMode === 'building_unavailable'
      ? 'nearby fallback'
      : String(result.exposure || 'sun check unavailable');
    detail.textContent = `${exposure} · ${formatVenueDistance(result.distance_meters, distanceOrigin)}`;
    copy.append(name, detail);
    choose.append(copy);
    choose.addEventListener('click', () => {
      moveToPlace({
        name: String(venue.name || 'Helsinki venue'),
        detail: String(venue.area || 'Helsinki'),
        latitude: Number(venue.latitude),
        longitude: Number(venue.longitude),
        kind: String(venue.kind || 'place')
      }, { announce: true });
    });
    card.append(choose);

    const source = venue.source;
    if (source?.url) {
      const link = document.createElement('a');
      link.href = String(source.url);
      link.target = '_blank';
      link.rel = 'noreferrer';
      link.textContent = 'Place source';
      card.append(link);
    }
    container.append(card);
  });

  const plannedTime = new Date(payload?.request?.at);
  if (Number.isFinite(plannedTime.getTime())) {
    state.date = plannedTime;
    setLive(false);
    syncInputsFromDate();
    refreshScene({ quiet: true });
  }
  const weather = payload?.weather || {};
  const geometryAvailable = payload?.meta?.building_geometry_available === true;
  elements['sun-planner-note'].textContent = planMode === 'no_projected_sun'
    ? 'No close point is projected to receive direct sun in this one-hour window. These are nearby fallbacks.'
    : planMode === 'after_sunset'
      ? 'The sun is below the horizon at the selected time. These are nearby fallbacks.'
    : !geometryAvailable
    ? 'The city building check did not complete for this area, so this list is by nearby distance only. Press Plan it again to retry the building check.'
    : weather.applies_to_selected_time
      ? 'The weather number is a city-wide estimate for an open point. Trees and small local clouds are not modelled.'
      : 'For this future time, the score is clear-sky potential. It is not a local weather forecast.';
  elements['sun-planner-response'].hidden = false;
}

function formatVenueDistance(distance, origin = 'map centre') {
  const meters = Number(distance);
  if (!Number.isFinite(meters)) return 'nearby';
  const label = meters < 1_000 ? `${Math.round(meters)} m` : `${(meters / 1_000).toFixed(1)} km`;
  return `${label} from ${origin}`;
}

function setSunPlannerLoading(isLoading) {
  const button = elements['sun-planner-submit'];
  button.disabled = isLoading;
  button.textContent = isLoading ? 'Planning…' : 'Plan it';
  button.setAttribute('aria-busy', String(isLoading));
}

function setSunPlannerStatus(message, stateName = 'idle') {
  const status = elements['sun-planner-status'];
  status.textContent = message;
  status.dataset.state = stateName;
}

function schedulePlaceSuggestions() {
  state.placeSearchAbortController?.abort();
  cancelPlaceSuggestions();
  hidePlaceSearchResults();

  const query = elements['place-search'].value.trim();
  if (query.length < 2) return;

  const requestId = ++state.placeSuggestionRequestId;
  state.placeSuggestionTimer = window.setTimeout(() => {
    loadPlaceSuggestions(query, requestId);
  }, PLACE_SUGGESTION_DEBOUNCE_MS);
}

async function loadPlaceSuggestions(query, requestId) {
  if (requestId !== state.placeSuggestionRequestId) return;

  state.placeSuggestionTimer = null;
  const abortController = new AbortController();
  state.placeSuggestionAbortController?.abort();
  state.placeSuggestionAbortController = abortController;
  elements['place-search-results'].setAttribute('aria-busy', 'true');

  try {
    const parameters = new URLSearchParams({ q: query });
    const response = await fetch(`/api/place-suggestions?${parameters}`, { signal: abortController.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `The API returned ${response.status}`);
    if (requestId !== state.placeSuggestionRequestId || query !== elements['place-search'].value.trim()) return;

    const results = (Array.isArray(payload.results) ? payload.results : [])
      .map(normalisePlaceSearchResult)
      .filter(Boolean);
    renderPlaceSearchResults(results);
  } catch (error) {
    if (error.name !== 'AbortError' && requestId === state.placeSuggestionRequestId) {
      console.warn('Place suggestions failed', error);
    }
  } finally {
    if (requestId === state.placeSuggestionRequestId) {
      elements['place-search-results'].setAttribute('aria-busy', 'false');
      if (state.placeSuggestionAbortController === abortController) state.placeSuggestionAbortController = null;
    }
  }
}

function cancelPlaceSuggestions() {
  window.clearTimeout(state.placeSuggestionTimer);
  state.placeSuggestionTimer = null;
  state.placeSuggestionAbortController?.abort();
  state.placeSuggestionAbortController = null;
  state.placeSuggestionRequestId += 1;
  elements['place-search-results']?.setAttribute('aria-busy', 'false');
}

function normalisePlaceSearchResult(item) {
  const latitude = Number(item?.latitude);
  const longitude = Number(item?.longitude);
  const name = String(item?.name || '').trim();
  if (!name || !Number.isFinite(latitude) || !Number.isFinite(longitude)) return null;
  return {
    name,
    detail: String(item?.detail || 'Helsinki').trim(),
    latitude,
    longitude,
    kind: String(item?.kind || 'place')
  };
}

function renderPlaceSearchResults(results, activeIndex = -1) {
  const container = elements['place-search-results'];
  container.replaceChildren();
  results.forEach((place, index) => {
    const button = document.createElement('button');
    button.className = 'place-search-result';
    button.type = 'button';
    button.setAttribute('role', 'option');
    button.setAttribute('aria-selected', String(index === activeIndex));

    const glyph = document.createElement('span');
    glyph.className = 'place-search-result-glyph';
    glyph.setAttribute('aria-hidden', 'true');
    glyph.textContent = ['amenity', 'bar', 'cafe', 'pub', 'restaurant'].includes(place.kind) ? '●' : '⌖';
    const copy = document.createElement('span');
    const name = document.createElement('strong');
    name.textContent = place.name;
    const detail = document.createElement('small');
    detail.textContent = place.detail;
    copy.append(name, detail);
    button.append(glyph, copy);
    button.addEventListener('click', () => {
      cancelPlaceSuggestions();
      elements['place-search'].value = place.name;
      moveToPlace(place, { announce: true });
      hidePlaceSearchResults();
    });
    container.append(button);
  });
  container.hidden = !results.length;
  elements['place-search'].setAttribute('aria-expanded', String(results.length > 0));
}

function hidePlaceSearchResults() {
  const container = elements['place-search-results'];
  if (!container) return;
  container.replaceChildren();
  container.hidden = true;
  container.setAttribute('aria-busy', 'false');
  elements['place-search']?.setAttribute('aria-expanded', 'false');
}

function setPlaceSearchLoading(isLoading) {
  const button = elements['place-search-submit'];
  button.disabled = isLoading;
  button.setAttribute('aria-busy', String(isLoading));
  button.textContent = isLoading ? 'Finding…' : 'Find';
}

function moveToPlace(place, { zoom = 16.25, pitch = state.is3d ? 54 : 0, duration = 900, announce = false } = {}) {
  if (!state.map) return;
  showPlaceMarker(place);
  flyToAndLoad({
    center: [place.longitude, place.latitude],
    zoom,
    pitch,
    bearing: -18,
    duration,
    essential: true
  });
  if (announce) showToast(`Showing ${place.name}.`);
}

function showPlaceMarker(place) {
  if (!state.map || !window.maplibregl) return;
  state.placeMarker?.remove();

  const marker = document.createElement('div');
  marker.className = 'place-highlight-marker';
  marker.setAttribute('role', 'img');
  marker.setAttribute('aria-label', `Selected place: ${place.name}`);
  const label = document.createElement('div');
  label.className = 'place-highlight-label';
  label.title = place.name;
  const name = document.createElement('strong');
  name.textContent = place.name;
  label.append(name);
  if (place.detail) {
    const detail = document.createElement('span');
    detail.textContent = place.detail;
    label.append(detail);
  }
  const stem = document.createElement('i');
  stem.className = 'place-highlight-stem';
  stem.setAttribute('aria-hidden', 'true');
  const pin = document.createElement('i');
  pin.className = 'place-highlight-pin';
  pin.setAttribute('aria-hidden', 'true');
  marker.append(label, stem, pin);
  state.placeMarker = new maplibregl.Marker({
    element: marker,
    anchor: 'bottom',
    offset: [0, 2],
    pitchAlignment: 'viewport',
    rotationAlignment: 'viewport'
  })
    .setLngLat([place.longitude, place.latitude])
    .addTo(state.map);
}

function flyToAndLoad(target) {
  const { map } = state;
  if (!map) return;
  const requestId = ++state.mapMoveRequestId;
  window.clearTimeout(state.refreshDebounce);
  setBuildingButtonLoading(true);
  setBuildingLoadStatus('Loading visible area…', 'Loading compact building tiles…');
  setMapLoading(true, 'Loading visible area…', 'Loading building tiles…');

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

function setControlPanelVisible(isVisible) {
  elements['control-panel'].hidden = !isVisible;
  elements['open-control-panel'].hidden = isVisible;
  if (isVisible) setMobileControlsOpen(false);
  if (state.map) window.setTimeout(() => state.map.resize(), 260);
}

function setPlacePanelVisible(isVisible) {
  elements['place-panel'].hidden = !isVisible;
  elements['open-place-panel'].hidden = isVisible;
  if (!isVisible) {
    state.placeSearchAbortController?.abort();
    cancelPlaceSuggestions();
    hidePlaceSearchResults();
  }
}

function setSunPlannerVisible(isVisible) {
  const canShowPlanner = Boolean(state.sunPlannerStatus?.enabled);
  const visible = isVisible && canShowPlanner;
  elements['sun-planner'].hidden = !visible;
  elements['open-sun-planner'].hidden = !canShowPlanner || visible;
  if (!visible) {
    state.sunPlannerAbortController?.abort();
    return;
  }
  if (window.matchMedia('(max-width: 780px)').matches && !elements['place-panel'].hidden) {
    setPlacePanelVisible(false);
  }
  elements['sun-planner-prompt'].focus();
}

function setLive(value) {
  state.live = value;
  elements['live-toggle'].checked = value;
  updateNowcastControl();
  if (!value) {
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
  scheduleLocalShadowPreview();
  window.clearTimeout(state.refreshDebounce);
  state.refreshDebounce = window.setTimeout(() => refreshConditions(), CONDITIONS_DEBOUNCE_MS);
}

function visibleBoundsKey() {
  const bounds = state.map?.getBounds();
  if (!bounds) return null;
  return [bounds.getSouth(), bounds.getWest(), bounds.getNorth(), bounds.getEast()]
    .map((value) => value.toFixed(5))
    .join(',');
}

function refreshScene({ quiet = false, syncDateInput = false, showProgress = !quiet } = {}) {
  if (syncDateInput) syncInputsFromDate();
  scheduleLocalShadowPreview();
  refreshConditions();
  refreshVisibleBuildingTiles({ showProgress });
}

function scheduleLocalShadowPreview() {
  window.cancelAnimationFrame(state.shadowPreviewFrame);
  state.shadowPreviewFrame = window.requestAnimationFrame(() => {
    state.solar = localSolarPosition(state.date);
    updateSunPanel();
    updateHeaderStatus();
    updateClientShadows();
  });
}

function refreshVisibleBuildingTiles({ showProgress = false, reload = false } = {}) {
  const { map } = state;
  if (!map?.isStyleLoaded() || !map.getSource(BUILDING_TILE_SOURCE)) return;
  if (showProgress) {
    setBuildingButtonLoading(true);
    setBuildingLoadStatus('Loading visible area…', 'Loading compact building tiles…');
    setMapLoading(true, 'Loading visible area…', 'Loading building tiles…');
  }
  if (reload) map.getSource(BUILDING_TILE_SOURCE).reload?.();
  scheduleBuildingTileSync({ showProgress, force: reload });
}

function scheduleBuildingTileSync({ showProgress = false, force = false } = {}) {
  window.clearTimeout(state.buildingTileSyncTimer);
  state.buildingTileSyncTimer = window.setTimeout(() => {
    syncBuildingTiles({ showProgress, force });
  }, 40);
}

function syncBuildingTiles({ showProgress = false, force = false } = {}) {
  const { map } = state;
  if (!map?.isStyleLoaded() || !map.getSource(BUILDING_TILE_SOURCE)) return;
  if (map.getZoom() < 13) {
    const boundsKey = visibleBoundsKey();
    if (boundsKey && (state.buildingsBoundsKey !== boundsKey || state.buildingTileFingerprint !== 'zoomed-out')) {
      state.buildings = emptyFeatureCollection();
      state.shadows = emptyFeatureCollection();
      state.buildingsBoundsKey = boundsKey;
      state.buildingTileFingerprint = 'zoomed-out';
      state.buildingsDirty = true;
      state.shadowsDirty = true;
      applyMapData();
    }
    setBuildingLoadStatus('Zoom in to show buildings', 'Building tiles appear from map zoom 13.');
    setBuildingButtonLoading(false);
    setMapLoading(false);
    return;
  }
  if (!map.isSourceLoaded(BUILDING_TILE_SOURCE)) return;
  const boundsKey = visibleBoundsKey();
  if (!boundsKey) return;

  try {
    const rawFeatures = map.querySourceFeatures(BUILDING_TILE_SOURCE, { sourceLayer: BUILDING_TILE_LAYER });
    const features = tileBuildingsForVisibleMap(rawFeatures);
    const fingerprint = buildingTileFingerprint(features);
    if (!force && state.buildingsBoundsKey === boundsKey && state.buildingTileFingerprint === fingerprint) return;

    state.buildings = { type: 'FeatureCollection', features };
    state.buildingsBoundsKey = boundsKey;
    state.buildingTileFingerprint = fingerprint;
    state.buildingTilesFailed = false;
    state.buildingsDirty = true;
    updateClientShadows();
    setBuildingLoadStatus(
      features.length ? `${features.length.toLocaleString()} buildings ready` : 'No buildings in this view',
      'Building tiles · visible map'
    );
    setBuildingButtonLoading(false);
    setMapLoading(false);
  } catch (error) {
    console.warn('Could not read building tiles', error);
    handleBuildingTileError(error);
  }
}

function handleBuildingTileError(error) {
  if (state.buildingTilesFailed) return;
  state.buildingTilesFailed = true;
  console.warn('Building tiles failed. Falling back to the Python API.', error);
  loadBuildingFallback();
}

async function loadBuildingFallback() {
  const boundsKey = visibleBoundsKey();
  if (!boundsKey) return;
  const requestId = ++state.buildingFallbackRequestId;
  state.buildingFallbackAbortController?.abort();
  state.buildingFallbackAbortController = new AbortController();
  setBuildingButtonLoading(true);
  setBuildingLoadStatus('Loading visible area…', 'Using the Helsinki building fallback…');
  setMapLoading(true, 'Loading visible area…', 'Loading building fallback…');

  try {
    const parameters = new URLSearchParams({ bbox: boundsKey, retry_buildings: 'true' });
    const response = await fetch(`/api/buildings?${parameters}`, { signal: state.buildingFallbackAbortController.signal });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || `The API returned ${response.status}`);
    if (requestId !== state.buildingFallbackRequestId || visibleBoundsKey() !== boundsKey) return;
    state.buildings = payload.buildings || emptyFeatureCollection();
    state.buildingsBoundsKey = boundsKey;
    state.buildingTileFingerprint = null;
    state.buildingsDirty = true;
    updateClientShadows();
    const count = Number(payload.meta?.building_count) || 0;
    const source = payload.meta?.source === 'fallback' ? 'Starter map fallback' : 'Helsinki building service fallback';
    setBuildingLoadStatus(
      count ? `${count.toLocaleString()} buildings ready` : 'No buildings in this view',
      source
    );
  } catch (error) {
    if (error.name === 'AbortError') return;
    if (requestId !== state.buildingFallbackRequestId) return;
    console.warn('Building fallback failed', error);
    setBuildingLoadStatus('Couldn’t load visible area', 'Try refresh again shortly.');
    showToast(error.message || 'Could not load building data.');
  } finally {
    if (requestId === state.buildingFallbackRequestId) {
      setBuildingButtonLoading(false);
      setMapLoading(false);
    }
  }
}

async function refreshConditions() {
  const requestId = ++state.conditionsRequestId;
  const requestedAt = state.date.toISOString();
  state.conditionsAbortController?.abort();
  state.conditionsAbortController = new AbortController();
  const parameters = new URLSearchParams({ at: requestedAt, live: String(state.live) });

  try {
    const response = await fetch(`/api/conditions?${parameters}`, { signal: state.conditionsAbortController.signal });
    const conditions = await response.json();
    if (!response.ok) throw new Error(conditions.detail || `The API returned ${response.status}`);
    if (requestId !== state.conditionsRequestId || requestedAt !== state.date.toISOString()) return;
    applyConditions(conditions);
  } catch (error) {
    if (error.name === 'AbortError' || requestId !== state.conditionsRequestId) return;
    console.warn('Conditions request failed', error);
    state.weather = state.live ? unavailableWeather() : clearSkyPotentialWeather();
    updateWeatherPanel();
    updateHeaderStatus();
    applyMapData();
  }
}

function applyConditions(conditions, { render = true } = {}) {
  state.solar = conditions.solar;
  state.sunTimes = conditions.sun_times;
  state.weather = conditions.weather || null;
  updateSunPanel();
  updateWeatherPanel();
  updateHeaderStatus();
  if (render) updateClientShadows();
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
    : state.showClearSkyPotential
      ? `Clear-sky potential · ${nowcastProbability}% direct sun next hour`
      : `${nowcastProbability}% direct sun next hour`;
  const potentialSuffix = state.showClearSkyPotential
    ? ' The map shows where the sun could reach if clouds open. It does not confirm a visible shadow right now.'
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
    && weather.available;
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
  if (state.showClearSkyPotential) return 0.58;
  const weather = state.weather;
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
    : 'Refresh buildings';
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

function tileBuildingsForVisibleMap(rawFeatures) {
  const bounds = shadowAwareVisibleBounds();
  const seen = new Set();
  const features = [];
  for (const rawFeature of rawFeatures) {
    const polygons = tileFeaturePolygons(rawFeature.geometry);
    const properties = rawFeature.properties || {};
    polygons.forEach((polygon, polygonIndex) => {
      const ring = normaliseRing(polygon[0]);
      if (!ring || !ringOverlapsBounds(ring, bounds)) return;
      const id = tileBuildingId(rawFeature, ring, polygonIndex);
      if (seen.has(id)) return;
      seen.add(id);
      features.push({
        type: 'Feature',
        properties: {
          id,
          name: properties.name || 'Helsinki building',
          height: normaliseHeight(properties.render_height),
          source: 'vector tile'
        },
        geometry: { type: 'Polygon', coordinates: [ring] }
      });
    });
  }
  return features;
}

function tileFeaturePolygons(geometry) {
  if (!geometry || !Array.isArray(geometry.coordinates)) return [];
  if (geometry.type === 'Polygon') return [geometry.coordinates];
  if (geometry.type === 'MultiPolygon') return geometry.coordinates;
  return [];
}

function normaliseRing(rawRing) {
  if (!Array.isArray(rawRing) || rawRing.length < 3) return null;
  const ring = rawRing
    .filter((point) => Array.isArray(point) && Number.isFinite(point[0]) && Number.isFinite(point[1]))
    .map(([longitude, latitude]) => [longitude, latitude]);
  if (ring.length < 3) return null;
  const lastPoint = ring[ring.length - 1];
  if (ring[0][0] !== lastPoint[0] || ring[0][1] !== lastPoint[1]) ring.push([...ring[0]]);
  return ring;
}

function tileBuildingId(feature, ring, polygonIndex) {
  const sourceId = feature.id === undefined || feature.id === null ? 'shape' : String(feature.id);
  const shape = ring.map(([longitude, latitude]) => `${longitude.toFixed(6)},${latitude.toFixed(6)}`).join('|');
  return `${sourceId}:${polygonIndex}:${shape}`;
}

function buildingTileFingerprint(features) {
  return features.map((feature) => feature.properties.id).sort().join(',');
}

function shadowAwareVisibleBounds() {
  const bounds = state.map.getBounds();
  const latitude = state.map.getCenter().lat;
  const latitudePadding = MAX_SHADOW_METERS / METERS_PER_DEGREE_LAT;
  const longitudePadding = latitudePadding / Math.max(0.2, Math.cos(toRadians(latitude)));
  return {
    south: bounds.getSouth() - latitudePadding,
    west: bounds.getWest() - longitudePadding,
    north: bounds.getNorth() + latitudePadding,
    east: bounds.getEast() + longitudePadding
  };
}

function ringOverlapsBounds(ring, bounds) {
  const longitudes = ring.map(([longitude]) => longitude);
  const latitudes = ring.map(([, latitude]) => latitude);
  return !(
    Math.max(...longitudes) < bounds.west
    || Math.min(...longitudes) > bounds.east
    || Math.max(...latitudes) < bounds.south
    || Math.min(...latitudes) > bounds.north
  );
}

function updateClientShadows() {
  state.shadows = {
    type: 'FeatureCollection',
    features: createClientShadows(state.buildings.features, state.solar)
  };
  state.shadowsDirty = true;
  applyMapData();
}

function createClientShadows(buildings, sun) {
  if (!sun || sun.altitude <= 0 || sun.altitude > 88) return [];
  return buildings.map((building) => createClientShadow(building, sun)).filter(Boolean);
}

function createClientShadow(building, sun) {
  const ring = building.geometry?.coordinates?.[0];
  if (!Array.isArray(ring) || ring.length < 4) return null;
  const footprint = ring.slice(0, -1);
  if (footprint.length < 3) return null;
  const height = normaliseHeight(building.properties?.height);
  const distance = Math.min(MAX_SHADOW_METERS, height / Math.tan(toRadians(sun.altitude)));
  const bearing = (sun.azimuth + 180) % 360;
  const projected = footprint.map((point) => shiftCoordinate(point, distance, bearing));
  const hull = convexHull([...footprint, ...projected]);
  if (hull.length < 3) return null;
  return {
    type: 'Feature',
    properties: {
      building: building.properties?.name || 'Building',
      height,
      length: Math.round(distance)
    },
    geometry: { type: 'Polygon', coordinates: [[...hull, hull[0]]] }
  };
}

function shiftCoordinate(point, distance, bearing) {
  const [longitude, latitude] = point;
  const radians = toRadians(bearing);
  const north = Math.cos(radians) * distance;
  const east = Math.sin(radians) * distance;
  return [
    longitude + east / (METERS_PER_DEGREE_LAT * Math.cos(toRadians(latitude))),
    latitude + north / METERS_PER_DEGREE_LAT
  ];
}

function convexHull(points) {
  const sorted = [...points].sort(([leftLongitude, leftLatitude], [rightLongitude, rightLatitude]) => (
    leftLongitude - rightLongitude || leftLatitude - rightLatitude
  ));
  if (sorted.length <= 1) return sorted;
  const cross = (origin, pointA, pointB) => (
    (pointA[0] - origin[0]) * (pointB[1] - origin[1])
    - (pointA[1] - origin[1]) * (pointB[0] - origin[0])
  );
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) lower.pop();
    lower.push(point);
  }
  const upper = [];
  for (const point of [...sorted].reverse()) {
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) upper.pop();
    upper.push(point);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function localSolarPosition(date) {
  const local = getZonedParts(date, HELSINKI.timeZone);
  const dayOfYear = dayOfYearFor(local.year, local.month, local.day);
  const daysInYear = isLeapYear(local.year) ? 366 : 365;
  const hour = local.hour + local.minute / 60 + local.second / 3_600;
  const gamma = 2 * Math.PI / daysInYear * (dayOfYear - 1 + (hour - 12) / 24);
  const equationOfTime = 229.18 * (
    0.000075
    + 0.001868 * Math.cos(gamma)
    - 0.032077 * Math.sin(gamma)
    - 0.014615 * Math.cos(2 * gamma)
    - 0.040849 * Math.sin(2 * gamma)
  );
  const declination = (
    0.006918
    - 0.399912 * Math.cos(gamma)
    + 0.070257 * Math.sin(gamma)
    - 0.006758 * Math.cos(2 * gamma)
    + 0.000907 * Math.sin(2 * gamma)
    - 0.002697 * Math.cos(3 * gamma)
    + 0.00148 * Math.sin(3 * gamma)
  );
  const utcOffsetMinutes = getTimeZoneOffset(date, HELSINKI.timeZone);
  const trueSolarTime = (hour * 60 + equationOfTime + 4 * HELSINKI.lng - utcOffsetMinutes + 1_440) % 1_440;
  const hourAngle = toRadians(trueSolarTime / 4 - 180);
  const latitude = toRadians(HELSINKI.lat);
  const cosineZenith = Math.max(-1, Math.min(1,
    Math.sin(latitude) * Math.sin(declination)
    + Math.cos(latitude) * Math.cos(declination) * Math.cos(hourAngle)
  ));
  const altitude = 90 - toDegrees(Math.acos(cosineZenith));
  const azimuth = (toDegrees(Math.atan2(
    Math.sin(hourAngle),
    Math.cos(hourAngle) * Math.sin(latitude) - Math.tan(declination) * Math.cos(latitude)
  )) + 180) % 360;
  return {
    altitude: roundSolar(altitude),
    azimuth: roundSolar(azimuth),
    declination: roundSolar(toDegrees(declination))
  };
}

function dayOfYearFor(year, month, day) {
  return Math.floor((Date.UTC(year, month - 1, day) - Date.UTC(year, 0, 1)) / 86_400_000) + 1;
}

function isLeapYear(year) {
  return year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
}

function roundSolar(value) {
  return Math.round(value * 100_000) / 100_000;
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
function toDegrees(value) { return value * 180 / Math.PI; }
function pad(value) { return String(value).padStart(2, '0'); }
function formatDegrees(value) { return value >= 0 ? Math.round(value) : `−${Math.abs(Math.round(value))}`; }

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add('show');
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => elements.toast.classList.remove('show'), 4_000);
}
