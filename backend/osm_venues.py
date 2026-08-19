"""OpenStreetMap venue import for Sunfinder's local Helsinki index."""

from __future__ import annotations

import json
from typing import Any
from urllib.request import Request, urlopen

from backend.sun_planner import Venue


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
OVERPASS_QUERY = """[out:json][timeout:180];
area["boundary"="administrative"]["name"="Helsinki"]["admin_level"="8"]->.helsinki;
(
  nwr["amenity"~"^(cafe|restaurant|bar|pub|biergarten)$"](area.helsinki);
);
out center tags;"""
OVERPASS_USER_AGENT = "Sunfinder Helsinki venue importer/1.0 (local learning project)"


def fetch_helsinki_osm_venues(endpoint: str = OVERPASS_ENDPOINT) -> tuple[Venue, ...]:
    request = Request(
        endpoint,
        data=OVERPASS_QUERY.encode("utf-8"),
        method="POST",
        headers={"Accept": "application/json", "Content-Type": "text/plain", "User-Agent": OVERPASS_USER_AGENT},
    )
    with urlopen(request, timeout=240) as response:  # noqa: S310 - fixed public Overpass endpoint
        payload = json.loads(response.read().decode("utf-8"))
    return parse_overpass_venues(payload)


def parse_overpass_venues(payload: dict[str, Any]) -> tuple[Venue, ...]:
    elements = payload.get("elements") if isinstance(payload, dict) else None
    if not isinstance(elements, list):
        raise ValueError("Overpass did not return an elements list")
    venues: dict[str, Venue] = {}
    for element in elements:
        venue = venue_from_overpass_element(element)
        if venue is not None:
            venues[venue.venue_id] = venue
    return tuple(sorted(venues.values(), key=lambda venue: venue.name.casefold()))


def venue_from_overpass_element(element: Any) -> Venue | None:
    if not isinstance(element, dict):
        return None
    tags = element.get("tags")
    element_type = element.get("type")
    element_id = element.get("id")
    if not isinstance(tags, dict) or not isinstance(element_type, str) or not isinstance(element_id, int):
        return None
    amenity = clean_tag(tags.get("amenity"))
    if amenity not in {"cafe", "restaurant", "bar", "pub", "biergarten"}:
        return None
    name = clean_tag(tags.get("name"))
    coordinate = overpass_coordinate(element)
    if name is None or coordinate is None:
        return None
    latitude, longitude = coordinate
    area = clean_tag(tags.get("addr:suburb")) or clean_tag(tags.get("addr:district")) or "Helsinki"
    kind_parts = [amenity]
    cuisine = clean_tag(tags.get("cuisine"))
    if cuisine:
        kind_parts.append(cuisine.replace(";", ", "))
    outdoor = clean_tag(tags.get("outdoor_seating"))
    note_parts = []
    if outdoor:
        note_parts.append(f"Outdoor seating: {outdoor}")
    if clean_tag(tags.get("brewery")):
        note_parts.append("Brewery tag in OpenStreetMap")
    if clean_tag(tags.get("wheelchair")):
        note_parts.append(f"Wheelchair: {clean_tag(tags.get('wheelchair'))}")
    terrace_note = ". ".join(note_parts) or "Venue details from OpenStreetMap tags"
    return Venue(
        venue_id=f"osm:{element_type}:{element_id}",
        name=name,
        area=area,
        kind=" ".join(kind_parts),
        latitude=latitude,
        longitude=longitude,
        terrace_note=terrace_note,
        source_label="OpenStreetMap",
        source_url=f"https://www.openstreetmap.org/{element_type}/{element_id}",
    )


def overpass_coordinate(element: dict[str, Any]) -> tuple[float, float] | None:
    latitude = element.get("lat")
    longitude = element.get("lon")
    if latitude is None or longitude is None:
        center = element.get("center")
        if isinstance(center, dict):
            latitude = center.get("lat")
            longitude = center.get("lon")
    try:
        return float(latitude), float(longitude)
    except (TypeError, ValueError):
        return None


def clean_tag(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())
    return cleaned or None
