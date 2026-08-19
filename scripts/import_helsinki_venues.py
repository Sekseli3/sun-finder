#!/usr/bin/env python3
"""Fetch Helsinki food and drink venues from OpenStreetMap and index them in Qdrant."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.osm_venues import fetch_helsinki_osm_venues
from backend.qdrant_venues import QdrantSettings, QdrantUnavailableError, QdrantVenueRetriever
from backend.sun_planner import AssistantSettings, build_assistant_client, load_environment_file


DEFAULT_SNAPSHOT_PATH = APP_ROOT / ".sunfinder" / "helsinki_osm_venues.json"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import named Helsinki cafes, restaurants, bars, pubs, and biergartens into Qdrant.")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT_PATH, help="Write the imported OpenStreetMap venue data here.")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    load_environment_file(APP_ROOT / ".env")
    qdrant_settings = QdrantSettings.from_environment()
    if not qdrant_settings.enabled:
        raise SystemExit("Set SUNFINDER_QDRANT_ENABLED=true in .env before importing venues")
    try:
        venues = fetch_helsinki_osm_venues()
    except Exception as error:  # Network and Overpass errors vary by server.
        raise SystemExit(f"Could not fetch Helsinki venues from OpenStreetMap: {error}") from error
    if not venues:
        raise SystemExit("OpenStreetMap returned no named Helsinki food and drink venues")
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    args.snapshot.write_text(
        json.dumps({"source": "OpenStreetMap via Overpass", "venues": [venue.as_public_dict() for venue in venues]}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    settings = AssistantSettings.from_environment()
    try:
        count = QdrantVenueRetriever(settings=qdrant_settings, client=build_assistant_client(settings)).rebuild(venues)
    except QdrantUnavailableError as error:
        raise SystemExit(str(error)) from error
    print(f"Imported and indexed {count} Helsinki venues in Qdrant collection {qdrant_settings.collection!r}.")
    print(f"Saved the source snapshot to {args.snapshot}.")


if __name__ == "__main__":
    main()
