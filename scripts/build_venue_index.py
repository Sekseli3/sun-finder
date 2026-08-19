"""Build the local embedding index for Sunfinder's venue catalogue."""

from __future__ import annotations

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.qdrant_venues import QdrantSettings, QdrantUnavailableError, QdrantVenueRetriever
from backend.sun_planner import AssistantSettings, VenueRetriever, build_assistant_client, load_environment_file, load_venues


VENUE_CATALOGUE_PATH = APP_ROOT / "backend" / "venue_data" / "helsinki_terraces.json"
INDEX_PATH = APP_ROOT / ".sunfinder" / "venue_index.json"


def main() -> None:
    load_environment_file(APP_ROOT / ".env")
    settings = AssistantSettings.from_environment()
    venues = load_venues(VENUE_CATALOGUE_PATH)
    client = build_assistant_client(settings)
    qdrant_settings = QdrantSettings.from_environment()
    if qdrant_settings.enabled:
        try:
            count = QdrantVenueRetriever(settings=qdrant_settings, client=client).rebuild(venues)
        except QdrantUnavailableError as error:
            raise SystemExit(str(error)) from error
        print(f"Indexed {count} seed venue notes in Qdrant with {settings.embedding_model} via {settings.provider}.")
        return
    retriever = VenueRetriever(venues=venues, index_path=INDEX_PATH, client=client)
    count = retriever.rebuild()
    print(f"Indexed {count} Helsinki venue notes locally with {settings.embedding_model} via {settings.provider}.")


if __name__ == "__main__":
    main()
