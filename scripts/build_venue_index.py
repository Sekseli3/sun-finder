"""Build the local Ollama embedding index for Sunfinder's venue catalogue."""

from __future__ import annotations

from pathlib import Path
import sys

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.sun_planner import AssistantSettings, OllamaClient, VenueRetriever, load_environment_file, load_venues


VENUE_CATALOGUE_PATH = APP_ROOT / "backend" / "venue_data" / "helsinki_terraces.json"
INDEX_PATH = APP_ROOT / ".sunfinder" / "venue_index.json"


def main() -> None:
    load_environment_file(APP_ROOT / ".env")
    settings = AssistantSettings.from_environment()
    venues = load_venues(VENUE_CATALOGUE_PATH)
    retriever = VenueRetriever(
        venues=venues,
        index_path=INDEX_PATH,
        client=OllamaClient(settings),
    )
    count = retriever.rebuild()
    print(f"Indexed {count} Helsinki venue notes with {settings.embedding_model}.")


if __name__ == "__main__":
    main()
