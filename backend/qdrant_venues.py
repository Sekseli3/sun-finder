"""Qdrant storage for the larger Helsinki venue catalogue.

The public map does not need Qdrant. This module is used only by the optional
local outing planner and keeps a small JSON catalogue available as a fallback.
"""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any

from backend.sun_planner import OllamaUnavailableError, PlannerModelClient, RetrievedVenueDocument, Venue, environment_flag


DEFAULT_QDRANT_URL = "http://127.0.0.1:6333"
DEFAULT_QDRANT_COLLECTION = "sunfinder_helsinki_venues"
EMBEDDING_BATCH_SIZE = 32
QDRANT_UPLOAD_BATCH_SIZE = 128


class QdrantUnavailableError(RuntimeError):
    """Raised when the optional local Qdrant service cannot be used."""


@dataclass(frozen=True)
class QdrantSettings:
    enabled: bool
    url: str
    api_key: str | None
    collection: str

    @classmethod
    def from_environment(cls) -> "QdrantSettings":
        api_key = os.environ.get("SUNFINDER_QDRANT_API_KEY", "").strip() or None
        return cls(
            enabled=environment_flag("SUNFINDER_QDRANT_ENABLED"),
            url=os.environ.get("SUNFINDER_QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/"),
            api_key=api_key,
            collection=os.environ.get("SUNFINDER_QDRANT_COLLECTION", DEFAULT_QDRANT_COLLECTION).strip()
            or DEFAULT_QDRANT_COLLECTION,
        )


class QdrantVenueRetriever:
    """Persistent venue data and vectors for the local planner."""

    def __init__(self, *, settings: QdrantSettings, client: PlannerModelClient, qdrant_client: Any | None = None) -> None:
        self.settings = settings
        self.client = client
        self._qdrant_client = qdrant_client

    def rebuild(self, venues: tuple[Venue, ...]) -> int:
        if not venues:
            raise QdrantUnavailableError("No Helsinki venues were supplied for Qdrant")
        vectors = embed_documents(self.client, [venue.document for venue in venues])
        if len(vectors) != len(venues) or not vectors or not vectors[0]:
            raise QdrantUnavailableError("The embedding model returned invalid venue vectors")
        vector_size = len(vectors[0])
        if any(len(vector) != vector_size for vector in vectors):
            raise QdrantUnavailableError("The embedding model returned inconsistent vector sizes")

        models = qdrant_models()
        database = self.database
        if database.collection_exists(self.settings.collection):
            database.delete_collection(self.settings.collection)
        database.create_collection(
            collection_name=self.settings.collection,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )
        database.create_payload_index(
            collection_name=self.settings.collection,
            field_name="kind",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        database.create_payload_index(
            collection_name=self.settings.collection,
            field_name="location",
            field_schema=models.PayloadSchemaType.GEO,
        )
        points = [
            models.PointStruct(id=qdrant_point_id(venue.venue_id), vector=vector, payload=venue_payload(venue))
            for venue, vector in zip(venues, vectors, strict=True)
        ]
        database.upload_points(
            collection_name=self.settings.collection,
            points=points,
            batch_size=QDRANT_UPLOAD_BATCH_SIZE,
            wait=True,
        )
        return len(points)

    def venues(self) -> tuple[Venue, ...]:
        records: list[Any] = []
        offset: Any | None = None
        database = self.database
        while True:
            batch, offset = database.scroll(
                collection_name=self.settings.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            records.extend(batch)
            if offset is None:
                break
        venues = [venue_from_payload(record.payload) for record in records if isinstance(getattr(record, "payload", None), dict)]
        return tuple(sorted((venue for venue in venues if venue is not None), key=lambda venue: venue.name.casefold()))

    def search(self, query: str, *, limit: int = 4) -> list[RetrievedVenueDocument]:
        query_vector = self.client.embed([query])[0]
        database = self.database
        if hasattr(database, "query_points"):
            response = database.query_points(
                collection_name=self.settings.collection,
                query=query_vector,
                limit=limit,
                with_payload=True,
            )
            points = response.points
        else:  # qdrant-client before query_points
            points = database.search(
                collection_name=self.settings.collection,
                query_vector=query_vector,
                limit=limit,
                with_payload=True,
            )
        documents: list[RetrievedVenueDocument] = []
        for point in points:
            payload = getattr(point, "payload", None)
            venue = venue_from_payload(payload) if isinstance(payload, dict) else None
            if venue is None:
                continue
            documents.append(
                RetrievedVenueDocument(
                    venue_id=venue.venue_id,
                    score=round(float(getattr(point, "score", 0.0)), 4),
                    text=venue.document,
                    source_label=venue.source_label,
                    source_url=venue.source_url,
                )
            )
        return documents

    @property
    def database(self) -> Any:
        if self._qdrant_client is None:
            self._qdrant_client = create_qdrant_client(self.settings)
        return self._qdrant_client


def embed_documents(client: PlannerModelClient, documents: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(documents), EMBEDDING_BATCH_SIZE):
        batch = documents[start : start + EMBEDDING_BATCH_SIZE]
        try:
            vectors.extend(client.embed(batch))
        except OllamaUnavailableError as error:
            raise QdrantUnavailableError(str(error)) from error
    return vectors


def create_qdrant_client(settings: QdrantSettings) -> Any:
    try:
        from qdrant_client import QdrantClient
    except ImportError as error:
        raise QdrantUnavailableError("Install qdrant-client with make install before using the Qdrant venue index") from error
    try:
        client = QdrantClient(url=settings.url, api_key=settings.api_key, timeout=30)
        client.get_collections()
        return client
    except Exception as error:  # the client has several transport-specific error types
        raise QdrantUnavailableError(f"Could not reach Qdrant at {settings.url}") from error


def qdrant_models() -> Any:
    try:
        from qdrant_client import models
    except ImportError as error:
        raise QdrantUnavailableError("Install qdrant-client with make install before using the Qdrant venue index") from error
    return models


def qdrant_point_id(venue_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://sunfinder.helsinki/venues/{venue_id}"))


def venue_payload(venue: Venue) -> dict[str, Any]:
    return {
        "venue_id": venue.venue_id,
        "name": venue.name,
        "area": venue.area,
        "kind": venue.kind,
        "latitude": venue.latitude,
        "longitude": venue.longitude,
        "location": {"lat": venue.latitude, "lon": venue.longitude},
        "terrace_note": venue.terrace_note,
        "source_label": venue.source_label,
        "source_url": venue.source_url,
    }


def venue_from_payload(payload: dict[str, Any]) -> Venue | None:
    try:
        venue = Venue(
            venue_id=str(payload["venue_id"]),
            name=str(payload["name"]),
            area=str(payload["area"]),
            kind=str(payload["kind"]),
            latitude=float(payload["latitude"]),
            longitude=float(payload["longitude"]),
            terrace_note=str(payload["terrace_note"]),
            source_label=str(payload["source_label"]),
            source_url=str(payload["source_url"]),
        )
    except (KeyError, TypeError, ValueError):
        return None
    if not all((venue.venue_id, venue.name, venue.area, venue.kind, venue.source_label, venue.source_url)):
        return None
    if not math.isfinite(venue.latitude) or not math.isfinite(venue.longitude):
        return None
    return venue
