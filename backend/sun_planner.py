"""Local LLM and retrieval helpers for Sunfinder's outing planner.

The planner deliberately keeps facts separate from language generation. Ollama
parses a freeform request and writes a short answer. Building geometry,
distance, weather, and venue source links stay deterministic Python data.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, ValidationError


DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_CHAT_MODEL = "qwen3:8b"
DEFAULT_EMBEDDING_MODEL = "qwen3-embedding:0.6b"
DEFAULT_LLM_PROVIDER = "ollama"
DEFAULT_VLLM_CHAT_BASE_URL = "http://127.0.0.1:8000/v1"
DEFAULT_VLLM_EMBEDDING_BASE_URL = "http://127.0.0.1:8001/v1"
DEFAULT_VLLM_API_KEY = "sunfinder-local"
DEFAULT_VLLM_CHAT_MODEL = "Qwen/Qwen3-8B"
DEFAULT_VLLM_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_ASSISTANT_TIMEOUT_SECONDS = 45
DEFAULT_VENUE_RADIUS_METERS = 2_000
PLANNER_LOCAL_VENUE_RADIUS_METERS = 1_000
PLANNER_VENUE_LIMIT = 6
EARTH_RADIUS_METERS = 6_371_000


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama service or a configured model is unavailable."""


class SunPlanIntent(BaseModel):
    """The narrow, validated part of an outing request that an LLM may decide."""

    model_config = ConfigDict(extra="forbid")

    anchor_query: str | None = Field(default=None, max_length=100)
    requested_time: datetime | None = None
    time_relation: Literal["at", "before"] = "at"
    venue_kind: str = Field(default="terrace_or_cafe", max_length=40)


class SunPlanRequest(BaseModel):
    """Browser context sent with one planning request."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=2, max_length=500)
    map_latitude: float = Field(ge=-90, le=90)
    map_longitude: float = Field(ge=-180, le=180)
    selected_time: datetime


@dataclass(frozen=True)
class AssistantSettings:
    enabled: bool
    ollama_base_url: str
    chat_model: str
    embedding_model: str
    timeout_seconds: int
    provider: Literal["ollama", "vllm"] = "ollama"
    vllm_chat_base_url: str = DEFAULT_VLLM_CHAT_BASE_URL
    vllm_embedding_base_url: str = DEFAULT_VLLM_EMBEDDING_BASE_URL
    vllm_api_key: str = DEFAULT_VLLM_API_KEY

    @classmethod
    def from_environment(cls) -> "AssistantSettings":
        provider = os.environ.get("SUNFINDER_LLM_PROVIDER", DEFAULT_LLM_PROVIDER).strip().casefold()
        if provider not in {"ollama", "vllm"}:
            raise ValueError("SUNFINDER_LLM_PROVIDER must be either ollama or vllm")
        if provider == "vllm":
            chat_model = os.environ.get("SUNFINDER_VLLM_CHAT_MODEL", DEFAULT_VLLM_CHAT_MODEL)
            embedding_model = os.environ.get("SUNFINDER_VLLM_EMBEDDING_MODEL", DEFAULT_VLLM_EMBEDDING_MODEL)
        else:
            chat_model = os.environ.get("SUNFINDER_CHAT_MODEL", DEFAULT_CHAT_MODEL)
            embedding_model = os.environ.get("SUNFINDER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        return cls(
            enabled=environment_flag("SUNFINDER_ASSISTANT_ENABLED"),
            ollama_base_url=os.environ.get("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL).rstrip("/"),
            chat_model=chat_model,
            embedding_model=embedding_model,
            timeout_seconds=int(os.environ.get("SUNFINDER_ASSISTANT_TIMEOUT_SECONDS", DEFAULT_ASSISTANT_TIMEOUT_SECONDS)),
            provider=provider,
            vllm_chat_base_url=os.environ.get("SUNFINDER_VLLM_CHAT_BASE_URL", DEFAULT_VLLM_CHAT_BASE_URL).rstrip("/"),
            vllm_embedding_base_url=os.environ.get("SUNFINDER_VLLM_EMBEDDING_BASE_URL", DEFAULT_VLLM_EMBEDDING_BASE_URL).rstrip("/"),
            vllm_api_key=os.environ.get("SUNFINDER_VLLM_API_KEY", DEFAULT_VLLM_API_KEY),
        )


@dataclass(frozen=True)
class Venue:
    venue_id: str
    name: str
    area: str
    kind: str
    latitude: float
    longitude: float
    terrace_note: str
    source_label: str
    source_url: str

    @property
    def document(self) -> str:
        return "\n".join(
            (
                self.name,
                f"Area: {self.area}",
                f"Type: {self.kind}",
                f"Outdoor note: {self.terrace_note}",
                f"Source: {self.source_label} {self.source_url}",
            )
        )

    def as_public_dict(self) -> dict[str, Any]:
        return {
            "id": self.venue_id,
            "name": self.name,
            "area": self.area,
            "kind": self.kind,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "terrace_note": self.terrace_note,
            "source": {"label": self.source_label, "url": self.source_url},
        }


@dataclass(frozen=True)
class RetrievedVenueDocument:
    venue_id: str
    score: float
    text: str
    source_label: str
    source_url: str


class PlannerModelClient(Protocol):
    """The narrow client contract shared by Ollama and vLLM."""

    settings: AssistantSettings

    def available_models(self) -> set[str]: ...

    def available_chat_models(self) -> set[str]: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...

    def structured_intent(self, *, message: str, selected_time: datetime, current_time: datetime) -> SunPlanIntent: ...

    def write_answer(self, *, request: str, facts: dict[str, Any], retrieved_documents: list[RetrievedVenueDocument]) -> str: ...


def structured_intent_prompt(*, message: str, selected_time: datetime, current_time: datetime, schema: dict[str, Any]) -> str:
    return "\n".join(
        (
            "Extract a Helsinki terrace or cafe outing request.",
            f"Current Helsinki time: {current_time.isoformat()}",
            f"Map selected Helsinki time: {selected_time.isoformat()}",
            "Use requested_time only when the person supplied a time.",
            "Interpret after work as 18:00 Helsinki time.",
            "Use anchor_query for a place, area, address, starting point, or departure point explicitly mentioned.",
            "For example, 'leaving from Karhupuisto' means anchor_query is 'Karhupuisto'.",
            "For here, nearby, or no location, leave anchor_query null.",
            "For 'before 14:00' or 'leaving at 14:00 and wanting sun before', set requested_time to 14:00 and time_relation to 'before'.",
            "Otherwise use time_relation 'at'.",
            "Use venue_kind 'bar' for beer, lager, wine, cocktails, or other drinking requests.",
            "Use venue_kind 'cafe' for coffee, tea, cake, or bakery requests.",
            "Use venue_kind 'terrace_or_cafe' when the venue type is not clear.",
            "Return only JSON matching this schema:",
            json.dumps(schema, ensure_ascii=False),
            "Request:",
            message,
        )
    )


def recommendation_answer_prompt(*, request: str, facts: dict[str, Any], retrieved_documents: list[RetrievedVenueDocument]) -> str:
    sources = [
        {
            "venue_id": document.venue_id,
            "text": document.text,
            "source_url": document.source_url,
        }
        for document in retrieved_documents
    ]
    return "\n".join(
        (
            "Write a short, friendly Helsinki sun outing recommendation.",
            "Use only the facts and retrieved venue notes below.",
            "Do not invent opening hours, menu details, terrace size, weather, or local sun data.",
            "Never call a ranking score a sun score or a weather probability.",
            "Venue exposure comes from projected building shade. Do not attribute 'sunny through the next hour' or building shade to the city-wide weather estimate.",
            "A direct-sun probability is city-wide, covers the next hour, and only applies to an open point.",
            "If building data is unavailable, do not call any venue sunny or shaded. Say the choices are nearby, not confirmed sun spots.",
            "Do not mention terrace availability, outdoor seating, menus, or opening hours unless those facts are explicitly supplied.",
            "Do not add generic advice such as checking local conditions before going.",
            "Only recommend venues listed in the deterministic facts. Retrieved notes support those venues only.",
            "Distances in deterministic facts are straight-line distances from facts.anchor.name. When mentioning a distance, name that anchor.",
            "The interface shows planned_window separately. Do not say 'right now' or 'the next hour' without that exact window.",
            "Give the top recommendation first and keep the answer under 130 words.",
            "Original request:",
            request,
            "Deterministic facts:",
            json.dumps(facts, ensure_ascii=False),
            "Retrieved venue notes:",
            json.dumps(sources, ensure_ascii=False),
        )
    )


class OllamaClient:
    """Small stdlib client so the local model protocol stays visible in code."""

    def __init__(self, settings: AssistantSettings) -> None:
        self.settings = settings

    def available_models(self) -> set[str]:
        payload = self._post("/api/tags", None, method="GET")
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise OllamaUnavailableError("Ollama did not return its installed models")
        return {
            str(model.get("name"))
            for model in models
            if isinstance(model, dict) and isinstance(model.get("name"), str)
        }

    def available_chat_models(self) -> set[str]:
        return self.available_models()

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = self._post(
            "/api/embed",
            {"model": self.settings.embedding_model, "input": texts},
        )
        embeddings = payload.get("embeddings") if isinstance(payload, dict) else None
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise OllamaUnavailableError("Ollama returned invalid embedding data")
        vectors: list[list[float]] = []
        for embedding in embeddings:
            if not isinstance(embedding, list) or not embedding:
                raise OllamaUnavailableError("Ollama returned an empty embedding")
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as error:
                raise OllamaUnavailableError("Ollama returned a non-numeric embedding") from error
            if not all(math.isfinite(value) for value in vector):
                raise OllamaUnavailableError("Ollama returned a non-finite embedding")
            vectors.append(vector)
        return vectors

    def structured_intent(self, *, message: str, selected_time: datetime, current_time: datetime) -> SunPlanIntent:
        schema = SunPlanIntent.model_json_schema()
        prompt = structured_intent_prompt(
            message=message,
            selected_time=selected_time,
            current_time=current_time,
            schema=schema,
        )
        payload = self._post(
            "/api/chat",
            {
                "model": self.settings.chat_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a careful structured data extractor. Never add facts that are not in the request.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "format": schema,
                "stream": False,
                "options": {"temperature": 0},
                "keep_alive": "5m",
            },
        )
        content = message_content(payload)
        try:
            return SunPlanIntent.model_validate_json(content)
        except ValidationError as error:
            raise OllamaUnavailableError("The local model returned an invalid plan") from error

    def write_answer(self, *, request: str, facts: dict[str, Any], retrieved_documents: list[RetrievedVenueDocument]) -> str:
        prompt = recommendation_answer_prompt(
            request=request,
            facts=facts,
            retrieved_documents=retrieved_documents,
        )
        payload = self._post(
            "/api/chat",
            {
                "model": self.settings.chat_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a cautious recommendation writer. Facts from the app always win over fluent wording.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "options": {"temperature": 0.2},
                "keep_alive": "5m",
            },
        )
        content = message_content(payload).strip()
        if not content:
            raise OllamaUnavailableError("The local model returned an empty answer")
        return content

    def _post(self, path: str, body: dict[str, Any] | None, *, method: str = "POST") -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.settings.ollama_base_url}{path}",
            data=data,
            method=method,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310 - configured local Ollama endpoint only
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OllamaUnavailableError("Could not reach the local Ollama service") from error
        if not isinstance(payload, dict):
            raise OllamaUnavailableError("Ollama returned an unexpected response")
        if isinstance(payload.get("error"), str):
            raise OllamaUnavailableError(payload["error"])
        return payload


class VllmClient:
    """OpenAI-compatible vLLM client for the same narrow planner contract."""

    def __init__(self, settings: AssistantSettings) -> None:
        self.settings = settings

    def available_models(self) -> set[str]:
        base_urls = {self.settings.vllm_chat_base_url, self.settings.vllm_embedding_base_url}
        models: set[str] = set()
        for base_url in base_urls:
            models.update(self._models_at(base_url))
        return models

    def available_chat_models(self) -> set[str]:
        return self._models_at(self.settings.vllm_chat_base_url)

    def embed(self, texts: list[str]) -> list[list[float]]:
        payload = self._request(
            self.settings.vllm_embedding_base_url,
            "/embeddings",
            {
                "model": self.settings.embedding_model,
                "input": texts,
                "encoding_format": "float",
            },
        )
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list) or len(entries) != len(texts):
            raise OllamaUnavailableError("vLLM returned invalid embedding data")
        vectors_by_index: dict[int, list[float]] = {}
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("index"), int):
                raise OllamaUnavailableError("vLLM returned an embedding without an index")
            embedding = entry.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise OllamaUnavailableError("vLLM returned an empty embedding")
            try:
                vector = [float(value) for value in embedding]
            except (TypeError, ValueError) as error:
                raise OllamaUnavailableError("vLLM returned a non-numeric embedding") from error
            if not all(math.isfinite(value) for value in vector):
                raise OllamaUnavailableError("vLLM returned a non-finite embedding")
            vectors_by_index[entry["index"]] = vector
        try:
            return [vectors_by_index[index] for index in range(len(texts))]
        except KeyError as error:
            raise OllamaUnavailableError("vLLM returned incomplete embedding data") from error

    def structured_intent(self, *, message: str, selected_time: datetime, current_time: datetime) -> SunPlanIntent:
        schema = SunPlanIntent.model_json_schema()
        prompt = structured_intent_prompt(
            message=message,
            selected_time=selected_time,
            current_time=current_time,
            schema=schema,
        )
        content = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful structured data extractor. Never add facts that are not in the request.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=256,
            schema=schema,
        )
        try:
            return SunPlanIntent.model_validate_json(content)
        except ValidationError as error:
            raise OllamaUnavailableError("The local vLLM model returned an invalid plan") from error

    def write_answer(self, *, request: str, facts: dict[str, Any], retrieved_documents: list[RetrievedVenueDocument]) -> str:
        prompt = recommendation_answer_prompt(
            request=request,
            facts=facts,
            retrieved_documents=retrieved_documents,
        )
        content = self._chat(
            messages=[
                {
                    "role": "system",
                    "content": "You are a cautious recommendation writer. Facts from the app always win over fluent wording.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=384,
        ).strip()
        if not content:
            raise OllamaUnavailableError("The local vLLM model returned an empty answer")
        return content

    def _chat(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        schema: dict[str, Any] | None = None,
    ) -> str:
        body: dict[str, Any] = {
            "model": self.settings.chat_model,
            "messages": messages,
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Qwen3 thinks by default. The planner needs short structured
            # extraction and grounded wording, not a visible reasoning trace.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "sun_plan_intent",
                    "schema": schema,
                },
            }
        payload = self._request(self.settings.vllm_chat_base_url, "/chat/completions", body)
        return vllm_message_content(payload)

    def _models_at(self, base_url: str) -> set[str]:
        payload = self._request(base_url, "/models", None, method="GET")
        entries = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise OllamaUnavailableError("vLLM did not return its served models")
        return {
            str(entry.get("id"))
            for entry in entries
            if isinstance(entry, dict) and isinstance(entry.get("id"), str)
        }

    def _request(
        self,
        base_url: str,
        path: str,
        body: dict[str, Any] | None,
        *,
        method: str = "POST",
    ) -> dict[str, Any]:
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.settings.vllm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.vllm_api_key}"
        request = Request(
            f"{base_url}{path}",
            data=data,
            method=method,
            headers=headers,
        )
        try:
            with urlopen(request, timeout=self.settings.timeout_seconds) as response:  # noqa: S310 - configured local vLLM endpoint only
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise OllamaUnavailableError(vllm_http_error_message(error)) from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise OllamaUnavailableError("Could not reach the local vLLM service") from error
        if not isinstance(payload, dict):
            raise OllamaUnavailableError("vLLM returned an unexpected response")
        if isinstance(payload.get("error"), str):
            raise OllamaUnavailableError(payload["error"])
        return payload


def vllm_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise OllamaUnavailableError("vLLM returned no chat choices")
    message = choices[0].get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise OllamaUnavailableError("vLLM returned no chat message")
    return content


def vllm_http_error_message(error: HTTPError) -> str:
    try:
        raw_detail = error.read().decode("utf-8")
        payload = json.loads(raw_detail)
        detail = payload.get("error") if isinstance(payload, dict) else raw_detail
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        detail = ""
    if isinstance(detail, dict):
        detail = detail.get("message") or json.dumps(detail, ensure_ascii=False)
    if isinstance(detail, str) and detail.strip():
        return f"vLLM returned HTTP {error.code}: {detail.strip()[:300]}"
    return f"vLLM returned HTTP {error.code}"


def build_assistant_client(settings: AssistantSettings) -> PlannerModelClient:
    """Select the local serving engine without changing planner logic."""
    if settings.provider == "vllm":
        return VllmClient(settings)
    return OllamaClient(settings)


class VenueRetriever:
    """A compact local vector index for a deliberately small venue catalogue."""

    def __init__(self, *, venues: tuple[Venue, ...], index_path: Path, client: PlannerModelClient) -> None:
        self.venues = venues
        self.index_path = index_path
        self.client = client

    def rebuild(self) -> int:
        documents = [venue.document for venue in self.venues]
        vectors = self.client.embed(documents)
        payload = {
            "fingerprint": venue_catalogue_fingerprint(self.venues, self.client.settings.embedding_model),
            "model": self.client.settings.embedding_model,
            "entries": [
                {"venue_id": venue.venue_id, "embedding": vector}
                for venue, vector in zip(self.venues, vectors, strict=True)
            ],
        }
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(payload), encoding="utf-8")
        return len(vectors)

    def search(self, query: str, *, limit: int = 4) -> list[RetrievedVenueDocument]:
        index = self._load_or_rebuild()
        query_vector = self.client.embed([query])[0]
        venues_by_id = {venue.venue_id: venue for venue in self.venues}
        matches: list[RetrievedVenueDocument] = []
        for entry in index:
            if not isinstance(entry, dict):
                continue
            venue_id = entry.get("venue_id")
            embedding = entry.get("embedding")
            venue = venues_by_id.get(venue_id)
            if venue is None or not isinstance(embedding, list):
                continue
            try:
                score = cosine_similarity(query_vector, [float(value) for value in embedding])
            except (TypeError, ValueError):
                continue
            matches.append(
                RetrievedVenueDocument(
                    venue_id=venue.venue_id,
                    score=round(score, 4),
                    text=venue.document,
                    source_label=venue.source_label,
                    source_url=venue.source_url,
                )
            )
        return sorted(matches, key=lambda match: match.score, reverse=True)[:limit]

    def _load_or_rebuild(self) -> list[dict[str, Any]]:
        expected_fingerprint = venue_catalogue_fingerprint(self.venues, self.client.settings.embedding_model)
        try:
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            self.rebuild()
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("fingerprint") != expected_fingerprint:
            self.rebuild()
            payload = json.loads(self.index_path.read_text(encoding="utf-8"))
        entries = payload.get("entries") if isinstance(payload, dict) else None
        if not isinstance(entries, list):
            raise OllamaUnavailableError("The local venue index is invalid")
        return entries


def load_venues(path: Path) -> tuple[Venue, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Could not load Sunfinder venue catalogue: {path}") from error
    raw_venues = payload.get("venues") if isinstance(payload, dict) else None
    if not isinstance(raw_venues, list):
        raise RuntimeError("Sunfinder venue catalogue must contain a venues list")
    venues: list[Venue] = []
    for raw_venue in raw_venues:
        if not isinstance(raw_venue, dict):
            continue
        try:
            venue = Venue(
                venue_id=clean_text(raw_venue["id"]),
                name=clean_text(raw_venue["name"]),
                area=clean_text(raw_venue["area"]),
                kind=clean_text(raw_venue["kind"]),
                latitude=float(raw_venue["latitude"]),
                longitude=float(raw_venue["longitude"]),
                terrace_note=clean_text(raw_venue["terrace_note"]),
                source_label=clean_text(raw_venue["source_label"]),
                source_url=clean_text(raw_venue["source_url"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("Sunfinder venue catalogue contains an invalid venue") from error
        if not math.isfinite(venue.latitude) or not math.isfinite(venue.longitude):
            raise RuntimeError("Sunfinder venue catalogue contains an invalid coordinate")
        venues.append(venue)
    if not venues:
        raise RuntimeError("Sunfinder venue catalogue is empty")
    return tuple(venues)


def venues_near(venues: Iterable[Venue], latitude: float, longitude: float, *, radius_meters: float = DEFAULT_VENUE_RADIUS_METERS) -> list[tuple[Venue, float]]:
    nearby = [
        (venue, haversine_meters(latitude, longitude, venue.latitude, venue.longitude))
        for venue in venues
    ]
    return sorted(
        ((venue, distance) for venue, distance in nearby if distance <= radius_meters),
        key=lambda item: item[1],
    )


def venue_preference_for_request(message: str, model_venue_kind: str) -> str:
    """Choose a small deterministic venue filter from an outing request.

    The model still extracts the broader intent, but a direct request for a
    cold lager should never drift into a nearby coffee shop just because its
    vector note happened to be similar.
    """
    words = request_words(message)
    if words & {"beer", "lager", "ale", "pint", "brewery", "olut", "bisse", "kalja"}:
        return "beer"
    if words & {"wine", "viini"}:
        return "wine"
    if words & {"cocktail", "cocktails", "drink", "drinks", "bar", "pub"}:
        return "bar"
    if words & {"coffee", "cafe", "café", "kahvi", "espresso", "latte", "tea", "cake", "bakery"}:
        return "cafe"
    if words & {"terrace", "patio", "outdoor", "outside", "ulko", "terassi"}:
        return "terrace"

    model_words = request_words(model_venue_kind)
    if "bar" in model_words:
        return "bar"
    if "cafe" in model_words or "café" in model_words:
        return "cafe"
    if "terrace" in model_words:
        return "terrace"
    return "any"


def planner_venues_near(
    venues: Iterable[Venue],
    latitude: float,
    longitude: float,
    *,
    preference: str,
) -> list[tuple[Venue, float]]:
    """Return a compact, relevant local set for a planner shadow check.

    The former planner envelope included every catalogue item within 2 km.
    In central Helsinki that became a multi-square-kilometre building request.
    Start with genuinely local choices, then expand only when the catalogue is
    sparse around the selected map point.
    """
    nearby = venues_near(venues, latitude, longitude)
    matching = [item for item in nearby if venue_matches_preference(item[0], preference)]
    if preference != "any" and not matching:
        return []
    candidates = matching if preference != "any" else nearby
    local = [item for item in candidates if item[1] <= PLANNER_LOCAL_VENUE_RADIUS_METERS]
    return (local or candidates)[:PLANNER_VENUE_LIMIT]


def venue_matches_preference(venue: Venue, preference: str) -> bool:
    kind_words = request_words(venue.kind)
    is_bar = "bar" in kind_words
    is_brewery = "brewery" in kind_words
    if preference == "beer":
        # A wine bar is a useful match for a wine request, but its catalogue
        # label alone does not justify presenting it as a place for lager.
        return is_brewery or (is_bar and "wine" not in kind_words)
    if preference == "wine":
        return "wine" in kind_words or is_bar
    if preference == "bar":
        return is_bar or is_brewery
    if preference == "cafe":
        return "cafe" in kind_words or "café" in kind_words or "bakery" in kind_words
    if preference == "terrace":
        return "terrace" in kind_words
    return True


def request_words(value: str) -> set[str]:
    return set("".join(character if character.isalnum() else " " for character in value.casefold()).split())


def time_relation_for_request(message: str, model_time_relation: str) -> Literal["at", "before"]:
    """Keep an explicit departure deadline from being lost in model phrasing."""
    words = request_words(message)
    if model_time_relation == "before" or ("before" in words and {"leave", "leaving", "departure"} & words):
        return "before"
    return "at"


def fallback_anchor_hint(message: str) -> str | None:
    """Extract a likely place after an unambiguous location cue as a safety net.

    This is deliberately only a fallback when the structured extractor did not
    return a place. The resulting text is still resolved by the normal place
    search rather than being treated as a hard-coded venue alias.
    """
    cue = re.search(r"\b(?:from|near|around|by)\s+(.+)", message, flags=re.IGNORECASE)
    if not cue:
        return None
    candidate = cue.group(1)
    boundary = re.search(
        r"\b(?:i|we|you|to|for|before|after|tomorrow|today|tonight|with|and|then|but|where|what)\b",
        candidate,
        flags=re.IGNORECASE,
    )
    if boundary:
        candidate = candidate[:boundary.start()]
    candidate = " ".join(candidate.strip(" ,.!?;:").split())
    if not candidate or request_words(candidate) <= {"helsinki"}:
        return None
    return candidate[:100]


def rank_venues_by_sun(
    venues_with_distances: Iterable[tuple[Venue, float]],
    shadow_samples: list[tuple[datetime, list[dict[str, Any]], bool]],
    *,
    building_geometry_available: bool,
) -> list[dict[str, Any]]:
    """Score terrace points using fixed shadow samples and straight-line distance."""
    recommendations: list[dict[str, Any]] = []
    for venue, distance_meters in venues_with_distances:
        daylight_samples = 0
        exposed_samples = 0
        sample_details: list[dict[str, Any]] = []
        for sample_time, shadows, is_daylight in shadow_samples:
            shaded = any(point_in_feature(venue.longitude, venue.latitude, shadow) for shadow in shadows)
            if is_daylight:
                daylight_samples += 1
                if not shaded:
                    exposed_samples += 1
            sample_details.append(
                {
                    "at": sample_time.isoformat(),
                    "daylight": is_daylight,
                    "building_shade": shaded if is_daylight and building_geometry_available else None,
                }
            )

        exposure_fraction = exposed_samples / daylight_samples if daylight_samples else 0.0
        distance_score = max(0.0, 1 - distance_meters / DEFAULT_VENUE_RADIUS_METERS)
        ranking_score = 0.75 * exposure_fraction + 0.25 * distance_score if building_geometry_available else distance_score
        if daylight_samples == 0:
            ranking_basis = "distance only because the sun is below the horizon"
        elif building_geometry_available:
            ranking_basis = "projected building shade over the next hour and distance"
        else:
            ranking_basis = "distance only because building data is unavailable"
        recommendations.append(
            {
                "venue": venue.as_public_dict(),
                "distance_meters": round(distance_meters),
                "ranking_score": round(ranking_score * 100),
                "sun_coverage_percent": round(exposure_fraction * 100)
                if building_geometry_available and daylight_samples
                else None,
                "ranking_basis": ranking_basis,
                "exposure": exposure_label(
                    daylight_samples=daylight_samples,
                    exposed_samples=exposed_samples,
                    building_geometry_available=building_geometry_available,
                ),
                "sample_details": sample_details,
            }
        )
    return sorted(recommendations, key=lambda recommendation: (-recommendation["ranking_score"], recommendation["distance_meters"]))[:3]


def point_in_feature(longitude: float, latitude: float, feature: dict[str, Any]) -> bool:
    geometry = feature.get("geometry") if isinstance(feature, dict) else None
    if not isinstance(geometry, dict) or geometry.get("type") != "Polygon":
        return False
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates or not isinstance(coordinates[0], list):
        return False
    return point_in_ring(longitude, latitude, coordinates[0])


def point_in_ring(longitude: float, latitude: float, ring: list[Any]) -> bool:
    """Ray casting point-in-polygon check for the shadow hulls used by the app."""
    inside = False
    previous = len(ring) - 1
    for index, coordinate in enumerate(ring):
        if not isinstance(coordinate, list) or len(coordinate) < 2:
            return False
        previous_coordinate = ring[previous]
        if not isinstance(previous_coordinate, list) or len(previous_coordinate) < 2:
            return False
        try:
            current_longitude, current_latitude = float(coordinate[0]), float(coordinate[1])
            previous_longitude, previous_latitude = float(previous_coordinate[0]), float(previous_coordinate[1])
        except (TypeError, ValueError):
            return False
        intersects = ((current_latitude > latitude) != (previous_latitude > latitude)) and (
            longitude
            < (previous_longitude - current_longitude) * (latitude - current_latitude)
            / (previous_latitude - current_latitude)
            + current_longitude
        )
        if intersects:
            inside = not inside
        previous = index
    return inside


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return -1.0
    return dot_product / (left_norm * right_norm)


def haversine_meters(latitude_a: float, longitude_a: float, latitude_b: float, longitude_b: float) -> float:
    latitude_delta = math.radians(latitude_b - latitude_a)
    longitude_delta = math.radians(longitude_b - longitude_a)
    latitude_a_radians = math.radians(latitude_a)
    latitude_b_radians = math.radians(latitude_b)
    haversine = (
        math.sin(latitude_delta / 2) ** 2
        + math.cos(latitude_a_radians) * math.cos(latitude_b_radians) * math.sin(longitude_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(haversine))


def exposure_label(*, daylight_samples: int, exposed_samples: int, building_geometry_available: bool) -> str:
    if daylight_samples == 0:
        return "after sunset"
    if not building_geometry_available:
        return "building data unavailable"
    if exposed_samples == daylight_samples:
        return "sunny through the next hour"
    if exposed_samples:
        return "some sun in the next hour"
    return "in projected building shade"


def venue_catalogue_fingerprint(venues: Iterable[Venue], embedding_model: str) -> str:
    source = json.dumps(
        {
            "embedding_model": embedding_model,
            "venues": [venue.document for venue in venues],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def environment_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def load_environment_file(path: Path) -> None:
    """Load simple local key=value settings without overriding real process env."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return
    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        os.environ[name] = value.strip().strip('"').strip("'")


def clean_text(value: Any) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError("Expected non-empty text")
    return text


def message_content(payload: dict[str, Any]) -> str:
    message = payload.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str):
        raise OllamaUnavailableError("Ollama returned no chat message")
    return content
