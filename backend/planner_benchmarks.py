"""Provider-neutral evaluation helpers for the local outing planner.

The benchmark deliberately evaluates the LLM-owned intent extraction step in
isolation. Building geometry, weather and ranking are deterministic Python
work, so including their live network calls would make an Ollama versus vLLM
comparison noisy and unfair.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from backend.sun_planner import SunPlanIntent


HELSINKI_TIME_ZONE = ZoneInfo("Europe/Helsinki")
INTENT_FIELDS = ("anchor_query", "requested_time", "time_relation", "venue_kind")


@dataclass(frozen=True)
class PlannerIntentBenchmarkCase:
    """One labelled request and its fixed planning context."""

    case_id: str
    description: str
    tags: tuple[str, ...]
    message: str
    current_time: datetime
    selected_time: datetime
    anchor_terms: tuple[str, ...] | None
    requested_time: datetime | None
    time_relation: str
    venue_kind: str

    def expected_as_dict(self) -> dict[str, Any]:
        return {
            "anchor_query": None if self.anchor_terms is None else {"contains": list(self.anchor_terms)},
            "requested_time": None if self.requested_time is None else self.requested_time.isoformat(),
            "time_relation": self.time_relation,
            "venue_kind": self.venue_kind,
        }


@dataclass(frozen=True)
class PlannerIntentBenchmarkSuite:
    """The versioned suite stored in ``benchmarks/planner_intents.json``."""

    name: str
    description: str
    schema_version: int
    cases: tuple[PlannerIntentBenchmarkCase, ...]


def load_planner_intent_benchmark(path: Path) -> PlannerIntentBenchmarkSuite:
    """Load and validate the human-readable benchmark fixture."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"Planner benchmark fixture does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"Planner benchmark fixture is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError("Planner benchmark fixture must be a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise ValueError(f"Unsupported planner benchmark schema version: {schema_version!r}")
    name = clean_required_text(payload.get("name"), "name")
    description = clean_required_text(payload.get("description"), "description")
    default_context = payload.get("default_context")
    if not isinstance(default_context, dict):
        raise ValueError("Planner benchmark fixture must include default_context")
    default_current_time = parse_datetime(default_context.get("current_time"), "default_context.current_time")
    default_selected_time = parse_datetime(default_context.get("selected_time"), "default_context.selected_time")

    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Planner benchmark fixture must include at least one case")
    cases = tuple(
        parse_case(
            raw_case,
            default_current_time=default_current_time,
            default_selected_time=default_selected_time,
        )
        for raw_case in raw_cases
    )
    ids = [case.case_id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Planner benchmark case ids must be unique")
    return PlannerIntentBenchmarkSuite(
        name=name,
        description=description,
        schema_version=schema_version,
        cases=cases,
    )


def parse_case(
    raw_case: Any,
    *,
    default_current_time: datetime,
    default_selected_time: datetime,
) -> PlannerIntentBenchmarkCase:
    """Validate one fixture record into a small immutable object."""
    if not isinstance(raw_case, dict):
        raise ValueError("Every planner benchmark case must be a JSON object")
    expected = raw_case.get("expected")
    if not isinstance(expected, dict):
        raise ValueError(f"Planner benchmark {raw_case.get('id')!r} must include expected")

    raw_tags = raw_case.get("tags")
    if not isinstance(raw_tags, list) or not raw_tags or not all(isinstance(tag, str) and tag.strip() for tag in raw_tags):
        raise ValueError(f"Planner benchmark {raw_case.get('id')!r} must include non-empty tags")
    anchor_terms = parse_anchor_expectation(expected.get("anchor_query"), raw_case.get("id"))
    requested_time = parse_optional_datetime(expected.get("requested_time"), f"{raw_case.get('id')}.expected.requested_time")
    time_relation = clean_required_text(expected.get("time_relation"), f"{raw_case.get('id')}.expected.time_relation")
    venue_kind = clean_required_text(expected.get("venue_kind"), f"{raw_case.get('id')}.expected.venue_kind")
    if time_relation not in {"at", "before"}:
        raise ValueError(f"Planner benchmark {raw_case.get('id')!r} has unsupported time_relation {time_relation!r}")

    current_time = parse_optional_datetime(raw_case.get("current_time"), f"{raw_case.get('id')}.current_time")
    selected_time = parse_optional_datetime(raw_case.get("selected_time"), f"{raw_case.get('id')}.selected_time")
    return PlannerIntentBenchmarkCase(
        case_id=clean_required_text(raw_case.get("id"), "case.id"),
        description=clean_required_text(raw_case.get("description"), "case.description"),
        tags=tuple(tag.strip() for tag in raw_tags),
        message=clean_required_text(raw_case.get("message"), "case.message"),
        current_time=current_time or default_current_time,
        selected_time=selected_time or default_selected_time,
        anchor_terms=anchor_terms,
        requested_time=requested_time,
        time_relation=time_relation,
        venue_kind=venue_kind,
    )


def evaluate_planner_intent(case: PlannerIntentBenchmarkCase, actual: SunPlanIntent) -> dict[str, Any]:
    """Return field-level, explainable scores for one parsed intent."""
    actual_time = actual.requested_time
    checks = {
        "anchor_query": anchor_matches(actual.anchor_query, case.anchor_terms),
        "requested_time": datetime_matches(actual_time, case.requested_time),
        "time_relation": actual.time_relation == case.time_relation,
        "venue_kind": actual.venue_kind == case.venue_kind,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "expected": case.expected_as_dict(),
        "actual": {
            "anchor_query": actual.anchor_query,
            "requested_time": None if actual_time is None else actual_time.isoformat(),
            "time_relation": actual.time_relation,
            "venue_kind": actual.venue_kind,
        },
    }


def summarize_planner_intent_evaluations(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise field accuracy without hiding individual errors."""
    total = len(evaluations)
    field_accuracy = {
        field: (sum(bool(evaluation.get("checks", {}).get(field)) for evaluation in evaluations) / total if total else 0.0)
        for field in INTENT_FIELDS
    }
    passed = sum(bool(evaluation.get("passed")) for evaluation in evaluations)
    return {
        "cases": total,
        "passed": passed,
        "pass_rate": passed / total if total else 0.0,
        "field_accuracy": field_accuracy,
    }


def clean_required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Planner benchmark {field} must be a non-empty string")
    return value.strip()


def parse_anchor_expectation(value: Any, case_id: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Planner benchmark {case_id!r} expected.anchor_query must be null or an object")
    contains = value.get("contains")
    if not isinstance(contains, list) or not contains or not all(isinstance(term, str) and term.strip() for term in contains):
        raise ValueError(f"Planner benchmark {case_id!r} expected.anchor_query.contains must be a non-empty string list")
    return tuple(term.strip() for term in contains)


def parse_optional_datetime(value: Any, field: str) -> datetime | None:
    if value is None:
        return None
    return parse_datetime(value, field)


def parse_datetime(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Planner benchmark {field} must be an ISO datetime string")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"Planner benchmark {field} is not an ISO datetime: {value!r}") from error
    if parsed.tzinfo is None:
        raise ValueError(f"Planner benchmark {field} must include a time zone")
    return parsed


def anchor_matches(actual: str | None, expected_terms: tuple[str, ...] | None) -> bool:
    if expected_terms is None:
        return actual is None
    if actual is None:
        return False
    normalized_actual = normalize_text(actual)
    return all(normalize_text(term) in normalized_actual for term in expected_terms)


def datetime_matches(actual: datetime | None, expected: datetime | None) -> bool:
    if expected is None:
        return actual is None
    if actual is None or actual.tzinfo is None:
        return False
    return actual.astimezone(HELSINKI_TIME_ZONE).replace(second=0, microsecond=0) == expected.astimezone(HELSINKI_TIME_ZONE).replace(second=0, microsecond=0)


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.casefold())
    without_accents = "".join(character for character in decomposed if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", without_accents)).strip()
