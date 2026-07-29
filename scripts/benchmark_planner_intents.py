#!/usr/bin/env python3
"""Run the fixed planner-intent suite against the configured local LLM.

The fixture and evaluator are deliberately provider-neutral, so Ollama and
vLLM use the same request set and scoring rules.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from backend.planner_benchmarks import (  # noqa: E402
    PlannerIntentBenchmarkCase,
    evaluate_planner_intent,
    load_planner_intent_benchmark,
    summarize_planner_intent_evaluations,
)
from backend.sun_planner import (  # noqa: E402
    AssistantSettings,
    OllamaUnavailableError,
    PlannerModelClient,
    build_assistant_client,
    load_environment_file,
)


DEFAULT_CASES_PATH = APP_ROOT / "benchmarks" / "planner_intents.json"
DEFAULT_OUTPUT_DIRECTORY = APP_ROOT / ".sunfinder" / "benchmarks"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the local planner's structured-intent extraction against a fixed Helsinki suite.",
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH, help="Path to the benchmark fixture JSON file.")
    parser.add_argument("--case", dest="case_ids", action="append", help="Run only one case id. Repeat this option for several cases.")
    parser.add_argument("--repeat", type=positive_int, default=1, help="Run every selected case this many times.")
    parser.add_argument("--label", help="A label stored in the result file, for example ollama-q4 or vllm-bf16.")
    parser.add_argument("--output", type=Path, help="Where to save the detailed JSON report. Defaults under .sunfinder/benchmarks.")
    parser.add_argument(
        "--min-pass-rate",
        type=percentage,
        help="Exit with status 1 when whole-case accuracy falls below this fraction. Example: 0.90",
    )
    parser.add_argument("--show-all", action="store_true", help="Print successful cases as well as failures.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    suite = load_planner_intent_benchmark(args.cases)
    load_environment_file(APP_ROOT / ".env")
    cases = select_cases(suite.cases, args.case_ids)
    settings = AssistantSettings.from_environment()
    client = build_assistant_client(settings)
    ensure_chat_model_available(client, settings.chat_model)
    label = args.label or settings.provider

    started_at = datetime.now(UTC)
    results: list[dict[str, Any]] = []
    for attempt in range(1, args.repeat + 1):
        for case in cases:
            results.append(run_case(client, case, attempt=attempt))

    evaluations = [result["evaluation"] for result in results]
    summary = summarize_planner_intent_evaluations(evaluations)
    latency = latency_summary([result["latency_ms"] for result in results])
    report = {
        "schema_version": 1,
        "suite": {
            "name": suite.name,
            "description": suite.description,
            "schema_version": suite.schema_version,
            "source": str(args.cases),
        },
        "run": {
            "label": label,
            "provider": settings.provider,
            "model": settings.chat_model,
            "base_url": settings.ollama_base_url,
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "repeat": args.repeat,
        },
        "summary": {**summary, "latency_ms": latency},
        "results": results,
    }
    output_path = args.output or default_output_path(label, started_at)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_report(report, output_path, show_all=args.show_all)

    if args.min_pass_rate is not None and summary["pass_rate"] < args.min_pass_rate:
        return 1
    return 0


def ensure_chat_model_available(client: PlannerModelClient, chat_model: str) -> None:
    try:
        installed_models = client.available_chat_models()
    except OllamaUnavailableError as error:
        raise SystemExit(f"Could not contact the configured local LLM before benchmarking: {error}") from error
    if chat_model not in installed_models:
        raise SystemExit(f"Configured chat model {chat_model!r} is not installed. Run make assistant-setup first.")


def select_cases(cases: tuple[PlannerIntentBenchmarkCase, ...], requested_ids: list[str] | None) -> tuple[PlannerIntentBenchmarkCase, ...]:
    if not requested_ids:
        return cases
    cases_by_id = {case.case_id: case for case in cases}
    missing = [case_id for case_id in requested_ids if case_id not in cases_by_id]
    if missing:
        raise SystemExit(f"Unknown benchmark case id: {', '.join(missing)}")
    return tuple(cases_by_id[case_id] for case_id in requested_ids)


def run_case(client: PlannerModelClient, case: PlannerIntentBenchmarkCase, *, attempt: int) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        intent = client.structured_intent(
            message=case.message,
            selected_time=case.selected_time,
            current_time=case.current_time,
        )
        evaluation = evaluate_planner_intent(case, intent)
        error = None
    except OllamaUnavailableError as caught_error:
        evaluation = {
            "passed": False,
            "checks": {
                "anchor_query": False,
                "requested_time": False,
                "time_relation": False,
                "venue_kind": False,
            },
            "expected": case.expected_as_dict(),
            "actual": None,
        }
        error = str(caught_error)
    latency_ms = round((time.perf_counter() - started) * 1_000, 2)
    return {
        "case_id": case.case_id,
        "description": case.description,
        "tags": list(case.tags),
        "attempt": attempt,
        "message": case.message,
        "context": {
            "current_time": case.current_time.isoformat(),
            "selected_time": case.selected_time.isoformat(),
        },
        "latency_ms": latency_ms,
        "evaluation": evaluation,
        "error": error,
    }


def latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(percentile(values, 95), 2),
        "max": round(max(values), 2),
    }


def percentile(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def default_output_path(label: str, started_at: datetime) -> Path:
    safe_label = "".join(character if character.isalnum() or character in {"-", "_"} else "-" for character in label).strip("-")
    safe_label = safe_label or "run"
    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIRECTORY / f"planner-intents-{safe_label}-{timestamp}.json"


def print_report(report: dict[str, Any], output_path: Path, *, show_all: bool) -> None:
    summary = report["summary"]
    field_accuracy = summary["field_accuracy"]
    latency = summary["latency_ms"]
    print(f"{report['suite']['name']} · {report['run']['label']}")
    print(f"Model: {report['run']['model']} via {report['run']['provider']}")
    print(f"Whole-case accuracy: {summary['passed']}/{summary['cases']} ({summary['pass_rate']:.0%})")
    print(
        "Field accuracy: "
        + ", ".join(f"{field} {field_accuracy[field]:.0%}" for field in ("anchor_query", "requested_time", "time_relation", "venue_kind"))
    )
    print(f"Latency: median {latency['median']:.0f} ms · p95 {latency['p95']:.0f} ms · max {latency['max']:.0f} ms")
    print(f"Saved detailed results to {output_path}")

    displayed = 0
    for result in report["results"]:
        passed = result["evaluation"]["passed"]
        if passed and not show_all:
            continue
        status = "PASS" if passed else "FAIL"
        print(f"{status} {result['case_id']} · {result['latency_ms']:.0f} ms")
        if not passed:
            evaluation = result["evaluation"]
            failed_fields = [field for field, correct in evaluation["checks"].items() if not correct]
            print(f"  failed: {', '.join(failed_fields)}")
            print(f"  expected: {json.dumps(evaluation['expected'], ensure_ascii=False)}")
            print(f"  actual: {json.dumps(evaluation['actual'], ensure_ascii=False)}")
            if result["error"]:
                print(f"  error: {result['error']}")
        displayed += 1
        if not show_all and displayed >= 10:
            remaining = sum(not item["evaluation"]["passed"] for item in report["results"]) - displayed
            if remaining > 0:
                print(f"… plus {remaining} more failures in the JSON report")
            break


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def percentage(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
