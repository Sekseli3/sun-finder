from __future__ import annotations

import unittest
from datetime import datetime
from pathlib import Path

from backend.planner_benchmarks import (
    evaluate_planner_intent,
    load_planner_intent_benchmark,
    summarize_planner_intent_evaluations,
)
from backend.sun_planner import SunPlanIntent


BENCHMARK_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "planner_intents.json"


class PlannerBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.suite = load_planner_intent_benchmark(BENCHMARK_PATH)

    def test_fixture_has_a_broad_stable_suite(self) -> None:
        self.assertEqual(self.suite.schema_version, 1)
        self.assertGreaterEqual(len(self.suite.cases), 30)
        self.assertEqual(len({case.case_id for case in self.suite.cases}), len(self.suite.cases))
        tags = {tag for case in self.suite.cases for tag in case.tags}
        self.assertTrue({"finnish", "anchor", "before", "fallback", "relative-date", "beer", "cafe"}.issubset(tags))
        self.assertTrue(all(case.current_time.tzinfo is not None for case in self.suite.cases))
        self.assertTrue(all(case.selected_time.tzinfo is not None for case in self.suite.cases))

    def test_evaluator_accepts_normalized_storyville_anchor_and_equivalent_timezone(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "storyville-tomorrow-lunch")
        intent = SunPlanIntent(
            anchor_query="Storyville, Töölö",
            requested_time=datetime.fromisoformat("2026-07-30T10:00:00+00:00"),
            time_relation="at",
            venue_kind="terrace_or_cafe",
        )

        evaluation = evaluate_planner_intent(case, intent)

        self.assertTrue(evaluation["passed"])
        self.assertTrue(all(evaluation["checks"].values()))

    def test_evaluator_explains_a_wrong_field_and_summarizes_it(self) -> None:
        case = next(case for case in self.suite.cases if case.case_id == "karhupuisto-departure-deadline")
        intent = SunPlanIntent(
            anchor_query="Karhupuisto",
            requested_time=datetime.fromisoformat("2026-07-30T14:00:00+03:00"),
            time_relation="at",
            venue_kind="bar",
        )

        evaluation = evaluate_planner_intent(case, intent)
        summary = summarize_planner_intent_evaluations([evaluation])

        self.assertFalse(evaluation["passed"])
        self.assertFalse(evaluation["checks"]["time_relation"])
        self.assertEqual(summary["pass_rate"], 0.0)
        self.assertEqual(summary["field_accuracy"]["anchor_query"], 1.0)
        self.assertEqual(summary["field_accuracy"]["time_relation"], 0.0)


if __name__ == "__main__":
    unittest.main()
