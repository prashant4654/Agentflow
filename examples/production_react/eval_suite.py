#!/usr/bin/env python3
"""
Production Evaluation Suite for React Agent App

Ready-to-run evaluation suite with pre-configured test cases for the
React production agent. Includes performance monitoring and reporting.

Run with:
    python eval_suite.py

Or with options:
    python eval_suite.py --sample 0.5     # Run 50% of tests
    python eval_suite.py --parallel 10    # Use 10 parallel workers
    python eval_suite.py --verbose        # Enable detailed logging
"""

import asyncio
import json
import sys
import time
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

from evaluation import (
    ReactEvaluator,
    EvalCase,
    ToolCallExpectation,
    MetricType,
    format_eval_report,
    save_report_json,
)

try:
    from react_sync import app as compiled_graph
except ImportError:
    print("Error: Could not import compiled_graph from react_sync.py")
    print("Make sure react_sync.py is in the same directory")
    sys.exit(1)


# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================


def get_critical_test_cases() -> list[EvalCase]:
    """
    Critical tests that must pass for deployment.
    Run these on every PR.
    """
    return [
        EvalCase(
            case_id="basic_greeting",
            user_input="Hello, how are you?",
            expected_response_contains="helpful",
            max_latency_ms=2000,
            description="Agent should respond to basic greeting",
        ),
        EvalCase(
            case_id="helpful_assistant",
            user_input="Can you help me?",
            expected_response_contains="help",
            max_latency_ms=2000,
            description="Agent should identify itself as helpful",
        ),
    ]


def get_tool_test_cases() -> list[EvalCase]:
    """
    Tests for tool usage and integration.
    Validates that the agent correctly invokes tools.
    """
    return [
        EvalCase(
            case_id="weather_tool_call",
            user_input="Please call the get_weather function for New York City",
            expected_response_contains="weather",
            expected_tools=[
                ToolCallExpectation(
                    name="get_weather",
                    arg_keys=["location"],  # Verify location arg exists
                    optional=False,
                )
            ],
            max_latency_ms=5000,
            description="Should call get_weather tool",
        ),
        EvalCase(
            case_id="weather_exact_location",
            user_input="What's the weather in London?",
            expected_response_contains="London",
            expected_tools=[
                ToolCallExpectation(
                    name="get_weather",
                    arg_keys=["location"],
                )
            ],
            max_latency_ms=5000,
            description="Should call get_weather with correct location",
        ),
    ]


def get_performance_test_cases() -> list[EvalCase]:
    """
    Tests focused on performance and latency.
    """
    return [
        EvalCase(
            case_id="fast_response_1",
            user_input="Hi",
            expected_response_contains="response",
            max_latency_ms=1500,
            description="Should respond quickly to simple input",
        ),
        EvalCase(
            case_id="fast_response_2",
            user_input="Hello?",
            expected_response_contains="response",
            max_latency_ms=1500,
            description="Should handle minimal input quickly",
        ),
        EvalCase(
            case_id="moderate_latency",
            user_input="Tell me about your capabilities",
            expected_response_contains="response",
            max_latency_ms=3000,
            description="Moderate latency acceptable for complex queries",
        ),
    ]


def get_edge_case_tests() -> list[EvalCase]:
    """
    Tests for edge cases and error handling.
    """
    return [
        EvalCase(
            case_id="empty_like_input",
            user_input=" ",
            expected_response_contains="",
            allow_errors=True,
            description="Should handle whitespace gracefully",
        ),
        EvalCase(
            case_id="very_long_input",
            user_input="x" * 1000,
            expected_response_contains="",
            allow_errors=True,
            max_latency_ms=10000,
            description="Should handle long input without crashing",
        ),
    ]


def get_all_test_cases() -> list[EvalCase]:
    """Combine all test cases."""
    return (
        get_critical_test_cases()
        + get_tool_test_cases()
        + get_performance_test_cases()
        + get_edge_case_tests()
    )


# ============================================================================
# EVALUATION EXECUTION
# ============================================================================


async def run_critical_evaluation() -> dict:
    """Run only critical tests (fast feedback)."""
    print("\n" + "=" * 70)
    print("CRITICAL EVALUATION SUITE")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=False)
    cases = get_critical_test_cases()

    print(f"\nRunning {len(cases)} critical test cases...")
    report = await evaluator.evaluate(cases, parallel=True, max_parallel=5)

    return {
        "suite": "critical",
        "report": report,
        "cases": cases,
    }


async def run_full_evaluation() -> dict:
    """Run complete test suite."""
    print("\n" + "=" * 70)
    print("FULL EVALUATION SUITE")
    print("=" * 70)

    evaluator = ReactEvaluator(compiled_graph, verbose=False)
    cases = get_all_test_cases()

    print(f"\nRunning {len(cases)} test cases...")

    # Group tests by category
    categories = {
        "critical": get_critical_test_cases(),
        "tools": get_tool_test_cases(),
        "performance": get_performance_test_cases(),
        "edge_cases": get_edge_case_tests(),
    }

    all_results = []
    category_reports = {}

    for category_name, category_cases in categories.items():
        print(f"  {category_name.upper()}: {len(category_cases)} tests...", end="", flush=True)

        report = await evaluator.evaluate(category_cases, parallel=True, max_parallel=5)
        all_results.extend(report.case_results)
        category_reports[category_name] = {
            "passed": report.passed_cases,
            "total": report.total_cases,
            "pass_rate": report.pass_rate,
        }

        status = "✓" if report.pass_rate >= 0.95 else "✗"
        print(f" {status} {report.pass_rate * 100:.0f}%")

    # Create aggregate report
    total_passed = sum(r["passed"] for r in category_reports.values())
    total_cases = sum(r["total"] for r in category_reports.values())

    from evaluation import EvalReport

    aggregate_report = EvalReport(
        total_cases=total_cases,
        passed_cases=total_passed,
        failed_cases=total_cases - total_passed,
        case_results=all_results,
        aggregate_metrics={},
    )

    return {
        "suite": "full",
        "report": aggregate_report,
        "cases": cases,
        "category_breakdown": category_reports,
    }


async def run_sampled_evaluation(sample_rate: float = 0.5) -> dict:
    """Run sampled tests for quick validation."""
    print("\n" + "=" * 70)
    print(f"SAMPLED EVALUATION ({int(sample_rate * 100)}%)")
    print("=" * 70)

    import random

    evaluator = ReactEvaluator(compiled_graph, verbose=False)
    all_cases = get_all_test_cases()

    sampled_cases = random.sample(all_cases, int(len(all_cases) * sample_rate))

    print(f"\nRunning {len(sampled_cases)} test cases (sampled from {len(all_cases)})...")
    report = await evaluator.evaluate(sampled_cases, parallel=True, max_parallel=5)

    return {
        "suite": "sampled",
        "report": report,
        "cases": sampled_cases,
        "sample_rate": sample_rate,
    }


# ============================================================================
# REPORTING
# ============================================================================


def print_detailed_report(suite_result: dict, verbose: bool = False):
    """Print detailed evaluation report."""
    report = suite_result["report"]

    print(format_eval_report(report))

    # Category breakdown if available
    if "category_breakdown" in suite_result:
        print("\nCATEGORY BREAKDOWN:")
        for category, stats in suite_result["category_breakdown"].items():
            status = "✓" if stats["pass_rate"] >= 0.95 else "✗"
            print(
                f"  {category.upper():15} {status} "
                f"{stats['passed']}/{stats['total']} "
                f"({stats['pass_rate'] * 100:.0f}%)"
            )

    # Metrics per type
    if report.aggregate_metrics:
        print("\nMETRIC SCORES:")
        for metric_type, score in sorted(report.aggregate_metrics.items()):
            bar_length = int(score * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {metric_type.value:20} [{bar}] {score * 100:6.1f}%")

    # Verbose: Show failed cases details
    if verbose and report.failed_cases_detail:
        print("\nFAILED CASES DETAILS:")
        for result in report.failed_cases_detail:
            print(f"\n  Case: {result.case_id}")
            print(f"    Input: {result.user_input[:80]}")
            if result.error:
                print(f"    Error: {result.error}")
            print(f"    Latency: {result.execution_time_ms:.0f}ms")
            if result.actual_response:
                preview = result.actual_response[:100].replace("\n", " ")
                print(f"    Response: {preview}...")


def save_evaluation_report(suite_result: dict, output_file: str = "eval_report.json"):
    """Save evaluation report to JSON file."""
    report = suite_result["report"]

    data = {
        "timestamp": report.timestamp,
        "suite": suite_result["suite"],
        "duration_seconds": report.duration_seconds,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "pass_rate": report.pass_rate,
        "aggregate_metrics": {k.value: v for k, v in report.aggregate_metrics.items()},
        "case_summary": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "execution_time_ms": r.execution_time_ms,
                "error": r.error,
            }
            for r in report.case_results
        ],
    }

    if "category_breakdown" in suite_result:
        data["category_breakdown"] = suite_result["category_breakdown"]

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nReport saved to {output_file}")


def check_pass_criteria(report, min_pass_rate: float = 0.95) -> bool:
    """Check if evaluation meets deployment criteria."""
    if report.pass_rate < min_pass_rate:
        print(f"\n❌ DEPLOYMENT BLOCKED")
        print(
            f"   Pass rate {report.pass_rate * 100:.1f}% below minimum {min_pass_rate * 100:.0f}%"
        )
        return False

    print(f"\n✅ DEPLOYMENT APPROVED")
    print(f"   Pass rate {report.pass_rate * 100:.1f}% meets minimum {min_pass_rate * 100:.0f}%")
    return True


# ============================================================================
# MAIN EXECUTION
# ============================================================================


async def main():
    parser = ArgumentParser(description="Run agent evaluation suite")
    parser.add_argument(
        "--suite",
        choices=["critical", "full", "sampled"],
        default="critical",
        help="Test suite to run (default: critical)",
    )
    parser.add_argument(
        "--sample",
        type=float,
        default=0.5,
        help="Sampling rate for 'sampled' suite (0.0-1.0, default: 0.5)",
    )
    parser.add_argument("--parallel", type=int, default=5, help="Max parallel workers (default: 5)")
    parser.add_argument("--output", default="eval_report.json", help="Output JSON report file")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.95,
        help="Minimum pass rate for deployment (0.0-1.0, default: 0.95)",
    )

    args = parser.parse_args()

    # Run selected suite
    start_time = time.time()

    if args.suite == "critical":
        result = await run_critical_evaluation()
    elif args.suite == "full":
        result = await run_full_evaluation()
    else:  # sampled
        result = await run_sampled_evaluation(args.sample)

    # Add duration
    result["report"].duration_seconds = time.time() - start_time

    # Print results
    print_detailed_report(result, verbose=args.verbose)

    # Save report
    save_evaluation_report(result, args.output)

    # Check deployment criteria
    passed = check_pass_criteria(result["report"], args.min_pass_rate)

    # Exit with appropriate code
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nEvaluation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
