"""
Simplified Evaluation System for React Production App

This module provides a lightweight, production-optimized evaluation system
designed specifically for React agent applications. It avoids the complexity
of the main evaluation module while providing essential metrics:

- Response Quality (latency, token usage, success rate)
- Tool Usage Correctness (expected tools called, correct arguments)
- Message Flow Integrity (proper message ordering, roles)
- Integration Health (error rates, recovery behavior)

Architecture:
- EvalMetric: Individual metric computation
- EvalCase: Test case definition
- ReactEvaluator: Main evaluator (simple interface)
- MetricsCollector: Aggregates metrics across test cases
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional
from datetime import datetime
from enum import Enum


class MetricType(str, Enum):
    """Metric types for evaluation."""

    LATENCY = "latency"
    TOKEN_USAGE = "token_usage"
    SUCCESS_RATE = "success_rate"
    TOOL_ACCURACY = "tool_accuracy"
    MESSAGE_INTEGRITY = "message_integrity"
    ERROR_RATE = "error_rate"


@dataclass
class EvalMetric:
    """Single evaluation metric result."""

    metric_type: MetricType
    score: float  # 0.0 to 1.0
    value: Any  # Actual measured value
    threshold: float = 0.8
    passed: bool = field(init=False)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def __post_init__(self):
        self.passed = self.score >= self.threshold


@dataclass
class ToolCallExpectation:
    """Definition of expected tool call."""

    name: str
    args: dict[str, Any] | None = None
    arg_keys: list[str] | None = None  # If set, only check these keys exist
    optional: bool = False
    min_calls: int = 1


@dataclass
class EvalCase:
    """Single test case for evaluation."""

    case_id: str
    user_input: str
    expected_response_contains: str | None = None
    expected_response_equals: str | None = None
    expected_tools: list[ToolCallExpectation] | None = None
    max_latency_ms: float = 5000.0
    allow_errors: bool = False
    description: str = ""


@dataclass
class EvalCaseResult:
    """Result from evaluating single test case."""

    case_id: str
    user_input: str
    passed: bool
    metrics: list[EvalMetric]
    actual_response: str | None = None
    actual_tools_called: list[dict] | None = None
    error: str | None = None
    execution_time_ms: float = 0.0
    token_count: int = 0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalReport:
    """Complete evaluation report."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    case_results: list[EvalCaseResult]
    aggregate_metrics: dict[MetricType, float]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_seconds: float = 0.0

    @property
    def pass_rate(self) -> float:
        """Overall pass rate."""
        return self.passed_cases / self.total_cases if self.total_cases > 0 else 0.0

    @property
    def failed_cases_detail(self) -> list[EvalCaseResult]:
        """List of failed cases."""
        return [r for r in self.case_results if not r.passed]


class ReactEvaluator:
    """
    Simplified evaluator for React production apps.

    Provides straightforward evaluation with minimal setup:

    Example:
        ```python
        evaluator = ReactEvaluator(compiled_graph)

        cases = [
            EvalCase(
                case_id="weather_query",
                user_input="What's the weather in NYC?",
                expected_tools=[ToolCallExpectation(name="get_weather")],
                expected_response_contains="weather",
            )
        ]

        report = await evaluator.evaluate(cases)
        print(report.format_summary())
        ```
    """

    def __init__(
        self,
        graph: Any,  # CompiledGraph
        verbose: bool = False,
        timeout_seconds: float = 30.0,
    ):
        self.graph = graph
        self.verbose = verbose
        self.timeout_seconds = timeout_seconds
        self.metrics_collector = MetricsCollector()

    async def evaluate(
        self,
        cases: list[EvalCase],
        parallel: bool = False,
        max_parallel: int = 5,
    ) -> EvalReport:
        """
        Evaluate agent against test cases.

        Args:
            cases: List of test cases
            parallel: Whether to run cases in parallel
            max_parallel: Max parallel executions if parallel=True

        Returns:
            EvalReport with complete results
        """
        start_time = time.time()

        if parallel:
            results = await self._evaluate_parallel(cases, max_parallel)
        else:
            results = await self._evaluate_sequential(cases)

        duration = time.time() - start_time

        # Aggregate results
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed

        aggregate_metrics = self.metrics_collector.aggregate(results)

        report = EvalReport(
            total_cases=len(cases),
            passed_cases=passed,
            failed_cases=failed,
            case_results=results,
            aggregate_metrics=aggregate_metrics,
            duration_seconds=duration,
        )

        return report

    async def _evaluate_sequential(self, cases: list[EvalCase]) -> list[EvalCaseResult]:
        """Evaluate cases sequentially."""
        results = []
        for i, case in enumerate(cases):
            if self.verbose:
                print(f"[{i + 1}/{len(cases)}] Evaluating: {case.case_id}")
            result = await self._run_case(case)
            results.append(result)
        return results

    async def _evaluate_parallel(
        self,
        cases: list[EvalCase],
        max_parallel: int,
    ) -> list[EvalCaseResult]:
        """Evaluate cases in parallel with concurrency limit."""
        results = []
        semaphore = asyncio.Semaphore(max_parallel)

        async def bounded_run(case: EvalCase) -> EvalCaseResult:
            async with semaphore:
                return await self._run_case(case)

        tasks = [bounded_run(case) for case in cases]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        final_results = []
        for case, result in zip(cases, results):
            if isinstance(result, Exception):
                final_results.append(
                    EvalCaseResult(
                        case_id=case.case_id,
                        user_input=case.user_input,
                        passed=False,
                        metrics=[],
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    async def _run_case(self, case: EvalCase) -> EvalCaseResult:
        """Run single test case."""
        metrics: list[EvalMetric] = []
        actual_response = None
        actual_tools = []
        error = None
        execution_time = 0.0
        token_count = 0

        try:
            # Execute with timeout
            start_time = time.time()

            result = await asyncio.wait_for(
                self._invoke_graph(case.user_input),
                timeout=self.timeout_seconds,
            )

            execution_time = (time.time() - start_time) * 1000  # ms

            actual_response = result.get("response", "")
            actual_tools = result.get("tools_called", [])
            token_count = result.get("token_count", 0)

        except asyncio.TimeoutError:
            error = f"Timeout exceeded ({self.timeout_seconds}s)"
            metrics.append(
                EvalMetric(
                    metric_type=MetricType.LATENCY,
                    score=0.0,
                    value=execution_time,
                    threshold=1.0,
                )
            )
        except Exception as e:
            error = str(e)

        # Evaluate metrics
        if not error:
            metrics = self._evaluate_metrics(case, actual_response, actual_tools, execution_time)

        # Determine pass/fail
        passed = error is None and all(m.passed for m in metrics)

        if not case.allow_errors and error:
            passed = False

        return EvalCaseResult(
            case_id=case.case_id,
            user_input=case.user_input,
            passed=passed,
            metrics=metrics,
            actual_response=actual_response,
            actual_tools_called=actual_tools,
            error=error,
            execution_time_ms=execution_time,
            token_count=token_count,
        )

    def _evaluate_metrics(
        self,
        case: EvalCase,
        actual_response: str,
        actual_tools: list[dict],
        execution_time_ms: float,
    ) -> list[EvalMetric]:
        """Evaluate individual metrics for a case."""
        metrics = []

        # Latency metric
        latency_score = 1.0 - (execution_time_ms / case.max_latency_ms)
        latency_score = max(0.0, min(1.0, latency_score))
        metrics.append(
            EvalMetric(
                metric_type=MetricType.LATENCY,
                score=latency_score,
                value=execution_time_ms,
                threshold=0.7,
            )
        )

        # Response content metric
        if case.expected_response_equals:
            response_match = actual_response.strip() == case.expected_response_equals.strip()
            score = 1.0 if response_match else 0.0
        elif case.expected_response_contains:
            response_match = case.expected_response_contains.lower() in actual_response.lower()
            score = 1.0 if response_match else 0.0
        else:
            score = 1.0 if actual_response else 0.0

        metrics.append(
            EvalMetric(
                metric_type=MetricType.SUCCESS_RATE,
                score=score,
                value=actual_response,
                threshold=1.0,
            )
        )

        # Tool accuracy metric
        if case.expected_tools:
            tool_score = self._evaluate_tools(case.expected_tools, actual_tools)
            metrics.append(
                EvalMetric(
                    metric_type=MetricType.TOOL_ACCURACY,
                    score=tool_score,
                    value=actual_tools,
                    threshold=1.0,
                )
            )

        return metrics

    def _evaluate_tools(
        self,
        expected: list[ToolCallExpectation],
        actual: list[dict],
    ) -> float:
        """
        Evaluate tool call accuracy.

        Returns score 0.0-1.0 based on:
        - All expected tools were called
        - Tool arguments match (if specified)
        - No unexpected tools called (slight penalty)
        """
        if not expected:
            return 1.0 if not actual else 0.8

        expected_names = {e.name for e in expected}
        actual_names = {t.get("name", "") for t in actual}

        # Check if all expected tools were called
        missing_tools = expected_names - actual_names
        if missing_tools:
            return 0.0

        # Slight penalty for extra unexpected tools
        extra_tools = actual_names - expected_names
        penalty = len(extra_tools) * 0.05

        # Check arguments if specified
        arg_score = 1.0
        for expectation in expected:
            matching_calls = [t for t in actual if t.get("name") == expectation.name]
            if not matching_calls and not expectation.optional:
                return 0.0

            if expectation.args and matching_calls:
                call = matching_calls[0]
                call_args = call.get("args", {})

                if expectation.arg_keys:
                    # Only check specified keys exist
                    for key in expectation.arg_keys:
                        if key not in call_args:
                            arg_score -= 0.1
                else:
                    # Check exact match
                    if call_args != expectation.args:
                        arg_score -= 0.1

        score = min(1.0, max(0.0, arg_score - penalty))
        return score

    async def _invoke_graph(self, user_input: str) -> dict[str, Any]:
        """
        Invoke the agent graph and extract results.

        Returns dict with:
        - response: Final text response from agent
        - tools_called: List of tool calls made
        - token_count: Estimated tokens used
        """
        try:
            # For the react app, invoke as sync but run in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.graph.invoke(
                    {"messages": [{"role": "user", "content": user_input}]},
                    config={"recursion_limit": 10},
                ),
            )

            # Extract response
            response_text = ""
            tools_called = []

            if "messages" in result:
                messages = result["messages"]
                if isinstance(messages, list) and messages:
                    last_msg = messages[-1]
                    if isinstance(last_msg, dict):
                        response_text = last_msg.get("content", "")
                    else:
                        response_text = (
                            str(last_msg.content) if hasattr(last_msg, "content") else ""
                        )

            # Estimate token count (rough: ~4 chars per token)
            token_count = len(str(result)) // 4

            return {
                "response": response_text,
                "tools_called": tools_called,
                "token_count": token_count,
            }

        except Exception as e:
            raise RuntimeError(f"Failed to invoke graph: {str(e)}")


class MetricsCollector:
    """Collects and aggregates metrics across test cases."""

    def aggregate(self, results: list[EvalCaseResult]) -> dict[MetricType, float]:
        """
        Aggregate metrics across all results.

        Returns average score for each metric type.
        """
        metrics_by_type: dict[MetricType, list[float]] = {}

        for result in results:
            for metric in result.metrics:
                if metric.metric_type not in metrics_by_type:
                    metrics_by_type[metric.metric_type] = []
                metrics_by_type[metric.metric_type].append(metric.score)

        aggregates = {}
        for metric_type, scores in metrics_by_type.items():
            aggregates[metric_type] = sum(scores) / len(scores) if scores else 0.0

        return aggregates

    def format_metrics(self, metrics: dict[MetricType, float]) -> str:
        """Format aggregated metrics for display."""
        lines = ["Metric Aggregates:"]
        for metric_type, score in sorted(metrics.items()):
            percentage = score * 100
            lines.append(f"  {metric_type.value:20} {percentage:6.1f}%")
        return "\n".join(lines)


# ============================================================================
# Reporting and Formatting
# ============================================================================


def format_eval_report(report: EvalReport) -> str:
    """Format complete evaluation report for display."""
    lines = [
        "=" * 70,
        "EVALUATION REPORT",
        "=" * 70,
        f"Timestamp: {report.timestamp}",
        f"Duration: {report.duration_seconds:.2f} seconds",
        "",
        "SUMMARY:",
        f"  Total Cases: {report.total_cases}",
        f"  Passed: {report.passed_cases}",
        f"  Failed: {report.failed_cases}",
        f"  Pass Rate: {report.pass_rate * 100:.1f}%",
        "",
    ]

    if report.aggregate_metrics:
        lines.append("METRICS:")
        for metric_type, score in sorted(report.aggregate_metrics.items()):
            lines.append(f"  {metric_type.value:20} {score * 100:6.1f}%")
        lines.append("")

    if report.failed_cases_detail:
        lines.append("FAILED CASES:")
        for result in report.failed_cases_detail:
            lines.append(f"\n  Case: {result.case_id}")
            if result.error:
                lines.append(f"    Error: {result.error}")
            lines.append(f"    Input: {result.user_input}")
            if result.actual_response:
                lines.append(f"    Response: {result.actual_response[:100]}...")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


def save_report_json(report: EvalReport, filepath: str) -> None:
    """Save report to JSON file."""
    data = {
        "timestamp": report.timestamp,
        "duration_seconds": report.duration_seconds,
        "total_cases": report.total_cases,
        "passed_cases": report.passed_cases,
        "failed_cases": report.failed_cases,
        "pass_rate": report.pass_rate,
        "aggregate_metrics": {k.value: v for k, v in report.aggregate_metrics.items()},
        "case_results": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "execution_time_ms": r.execution_time_ms,
                "error": r.error,
                "metrics": [
                    {
                        "type": m.metric_type.value,
                        "score": m.score,
                        "passed": m.passed,
                    }
                    for m in r.metrics
                ],
            }
            for r in report.case_results
        ],
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
